#!/usr/bin/env python3
"""训子任务进度头：在缓存好的 SigLIP2 特征上训一个小 MLP。

抗卡顿是这个设计的第一约束
--------------------------
部署时网络卡顿会让机械臂停在原地。所以：
  · 输入**只有当前帧**——不喂帧号、不喂已执行时长、不用任何时序模块。
    卡住时输入不变 → 输出不变 → 进度不推进（这是天然的，不是靠规则打补丁）。
  · 因此这里也**不做时序平滑**。平滑要放到部署侧，且必须带停滞门控。

为什么默认只用第三视角
--------------------
进度是**世界状态**的函数（盒子进箱没有、盖子合没合），第三视角是固定机位，
同一世界状态映射到相近画面；腕部相机跟着手臂动，同一状态因姿态不同差别巨大，
100 集不够学会忽略这种干扰。更要命的是腕部画面与手臂位姿强相关 ——
模型会从中读出"走到轨迹第几步"这种**时间性线索**，正是抗卡顿要避免的。
`--views` 可以切换，用来做对照实验。

用法:
  train_progress.py --views third            # 主方案
  train_progress.py --views third,right,left # 三视角对照
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

FEAT = Path("<SWITCH_ROOT>/feats")
LAB = Path("<SWITCH_ROOT>/labels/labels.npy")
VIEW_IDX = {"third": 0, "right": 1, "left": 2}
STATE_DIM = {0: 10, 1: 16, 2: 20, 3: 20}   # E1 EE / E2 joint / E3 / E4
N_SUB = {0: 2, 1: 2, 2: 7, 3: 7}           # 各专家的子任务数
TTB_W = 30                                 # 「距边界」的关注窗口（帧）


class ProgressHead(nn.Module):
    """图像特征 + state + (expert, subtask) 条件 → p_t。

    state 按专家各配一个编码器：三个专家维度不同(10/16/20)，且同一下标语义不同，
    padding 成同一张量会互相污染，不如各编各的。
    """

    def __init__(self, n_view, d_img=1152, d_hid=512, d_cond=64):
        super().__init__()
        self.img = nn.Sequential(nn.LayerNorm(d_img * n_view),
                                 nn.Linear(d_img * n_view, d_hid), nn.GELU())
        self.state_enc = nn.ModuleDict({
            str(k): nn.Sequential(nn.LayerNorm(v), nn.Linear(v, 128), nn.GELU())
            for k, v in STATE_DIM.items()})
        self.emb_expert = nn.Embedding(len(STATE_DIM), d_cond)
        self.emb_sub = nn.Embedding(max(N_SUB.values()) + 1, d_cond)
        self.trunk = nn.Sequential(
            nn.Linear(d_hid + 128 + 2 * d_cond, d_hid), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d_hid, 256), nn.GELU())
        self.head_p = nn.Linear(256, 1)       # 进度回归（0→1，整段线性）
        self.head_done = nn.Linear(256, 1)    # 「这一段结束了」二分类
        # ★ 距边界还有多远：ttb = clip((end - t) / TTB_W, 0, 1)
        #   线性 p_t 把容量摊在整段上，可决策只发生在段末几十帧 ——
        #   段中 40% 和 60% 分不清对切换毫无影响，却占了大部分 loss。
        #   ttb 在离边界 >TTB_W 帧处恒为 1（无梯度压力），容量全给边界附近。
        self.head_ttb = nn.Linear(256, 1)

    def forward(self, img, state, eid, sub):
        h = self.img(img)
        s = torch.zeros(img.shape[0], 128, device=img.device, dtype=h.dtype)
        for k in STATE_DIM:                    # 按专家分组过各自的编码器
            m = eid == k
            if m.any():
                s[m] = self.state_enc[str(k)](state[m, :STATE_DIM[k]]).to(h.dtype)
        z = self.trunk(torch.cat([h, s, self.emb_expert(eid), self.emb_sub(sub)], -1))
        return (self.head_p(z).squeeze(-1), self.head_done(z).squeeze(-1),
                self.head_ttb(z).squeeze(-1))


def load(views, experts, feat_dirs=(FEAT,)):
    """把缓存特征与标签对齐成扁平样本。按 episode 划分 train/val，不按帧 ——
    同一集相邻帧几乎一样，按帧划分会让验证集泄漏，指标虚高。"""
    lab = np.load(LAB)
    vi = [VIEW_IDX[v] for v in views]
    X, S, E, U, P, D, EP, T, SRC = [], [], [], [], [], [], [], [], []
    for di, fdir in enumerate(feat_dirs):
      for eid in experts:
          m = lab[:, 0] == eid
          sub = lab[m]
          by_ep = {}
          for r in sub:
              by_ep.setdefault(int(r[1]), []).append(r)
          for ep, rows in sorted(by_ep.items()):
              f = fdir / f"e{eid}" / f"ep{ep:04d}.npz"
              if not f.exists():
                  continue
              d = np.load(f)
              ft, st = d["feat"], d["state"]
              rows = np.array(rows, dtype=np.float32)
              t = rows[:, 2].astype(int)
              t = t[t < len(ft)]
              rows = rows[: len(t)]
              X.append(ft[t][:, vi].reshape(len(t), -1).astype(np.float32))
              s = np.zeros((len(t), max(STATE_DIM.values())), np.float32)
              s[:, : st.shape[1]] = st[t]
              S.append(s)
              E.append(np.full(len(t), eid, np.int64))
              U.append(rows[:, 3].astype(np.int64))
              P.append(rows[:, 6]); D.append(rows[:, 7])
              # ttb：同一 (episode, subtask) 内，段末 = 该组最大帧号
              ttb = np.ones(len(t), np.float32)
              su = rows[:, 3]
              for k in np.unique(su):
                  m2 = su == k
                  ttb[m2] = np.clip((t[m2].max() - t[m2]) / TTB_W, 0, 1)
              T.append(ttb)
              SRC.append(np.full(len(t), di, np.int64))   # 0=干净 1..=增强
              EP.append(np.full(len(t), eid * 1000 + ep, np.int64))
    if not X:
        raise SystemExit("✘ 没有可用样本：特征还没缓存完？")
    return (np.concatenate(X), np.concatenate(S), np.concatenate(E),
            np.concatenate(U), np.concatenate(P), np.concatenate(D),
            np.concatenate(EP), np.concatenate(T), np.concatenate(SRC))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default="third")
    ap.add_argument("--experts", default="2")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam-done", type=float, default=1.0)
    ap.add_argument("--lam-mono", type=float, default=0.1)
    ap.add_argument("--lam-ttb", type=float, default=2.0)
    ap.add_argument("--feats", default=str(FEAT),
                    help="逗号分隔的特征目录；第一个必须是干净特征（验证只在它上面算）")
    ap.add_argument("--tag", default="", help="附加到输出目录名，区分不同配置")
    ap.add_argument("--drop", default="", choices=["", "img", "state"],
                    help="消融：img=图像置零只留 state；state=state 置零只留图像。"
                         "用来分解「进度到底靠什么判断」")
    ap.add_argument("--seed", type=int, default=0,
                    help="只改权重初始化与批次顺序；**不改** train/val 划分 —— "
                         "集成成员必须共用同一留出集，否则标定和评估会串味")
    ap.add_argument("--out", default="<SWITCH_ROOT>/runs")
    a = ap.parse_args()
    views = a.views.split(","); experts = [int(x) for x in a.experts.split(",")]
    feat_dirs = [Path(x) for x in a.feats.split(",")]
    out = Path(a.out) / (f"p_{'_'.join(views)}_e{''.join(map(str, experts))}" + a.tag)
    out.mkdir(parents=True, exist_ok=True)

    X, S, E, U, P, D, EP, TB, SRC = load(views, experts, feat_dirs)
    if a.drop == "img":
        X[:] = 0.0        # 只留 state：回答「靠末端位姿/关节角能判断到什么程度」
    elif a.drop == "state":
        S[:] = 0.0        # 只留图像
    eps = np.unique(EP)
    rng = np.random.default_rng(0); rng.shuffle(eps)
    n_val = max(1, int(len(eps) * 0.2))
    val_ep = set(eps[:n_val].tolist())
    # 划分要点：① 同一集的干净版与增强版必须落在同一侧，否则增强版会把留出集的
    # 内容泄漏进训练；EP 键不含来源，天然满足。② 验证**只在干净特征上算**，
    # 否则和之前的数字口径不同、没法比较。
    in_val = np.isin(EP, list(val_ep))
    vm = in_val & (SRC == 0)
    tm = ~in_val
    print(f"· 样本 {len(X)} 帧（{len(feat_dirs)} 份特征）  "
          f"训练 {tm.sum()} / 验证 {vm.sum()}（仅干净）  "
          f"（按 episode 划分，{len(eps)-n_val}/{n_val} 集）")
    print(f"· 视角 {views}  专家 {experts}  done 正样本 {D.mean()*100:.1f}%")

    dev = "cuda"
    to = lambda x, t=torch.float32: torch.as_tensor(x, dtype=t, device=dev)
    # ★ 图像特征常驻 CPU，按批送 GPU。网格池化版每视角 4×1152，三专家合计 9.2 GB，
    #   整块塞显存会和占卡进程撞 OOM；而且训练本来就只需要一个 batch。
    Xt = torch.as_tensor(X)                       # CPU
    St, Et, Ut, Pt, Dt = to(S), to(E, torch.long), to(U, torch.long), to(P), to(D)
    Tt = to(TB)
    tr = torch.where(torch.as_tensor(tm))[0]      # 索引也留 CPU
    va = torch.where(torch.as_tensor(vm))[0]

    torch.manual_seed(a.seed)          # 划分用的 rng 在上面、种子固定为 0，不受这里影响
    # 单视角特征维度从数据推断：网格池化版是 1152*G*G，不能写死
    d_img = X.shape[1] // len(views)
    net = ProgressHead(len(views), d_img=d_img).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    pos_w = torch.tensor([(1 - D.mean()) / max(D.mean(), 1e-6)], device=dev)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    best = 1e9

    for ep in range(a.epochs):
        net.train(); perm = tr[torch.randperm(len(tr))]
        tot = 0.0
        for i in range(0, len(perm), a.bs):
            b = perm[i:i + a.bs]
            xb = Xt[b].to(dev, non_blocking=True)
            bd = b.to(dev)
            p, dn, tb = net(xb, St[bd], Et[bd], Ut[bd])
            # Smooth-L1 比 MSE 抗末段标注噪声（子任务边界本身有 ±几帧的不确定）
            l_p = nn.functional.smooth_l1_loss(p, Pt[bd], beta=0.1)
            l_d = bce(dn, Dt[bd])
            # ttb 用较小的 beta：边界附近要精确，远处本来就恒为 1
            l_t = nn.functional.smooth_l1_loss(tb, Tt[bd], beta=0.05)
            # 单调性：同一段内 p 不该回退。按标签顺序近似取相邻对。
            o = torch.argsort(Pt[bd])
            dp = p[o][1:] - p[o][:-1]
            l_m = torch.clamp(-dp, min=0).mean()
            loss = l_p + a.lam_done * l_d + a.lam_mono * l_m + a.lam_ttb * l_t
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        sch.step()

        net.eval()
        with torch.no_grad():
            # 验证也要分块，否则同样会 OOM
            ps, ds_, ts_ = [], [], []
            for j in range(0, len(va), 4096):
                vb = va[j:j + 4096]; vd = vb.to(dev)
                o = net(Xt[vb].to(dev), St[vd], Et[vd], Ut[vd])
                ps.append(o[0]); ds_.append(o[1]); ts_.append(o[2])
            p, dn, tb = torch.cat(ps), torch.cat(ds_), torch.cat(ts_)
            vad = va.to(dev)
            mae = (p - Pt[vad]).abs().mean().item()
            # 只看边界窗口内的 ttb 误差 —— 远处恒 1 太好预测，混进去会稀释掉信息
            nb = Tt[vad] < 1.0
            tmae = (tb[nb] - Tt[vad][nb]).abs().mean().item() * TTB_W
            pr = torch.sigmoid(dn)
            tp = ((pr > 0.5) & (Dt[vad] > 0.5)).sum().item()
            fp = ((pr > 0.5) & (Dt[vad] < 0.5)).sum().item()
            fn = ((pr < 0.5) & (Dt[vad] > 0.5)).sum().item()
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        sel = mae + tmae / TTB_W          # 选点看进度和边界两件事
        if sel < best:
            best = sel
            torch.save({"model": net.state_dict(), "views": views, "experts": experts,
                        "state_dim": STATE_DIM, "n_sub": N_SUB, "ttb_w": TTB_W,
                        "d_img": d_img,
                        # 存下验证集 episode，评估只能在留出集上做，否则指标虚高
                        "val_ep": sorted(int(x) for x in val_ep)}, out / "best.pt")
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f"  ep{ep:3d}  train {tot/len(perm):.4f}   val: progress MAE {mae:.4f}   "
                  f"done F1 {f1:.3f}   边界窗内 ttb 误差 {tmae:.1f}帧")

    (out / "result.json").write_text(json.dumps(
        {"views": views, "experts": experts, "best_val_sel": best,
         "n_train": int(tm.sum()), "n_val": int(vm.sum()),
         "feat_dirs": [str(x) for x in feat_dirs]}, indent=2))
    print(f"\n· 最好 val 选点分 = {best:.4f} → {out}/best.pt")


if __name__ == "__main__":
    main()
