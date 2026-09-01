#!/usr/bin/env python3
"""第二层：专家间切换的图像分类器（E1 / E2 / E3 三分类）。

和子任务进度头是**不同性质的问题**
----------------------------------
进度头问「当前这段动作做完没有」；这里问「现在该由哪个专家接管」。
后者是纯世界状态判断：箱子装满了没有、盖子该合了没有。

同样只吃当前帧（抗卡顿的理由完全一样），同样冻结 SigLIP2 视觉塔。

⚠ 这个训练目标有个内在的**混淆**，必须在评估里拆开看：
   E1 集和 E2 集的区别，很大程度上就是「哪只手臂在动」——
   这个区别模型学起来毫不费力，但对「什么时候该切」毫无用处。
   真正有用的是 E1/E2 ↔ E3 的区别（箱子装满 → 该封箱了）。
   所以除了逐帧准确率，还要看**混淆矩阵**和**时序剖面**。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

FEAT = Path("<SWITCH_ROOT>/feats")
LAB = Path("<SWITCH_ROOT>/labels/labels.npy")
VIEW_IDX = {"third": 0, "right": 1, "left": 2}
NAME = {0: "E1 右臂", 1: "E2 左臂", 2: "E3 封箱"}


class ExpertCls(nn.Module):
    def __init__(self, n_view, d_img=1152, d_hid=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_img * n_view), nn.Linear(d_img * n_view, d_hid), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(d_hid, 256), nn.GELU(), nn.Linear(256, 3))

    def forward(self, x):
        return self.net(x)


def load(views):
    """只用图像，不喂 state —— state 的维度本身就泄漏专家身份
    （E1 是 10 维 EE、E2 是 16 维关节、E3 是 20 维），喂了等于直接告诉它答案。"""
    lab = np.load(LAB)
    vi = [VIEW_IDX[v] for v in views]
    X, Y, EP, FR, NF = [], [], [], [], []
    for eid in (0, 1, 2):
        by_ep = {}
        for r in lab[lab[:, 0] == eid]:
            by_ep.setdefault(int(r[1]), []).append(r)
        for ep, rows in sorted(by_ep.items()):
            f = FEAT / f"e{eid}" / f"ep{ep:04d}.npz"
            if not f.exists():
                continue
            ft = np.load(f)["feat"]
            rows = np.array(rows, dtype=np.float32)
            t = rows[:, 2].astype(int); t = t[t < len(ft)]
            X.append(ft[t][:, vi].reshape(len(t), -1).astype(np.float32))
            Y.append(np.full(len(t), eid, np.int64))
            EP.append(np.full(len(t), eid * 1000 + ep, np.int64))
            FR.append(t); NF.append(np.full(len(t), len(t), np.int64))
    return (np.concatenate(X), np.concatenate(Y), np.concatenate(EP),
            np.concatenate(FR), np.concatenate(NF))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default="third,right,left")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--out", default="<SWITCH_ROOT>/runs")
    a = ap.parse_args()
    views = a.views.split(",")
    out = Path(a.out) / f"cls_{'_'.join(views)}"
    out.mkdir(parents=True, exist_ok=True)

    X, Y, EP, FR, NF = load(views)
    eps = np.unique(EP)
    rng = np.random.default_rng(0); rng.shuffle(eps)
    val_ep = set(eps[:max(1, int(len(eps) * 0.2))].tolist())
    vm = np.isin(EP, list(val_ep))
    print(f"· 样本 {len(X)} 帧   训练 {(~vm).sum()} / 留出 {vm.sum()}"
          f"   （按 episode 划分，{len(eps)-len(val_ep)}/{len(val_ep)} 集）")

    dev = "cuda"
    Xt = torch.as_tensor(X, device=dev); Yt = torch.as_tensor(Y, device=dev)
    tr = torch.where(torch.as_tensor(~vm, device=dev))[0]
    va = torch.where(torch.as_tensor(vm, device=dev))[0]
    net = ExpertCls(len(views)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    ce = nn.CrossEntropyLoss()
    best = 0.0
    for ep in range(a.epochs):
        net.train(); perm = tr[torch.randperm(len(tr), device=dev)]
        for i in range(0, len(perm), a.bs):
            b = perm[i:i + a.bs]
            loss = ce(net(Xt[b]), Yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        net.eval()
        with torch.no_grad():
            acc = (net(Xt[va]).argmax(1) == Yt[va]).float().mean().item()
        if acc > best:
            best = acc
            torch.save({"model": net.state_dict(), "views": views,
                        "val_ep": sorted(int(x) for x in val_ep)}, out / "best.pt")
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f"  ep{ep:3d}  val 逐帧准确率 {acc*100:.1f}%")

    # ── 混淆矩阵：拆开「E1↔E2（没用的区分）」和「E1E2↔E3（有用的区分）」 ──
    ck = torch.load(out / "best.pt", map_location=dev, weights_only=False)
    net.load_state_dict(ck["model"]); net.eval()
    with torch.no_grad():
        pr = net(Xt[va]).argmax(1).cpu().numpy()
    gt = Y[vm]
    cm = np.zeros((3, 3), int)
    for g, p in zip(gt, pr):
        cm[g, p] += 1
    print(f"\n  混淆矩阵（行=真实，列=预测，按行归一）")
    print(f"  {'':<10}{'→E1':>8}{'→E2':>8}{'→E3':>8}")
    for i in range(3):
        r = cm[i] / max(cm[i].sum(), 1)
        print(f"  {NAME[i]:<10}{r[0]*100:>7.1f}%{r[1]*100:>7.1f}%{r[2]*100:>7.1f}%")
    binm = ((gt == 2) == (pr == 2)).mean()
    print(f"\n  ★ 「该不该封箱了」二分类准确率 {binm*100:.1f}%  ← 这才是有用的那个区分")
    print(f"    E1↔E2 互相混淆 {(cm[0,1]+cm[1,0])/max(cm[0].sum()+cm[1].sum(),1)*100:.1f}%"
          f"   （这个区分对「何时切」没用，混了也不要紧）")

    # ── 时序剖面：E1/E2 集在末尾是否翻向 E3 ──
    print(f"\n  时序剖面：各集按进度分 10 档，看预测为 E3 的比例")
    with torch.no_grad():
        prob = torch.softmax(net(Xt[va]), 1).cpu().numpy()
    fr, nf = FR[vm], NF[vm]
    bins = np.clip((fr / np.maximum(nf - 1, 1) * 10).astype(int), 0, 9)
    print(f"  {'真实':<10}" + "".join(f"{i*10:>6}%" for i in range(10)))
    for eid in (0, 1, 2):
        m = gt == eid
        row = [prob[m & (bins == b), 2].mean() * 100 if (m & (bins == b)).any() else np.nan
               for b in range(10)]
        print(f"  {NAME[eid]:<10}" + "".join(f"{v:>6.0f}%" for v in row))
    print(f"    E1/E2 行若在末尾抬升 → 分类器能看出「这一趟放完了、箱子更满了」")

    (out / "result.json").write_text(json.dumps(
        {"views": views, "frame_acc": best, "close_binary_acc": float(binm)}, indent=2))
    print(f"\n· → {out}/best.pt")


if __name__ == "__main__":
    main()
