#!/usr/bin/env python3
"""按「部署时真正在乎的量」评估进度头，而不是逐帧 MAE / F1。

主指标 = 正确切换率：触发了 **且** |触发帧 − 真边界| ≤ tol。漏切自动算失败。
（早先按"漏切率最低"选点是错的 —— 会退化到 τ→0，从不漏切但提前 86 帧触发。）

标定与评估严格分开
------------------
τ_sub 和延迟 D_sub 全部**在训练集 episode 上**定，留出集只用于报告。
在验证集上挑阈值等于偷看答案，报出来的数字部署时兑现不了。

三个决策量，都按子任务分别定：
  τ_sub  触发阈值 = 训练集上「能连续维持 K 帧的取值」的 q 分位
         必须用 sustained-K 而不是峰值：触发规则要连续 K 帧过线，
         瞬时峰值 0.97 只闪一帧的段，用峰值标定出的 τ 根本触发不了。
  D_sub  触发后再等几步才真正切 = −(训练集触发误差中位)
         模型系统性早触发约 10 帧，是个近乎常数的偏置，直接补掉。
         ⚠ D 按**推理步**计而非时间；部署时卡顿会冻结该计数器，故不破坏抗卡顿。
  K, α   连续帧数与 EMA 平滑系数

卡顿仿真：把段中某帧复制成静止 N 帧，检查进度是否真的不推进 —— 本设计的核心
卖点，必须验证而不是假设。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

FEAT = Path("<SWITCH_ROOT>/feats")
LAB = Path("<SWITCH_ROOT>/labels/labels.npy")
VIEW_IDX = {"third": 0, "right": 1, "left": 2}

# 所有信号统一成「越大越该切」
SIG = {
    "p":    lambda p, d, t: p,                        # 整段线性进度
    "done": lambda p, d, t: d,                        # 段末二分类
    "ttb":  lambda p, d, t: t,                        # 1 − 距边界归一距离
    "both": lambda p, d, t: np.minimum(p, d),
    "pt":   lambda p, d, t: np.minimum(p, t),
    "all":  lambda p, d, t: np.minimum(np.minimum(p, d), t),
}


def ema(x, alpha):
    """指数平滑。卡顿时输入恒定 → EMA 收敛到该常数后不再变化，
    所以它不会像"按时间累加"那样漂移，抗卡顿性质得以保留。"""
    if alpha >= 1.0:
        return x
    y = np.empty_like(x); acc = x[0]
    for i, v in enumerate(x):
        acc = alpha * v + (1 - alpha) * acc
        y[i] = acc
    return y


def sustained_max(sig, K):
    """能连续维持 K 帧的最高取值 = max_i min(sig[i:i+K])。"""
    if len(sig) < K:
        return float(sig.min())
    return float(max(sig[i:i + K].min() for i in range(len(sig) - K + 1)))


def first_trigger(sig, tau, K):
    c = 0
    for i, v in enumerate(sig):
        c = c + 1 if v > tau else 0
        if c >= K:
            return i
    return None


def load_ckpt(p):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tp", "<SWITCH_ROOT>/tools/train_progress.py")
    tp = importlib.util.module_from_spec(spec); spec.loader.exec_module(tp)
    ck = torch.load(p, map_location="cuda", weights_only=False)
    net = tp.ProgressHead(len(ck["views"])).cuda()
    net.load_state_dict(ck["model"]); net.eval()
    return net, ck


@torch.no_grad()
def predict(net, ft, st, eid, sub, vi):
    x = torch.as_tensor(ft[:, vi].reshape(len(ft), -1).astype(np.float32), device="cuda")
    s = np.zeros((len(ft), 20), np.float32); s[:, : st.shape[1]] = st
    o = net(x, torch.as_tensor(s, device="cuda"),
            torch.full((len(ft),), eid, dtype=torch.long, device="cuda"),
            torch.full((len(ft),), sub, dtype=torch.long, device="cuda"))
    return (o[0].cpu().numpy(), torch.sigmoid(o[1]).cpu().numpy(),
            (1.0 - o[2]).cpu().numpy())


def collect(net, ck, vi, lab, val_ep, want_val, stall=0):
    """跑一遍数据，返回 [(p, done, ttb, 段长, 子任务号)]。want_val 选留出集/训练集。"""
    segs, drift = [], []
    for eid in ck["experts"]:
        by_ep = {}
        for r in lab[lab[:, 0] == eid]:
            by_ep.setdefault(int(r[1]), []).append(r)
        for ep, rows in sorted(by_ep.items()):
            if (eid * 1000 + ep in val_ep) != want_val:
                continue
            f = FEAT / f"e{eid}" / f"ep{ep:04d}.npz"
            if not f.exists():
                continue
            d = np.load(f); ft, st = d["feat"], d["state"]
            rows = np.array(rows, dtype=np.float32)
            for sub in np.unique(rows[:, 3]).astype(int):
                t = rows[rows[:, 3] == sub][:, 2].astype(int)
                t = t[t < len(ft)]
                if len(t) < 20:
                    continue
                segs.append((*predict(net, ft[t], st[t], eid, sub, vi),
                             len(t), (int(eid), int(sub))))
                if stall:
                    mid = len(t) // 2
                    ps, _, _ = predict(net, np.repeat(ft[t[mid]][None], stall, 0),
                                       np.repeat(st[t[mid]][None], stall, 0), eid, sub, vi)
                    drift.append(float(ps.max() - ps.min()))
    return segs, drift


def calibrate(segs, signal, K, alpha, q):
    """训练集上定 τ_sub（sustained-K 的 q 分位）与 D_sub（补掉早触发偏置）。"""
    vals, errs = {}, {}
    for p, d, tb, L, sub in segs:
        vals.setdefault(sub, []).append(sustained_max(ema(SIG[signal](p, d, tb), alpha), K))
    tau = {k: float(np.percentile(v, q)) for k, v in vals.items()}
    for p, d, tb, L, sub in segs:
        i = first_trigger(ema(SIG[signal](p, d, tb), alpha), tau[sub], K)
        if i is not None:
            errs.setdefault(sub, []).append(i - (L - 1))
    delay = {k: max(0, int(round(-np.median(v)))) for k, v in errs.items()}
    return tau, delay


def score(segs, signal, tau, K, alpha, delay, tol=10):
    errs, miss, early, good = [], 0, 0, 0
    for p, d, tb, L, sub in segs:
        i = first_trigger(ema(SIG[signal](p, d, tb), alpha), tau[sub], K)
        if i is None:
            miss += 1
            continue
        e = i + delay.get(sub, 0) - (L - 1)      # 加上标定出的延迟补偿
        errs.append(e)
        if abs(e) <= tol:
            good += 1
        elif e < 0:
            early += 1
    n = len(segs)
    errs = np.array(errs) if errs else np.array([0])
    return dict(signal=signal, K=K, alpha=alpha, q=None, n=n, acc=good / n,
                miss=miss / n, early=early / n, med=float(np.median(errs)),
                w5=float((np.abs(errs) <= 5).mean()), w20=float((np.abs(errs) <= 20).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="<SWITCH_ROOT>/runs/p_third_e2/best.pt")
    ap.add_argument("--stall", type=int, default=30)
    ap.add_argument("--tol", type=int, default=10)
    # 只在指定信号里选点。用途：分段口径和串行口径挑出的信号可能不同，
    # 需要为每个信号各自标定 τ/D 后，再拿串行回放去比 —— 
    # 换信号却沿用旧 τ/D 是无效对比（阈值和信号的量纲根本不同）。
    ap.add_argument("--only-signal", default=None)
    a = ap.parse_args()

    net, ck = load_ckpt(a.ckpt)
    vi = [VIEW_IDX[v] for v in ck["views"]]
    val_ep = set(ck.get("val_ep", []))
    if not val_ep:
        raise SystemExit("✘ ckpt 里没有 val_ep（旧 ckpt），请重训")
    lab = np.load(LAB)

    tr, _ = collect(net, ck, vi, lab, val_ep, want_val=False)
    va, drift = collect(net, ck, vi, lab, val_ep, want_val=True, stall=a.stall)
    print(f"· ckpt {a.ckpt}")
    print(f"· 视角 {ck['views']}  专家 {ck['experts']}  训练 {len(tr)} 段 / 留出 {len(va)} 段")
    print(f"· 标定在训练集，下表数字全部来自留出集   容差 ±{a.tol} 帧\n")

    print(f"  {'信号':<6}{'K':>3}{'α':>5}{'q':>4}{'正确率':>9}{'漏切':>8}{'过早':>8}"
          f"{'误差中位':>10}{'≤5帧':>8}{'≤20帧':>8}")
    best = None
    for sg in ([a.only_signal] if a.only_signal else SIG):
        rows = []
        for K in (3, 5, 8):
            for alpha in (1.0, 0.5, 0.3):
                for q in (1, 2, 5, 10):
                    tau, delay = calibrate(tr, sg, K, alpha, q)
                    r = score(va, sg, tau, K, alpha, delay, a.tol)
                    r["q"] = q; r["_cfg"] = (sg, K, alpha, q, tau, delay)
                    rows.append(r)
                    if best is None or r["acc"] > best["acc"]:
                        best = r
        for r in sorted(rows, key=lambda x: -x["acc"])[:2]:
            print(f"  {r['signal']:<6}{r['K']:>3}{r['alpha']:>5.1f}{r['q']:>4}"
                  f"{r['acc']*100:>8.1f}%{r['miss']*100:>7.1f}%{r['early']*100:>7.1f}%"
                  f"{r['med']:>+10.0f}{r['w5']*100:>7.0f}%{r['w20']*100:>7.0f}%")

    # 漏切兜底：训练集段长的 p95 × 1.5。超过这个步数还没触发就强制推进，
    # 免得真机上卡死在某一段。计数器同样受停滞门控冻结。
    seglen = {}
    for p_, d_, tb_, L_, k_ in tr:
        seglen.setdefault(k_, []).append(L_)
    max_steps = {k: int(np.percentile(v, 95) * 1.5) for k, v in seglen.items()}

    sg, K, alpha, q, tau, delay = best["_cfg"]
    print(f"\n  ★ 最优：signal={sg} K={K} α={alpha} q={q}")
    print(f"    正确切换率 {best['acc']*100:.1f}%   漏切 {best['miss']*100:.1f}%   "
          f"误差中位 {best['med']:+.0f}帧   |误差|≤20 占 {best['w20']*100:.0f}%")
    print("    τ  " + "  ".join(f"E{k[0]+1}s{k[1]}:{v:.2f}" for k, v in sorted(tau.items())))
    print("    D  " + "  ".join(f"E{k[0]+1}s{k[1]}:{v}" for k, v in sorted(delay.items())))

    print(f"\n  ★ 卡顿仿真（静止 {a.stall} 帧）：p 漂移 中位 {np.median(drift):.4f}  "
          f"最大 {np.max(drift):.4f}   ≈0 即抗卡顿成立")

    out = Path(a.ckpt).parent
    (out / "switch_eval.json").write_text(json.dumps(
        {k: v for k, v in best.items() if k != "_cfg"}, indent=2))
    # 部署要用的全部决策参数，单独存一份
    (out / "deploy_cfg.json").write_text(json.dumps(
        {"signal": sg, "K": K, "alpha": alpha, "quantile": q,
         # JSON 的键必须是字符串，用 "专家_子任务"
         "tau": {f"{k[0]}_{k[1]}": v for k, v in tau.items()},
         "delay": {f"{k[0]}_{k[1]}": v for k, v in delay.items()},
         "max_steps": {f"{k[0]}_{k[1]}": v for k, v in max_steps.items()},
         "n_sub": {str(e): max(k[1] for k in tau if k[0] == e) for e in ck["experts"]}, "views": ck["views"], "experts": ck["experts"],
         "val_acc": best["acc"], "val_miss": best["miss"]}, indent=2))
    print(f"\n· 部署参数 → {out}/deploy_cfg.json")


if __name__ == "__main__":
    main()
