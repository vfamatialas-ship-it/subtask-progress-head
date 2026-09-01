#!/usr/bin/env python3
"""多头集成：平均若干个不同随机种子的进度头，按**串行口径**评估。

为什么值得试
------------
头本身只有几 MB、训一个 2 分钟，部署侧多跑两次 MLP 相对 SigLIP 的 7ms 可以忽略。
而集成对「边界附近抖动」这类方差型误差最有效 —— 我们的误差正是这一类
（误差中位只有 −2 帧，但 24% 的切换偏出 ±10 帧，是散布不是偏置）。

评估要点
--------
· τ/D 必须在**集成后的输出**上重新标定，不能沿用单头的（分布变了）
· 标定在训练集 episode，评估在留出集
· 用串行口径（整集跑状态机，误差累积），不是分段口径
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("<SWITCH_ROOT>")
sys.path.insert(0, str(ROOT / "deploy_aug"))
import switcher as S                                          # noqa: E402

# --g2: 用网格池化特征（每视角 4×1152）而不是全局向量
FEATDIR = ROOT / ("feats_g2" if "--g2" in sys.argv else "feats")
RUNS = [x for x in sys.argv[1:] if not x.startswith("--")] or \
    [str(ROOT / f"runs/p_third_right_left_e012_aug_s{i}") for i in range(3)]
VIEW_IDX = {"third": 0, "right": 1, "left": 2}
K, ALPHA, Q, LOCKOUT = 8, 0.5, 1, 10
SIGNAL = "all"

nets, ck = [], None
for r in RUNS:
    c = torch.load(Path(r) / "best.pt", map_location="cuda", weights_only=False)
    # d_img 从 ckpt 取：网格池化版每视角 4×1152，写死 1152 会加载失败
    n = S.ProgressHead(len(c["views"]), d_img=c.get("d_img", 1152)).cuda()
    n.load_state_dict(c["model"]); n.eval()
    nets.append(n); ck = c
vi = [VIEW_IDX[v] for v in ck["views"]]
val_ep = set(ck["val_ep"])
lab = np.load(ROOT / "labels/labels.npy")
print(f"· 集成 {len(nets)} 个头   视角 {ck['views']}   信号 {SIGNAL} K={K} α={ALPHA} q={Q}")


@torch.no_grad()
def predict(ft, st, eid, sub, use=None):
    """use=None 用全部头（集成）；use=i 只用第 i 个（单头对照）。"""
    x = torch.as_tensor(ft[:, vi].reshape(len(ft), -1).astype(np.float32), device="cuda")
    s = np.zeros((len(ft), 20), np.float32); s[:, :st.shape[1]] = st
    s = torch.as_tensor(s, device="cuda")
    e = torch.full((len(ft),), eid, dtype=torch.long, device="cuda")
    u = torch.full((len(ft),), sub, dtype=torch.long, device="cuda")
    outs = [nets[use]] if use is not None else nets
    ps, ds, ts = [], [], []
    for n in outs:
        o = n(x, s, e, u)
        ps.append(o[0]); ds.append(torch.sigmoid(o[1])); ts.append(1.0 - o[2])
    P = torch.stack(ps).mean(0).cpu().numpy()
    D = torch.stack(ds).mean(0).cpu().numpy()
    T = torch.stack(ts).mean(0).cpu().numpy()
    return np.minimum(np.minimum(P, D), T)      # SIGNAL="all"


def ema(x, a):
    y = np.empty_like(x); acc = x[0]
    for i, v in enumerate(x):
        acc = a * v + (1 - a) * acc; y[i] = acc
    return y


def sustained_max(sig, k):
    return float(max(sig[i:i + k].min() for i in range(len(sig) - k + 1))) if len(sig) >= k \
        else float(sig.min())


def episodes(want_val):
    for eid in ck["experts"]:
        by = {}
        for r in lab[lab[:, 0] == eid]:
            by.setdefault(int(r[1]), []).append(r)
        for ep, rows in sorted(by.items()):
            if (eid * 1000 + ep in val_ep) != want_val:
                continue
            f = FEATDIR / f"e{eid}/ep{ep:04d}.npz"
            if not f.exists():
                continue
            d = np.load(f)
            yield eid, ep, d["feat"], d["state"], np.array(rows, dtype=np.float32)


def run(use=None):
    # ① 训练集上标定 τ 与延迟 D（集成后的输出，分布和单头不同，必须重标）
    vals, errs = {}, {}
    for eid, ep, ft, st, rows in episodes(False):
        for sub in np.unique(rows[:, 3]).astype(int):
            t = rows[rows[:, 3] == sub][:, 2].astype(int); t = t[t < len(ft)]
            if len(t) < 20: continue
            sig = ema(predict(ft[t], st[t], eid, sub, use), ALPHA)
            vals.setdefault((eid, sub), []).append(sustained_max(sig, K))
    tau = {k: float(np.percentile(v, Q)) for k, v in vals.items()}
    for eid, ep, ft, st, rows in episodes(False):
        for sub in np.unique(rows[:, 3]).astype(int):
            t = rows[rows[:, 3] == sub][:, 2].astype(int); t = t[t < len(ft)]
            if len(t) < 20: continue
            sig = ema(predict(ft[t], st[t], eid, sub, use), ALPHA)
            c = 0
            for i, v in enumerate(sig):
                c = c + 1 if v > tau[(eid, sub)] else 0
                if c >= K:
                    errs.setdefault((eid, sub), []).append(i - (len(t) - 1)); break
    delay = {k: max(0, int(round(-np.median(v)))) for k, v in errs.items()}
    maxst = {}
    for eid, ep, ft, st, rows in episodes(False):
        for sub in np.unique(rows[:, 3]).astype(int):
            t = rows[rows[:, 3] == sub][:, 2].astype(int)
            maxst.setdefault((eid, sub), []).append(len(t))
    maxst = {k: int(np.percentile(v, 95) * 1.5) for k, v in maxst.items()}

    # ② 留出集串行回放
    out, done, tot = [], 0, 0
    for eid, ep, ft, st, rows in episodes(True):
        truth = {int(s_): int(rows[rows[:, 3] == s_][:, 2].max()) for s_ in np.unique(rows[:, 3])}
        n_sub = max(truth)
        sub, emav, hits, pend, steps, since, fin = 1, None, 0, None, 0, 10**9, False
        tot += 1
        for t in list(range(len(ft))) + [len(ft) - 1] * 60:
            if fin: break
            raw = float(predict(ft[t:t+1], st[t:t+1], eid, sub, use)[0])
            tt = tau.get((eid, sub), 0.9)
            emav = raw if emav is None else ALPHA * raw + (1 - ALPHA) * emav
            steps += 1; since += 1
            hits = hits + 1 if emav > tt else 0
            if pend is not None: pend -= 1
            if pend is None and hits >= K and since >= LOCKOUT:
                pend = delay.get((eid, sub), 0)
            if pend is None and steps > maxst.get((eid, sub), 10**9): pend = 0
            if pend is not None and pend <= 0:
                b = truth.get(sub)
                if b is not None: out.append(t - b)
                if sub >= n_sub: fin = True
                else:
                    sub += 1; emav = None; hits = 0; steps = 0; since = 0
                pend = None
        if fin: done += 1
    e = np.array(out)
    return done, tot, (np.abs(e) <= 10).mean() * 100, (np.abs(e) <= 20).mean() * 100, np.median(e)


print(f"\n  {'配置':<14}{'跑完整集':>10}{'≤10帧':>8}{'≤20帧':>8}{'误差中位':>10}")
for i in range(len(nets)):
    d_, t_, w10, w20, md = run(use=i)
    print(f"  {'单头 seed'+str(i):<14}{d_}/{t_:<8}{w10:>7.0f}%{w20:>7.0f}%{md:>+10.0f}")
d_, t_, w10, w20, md = run()
print(f"  {'集成 ×'+str(len(nets)):<14}{d_}/{t_:<8}{w10:>7.0f}%{w20:>7.0f}%{md:>+10.0f}")
