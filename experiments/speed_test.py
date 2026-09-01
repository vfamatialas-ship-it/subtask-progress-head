#!/usr/bin/env python3
"""速度鲁棒性：部署时机械臂快慢和采集时不一样，切换还准不准。

为什么必须测
------------
进度标签是按**采集时间**线性插值出来的（p_t = (t−起点)/(终点−起点)）。
如果模型学到的是「时间」而不是「场景」，那部署时速度一变就全乱 ——
这是个会直接毁掉整套方案的风险，不能靠"我们没喂时间所以应该没事"这种推理带过。

测法
----
把整集**重采样**成不同速度再跑状态机：
  s = 0.5  慢放（每帧复制一份）→ 手臂每步走的距离减半
  s = 2.0  快放（隔帧取）      → 手臂每步走的距离翻倍
真边界在重采样后的下标空间里变成 boundary / s，据此算误差。

两个量要分开看：
  · **触发时的视觉位置**（误差 ÷ 该段重采样后长度）—— 模型认不认得出「做到哪了」
  · **绝对帧误差** —— 受延迟补偿 D 影响，D 是按推理步计的，慢放时会补偿不足
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("<SWITCH_ROOT>")
sys.path.insert(0, str(ROOT / "deploy_aug"))
import switcher as S                                      # noqa: E402

RUN = ROOT / "runs/p_third_right_left_e0123_e0123_aug"   # 出货的四专家权重
PKG = ROOT / "deploy4"
VIEW_IDX = {"third": 0, "right": 1, "left": 2}

import json                                               # noqa: E402
cfg = json.loads((PKG / "deploy_cfg.json").read_text())
ck = torch.load(RUN / "best.pt", map_location="cuda", weights_only=False)
SD = {int(k): int(v) for k, v in ck["state_dim"].items()}
net = S.ProgressHead(len(ck["views"]), state_dim=SD).cuda()
net.load_state_dict(ck["model"]); net.eval()
vi = [VIEW_IDX[v] for v in ck["views"]]
val_ep = set(ck["val_ep"])
lab = np.load(ROOT / "labels/labels.npy")
K, A, LOCK = cfg["K"], cfg["alpha"], 10
N_SUB = {int(k): v for k, v in cfg["n_sub"].items()}


@torch.no_grad()
def sig_of(ft_row, st_row, eid, sub):
    x = torch.as_tensor(ft_row[vi].reshape(1, -1).astype(np.float32), device="cuda")
    s = torch.zeros(1, 20, device="cuda")
    d = SD[eid]
    s[0, :d] = torch.as_tensor(st_row[:d], device="cuda")
    o = net(x, s, torch.tensor([eid], device="cuda"), torch.tensor([sub], device="cuda"))
    p, dn, tb = float(o[0]), float(torch.sigmoid(o[1])), float(1.0 - o[2])
    return min(p, dn, tb)                                   # signal = "all"


# ── 训练速度基准：每个 (专家,子任务) 在原速下「信号每步涨多少」──
#    部署时实测同一个量，比值就是速度比，用它缩放 D。
#    D 之所以要缩放：它补的是「模型系统性早触发的那段**距离**」，
#    而固定步数在慢放时只能覆盖一半距离。
def measure_rate(speed=1.0):
    rate = {}
    for eid in ck["experts"]:
        by = {}
        for r in lab[lab[:, 0] == eid]:
            by.setdefault(int(r[1]), []).append(r)
        for ep, rows in sorted(by.items()):
            if eid * 1000 + ep in val_ep:      # 基准只用训练集
                continue
            f = ROOT / f"feats/e{eid}/ep{ep:04d}.npz"
            if not f.exists():
                continue
            d = np.load(f); ft, st = d["feat"], d["state"]
            rows = np.array(rows, dtype=np.float32)
            for sub in np.unique(rows[:, 3]).astype(int):
                t = rows[rows[:, 3] == sub][:, 2].astype(int); t = t[t < len(ft)]
                if len(t) < 30: continue
                sg = np.array([sig_of(ft[x], st[x], eid, sub) for x in t[::3]])
                e = None; sm = []
                for v in sg:
                    e = v if e is None else A * v + (1 - A) * e
                    sm.append(e)
                sm = np.array(sm)
                # 段末 1/3 的平均涨幅（每原始帧），触发点就在这一带
                k = len(sm) // 3
                rate.setdefault((eid, sub), []).append((sm[-1] - sm[-k]) / max(k * 3, 1))
    return {k: float(np.median(v)) for k, v in rate.items()}


RATE = measure_rate()
print(f"  已标定 {len(RATE)} 组「信号每帧涨幅」基准\n")


def run(speed, adaptive_d=False):
    rel, absf, done, tot = [], [], 0, 0
    for eid in ck["experts"]:
        by = {}
        for r in lab[lab[:, 0] == eid]:
            by.setdefault(int(r[1]), []).append(r)
        for ep, rows in sorted(by.items()):
            if eid * 1000 + ep not in val_ep:
                continue
            f = ROOT / f"feats/e{eid}/ep{ep:04d}.npz"
            if not f.exists():
                continue
            d = np.load(f); ft, st = d["feat"], d["state"]
            rows = np.array(rows, dtype=np.float32)
            truth = {int(x): int(rows[rows[:, 3] == x][:, 2].max()) for x in np.unique(rows[:, 3])}
            seglen = {int(x): int((rows[:, 3] == x).sum()) for x in np.unique(rows[:, 3])}
            n = len(ft)
            # 重采样：idx[i] 是第 i 个推理步该看原视频的哪一帧
            idx = np.clip((np.arange(int(n / speed)) * speed).round().astype(int), 0, n - 1)
            idx = np.concatenate([idx, np.full(int(60 / speed) + 1, n - 1)])
            tot += 1
            sub, ema, hits, pend, steps, since, fin = 1, None, 0, None, 0, 10**9, False
            hist = []
            for i, t in enumerate(idx):
                if fin:
                    break
                raw = sig_of(ft[t], st[t], eid, sub)
                tau = cfg["tau"][f"{eid}_{sub}"]
                ema = raw if ema is None else A * raw + (1 - A) * ema
                hist.append(ema)
                steps += 1; since += 1
                # 速度比：拿本段前 24 步的涨幅比训练基准。K 和 D 共用它 ——
                # 它们都是「步数」量纲，而正确的不变量是「视觉距离」。
                ratio = 1.0
                if adaptive_d and len(hist) >= 24:
                    v_obs = (hist[23] - hist[0]) / 23
                    v_tr = RATE.get((eid, sub), v_obs)
                    if v_obs > 1e-6 and v_tr > 1e-6:
                        ratio = float(np.clip(v_tr / v_obs, 0.4, 2.5))
                Keff = max(3, int(round(K * ratio)))
                hits = hits + 1 if ema > tau else 0
                if pend is not None:
                    pend -= 1
                if pend is None and hits >= Keff and since >= LOCK:
                    pend = int(round(cfg["delay"][f"{eid}_{sub}"] * ratio))
                if pend is None and steps > int(cfg["max_steps"][f"{eid}_{sub}"] / speed):
                    pend = 0
                if pend is not None and pend <= 0:
                    b = truth.get(sub)
                    if b is not None:
                        e_steps = i - (b / speed)              # 重采样下标空间里的误差
                        absf.append(e_steps * speed)           # 折回原始帧
                        rel.append(e_steps / max(seglen[sub] / speed, 1) * 100)
                    if sub >= N_SUB[eid]:
                        fin = True
                    else:
                        sub += 1; ema = None; hits = 0; steps = 0; since = 0; hist = []
                    pend = None
            if fin:
                done += 1
    r, a = np.array(rel), np.array(absf)
    return done, tot, np.median(r), (np.abs(r) <= 8).mean() * 100, np.median(a)


for ad in (False, True):
    print(f"  ── K 与 D：{'按实测速度自适应缩放' if ad else '固定步数（现状）'} ──")
    print(f"  {'速度':<12}{'整集跑完':>10}{'视觉位置误差':>14}{'占段长≤8%':>11}{'折回原帧误差':>14}")
    for sp, name in [(0.5, "0.5× 慢一半"), (1.0, "1.0× 原速(15Hz)"),
                     (1.5, "1.5× (策略10Hz)"), (2.0, "2.0× (7.5Hz)"),
                     (3.0, "3.0× (5Hz)")]:
        d_, t_, med_r, w8, med_a = run(sp, ad)
        print(f"  {name:<12}{d_}/{t_:<8}{med_r:>+13.1f}%{w8:>10.0f}%{med_a:>+13.0f}帧")
    print()
print()
print("  「视觉位置误差」= 触发点偏离真边界多少（占该段长度的百分比）——")
print("     这个量与速度无关才说明模型看的是场景而不是时间")
print("  「折回原帧误差」= 换算回原始视频帧数，受延迟补偿 D 影响（D 按推理步计）")
