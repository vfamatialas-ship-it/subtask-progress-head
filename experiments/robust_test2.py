#!/usr/bin/env python3
"""扰动鲁棒性复测（v2）：解码一次跑全部扰动 + 区分「哪一路相机移位」。

对 v1 的两处修正
----------------
1. **解码一次**。v1 对每种扰动都重解一遍视频，40 分钟里大半耗在解码上。
   现在每集只解一次，7 种扰动共用同一份原始帧。

2. **区分受扰的视角**。v1 把三路一起扰动，但腕部相机是**刚性装在夹爪上的**，
   不会相对手臂独立移位 —— 三路同时平移模拟的是「三个相机同时被碰」，
   现实中罕见。真实场景是**第三视角机位被碰**（它是独立支架）。
   所以分开测：只扰第三视角 / 只扰腕部 / 三路全扰，
   才知道 v1 那 23 个点的跌幅里，有多少落在真实会发生的那一档。

用法: robust_test2.py [权重目录]   默认 runs/p_third_right_left_e012
"""
from __future__ import annotations

import sys
from pathlib import Path

import os

import numpy as np
import torch

sys.path.insert(0, "<SWITCH_ROOT>/deploy")
from switcher import SubtaskSwitcher                            # noqa: E402

ROOT = Path("<SWITCH_ROOT>")
B = Path("<DATA_ROOT>/local")
PKG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("<SWITCH_ROOT>/deploy")
DS = {0: "nero_right_box_pick_ee_v1", 1: "nero_left_box_pick_v2_sub",
      2: "nero_hezi_closing_ee_v1", 3: "nero_stage56_flap_closing_ee_v2"}
VIEWS = ["observation.images.third_view", "observation.images.right_wrist",
         "observation.images.left_wrist"]
KEYS = ["third_view", "right_wrist", "left_wrist"]

import av                                                       # noqa: E402
import pyarrow.parquet as pq                                    # noqa: E402


def warp(im, fx=0.0, fy=0.0, zf=1.0, br=1.0):
    """缩放+平移用一条坐标映射，边缘 clip 复制。zf<1 也能正确处理。"""
    h, w = im.shape[:2]
    ys = np.clip(((np.arange(h) - h / 2) / zf + h / 2 - fy * h).round().astype(int), 0, h - 1)
    xs = np.clip(((np.arange(w) - w / 2) / zf + w / 2 - fx * w).round().astype(int), 0, w - 1)
    o = im[ys][:, xs]
    return np.clip(o.astype(np.float32) * br, 0, 255).astype(np.uint8) if br != 1.0 else o


# (名称, 变换, 作用在哪几路)   0=第三视角 1=右腕 2=左腕
CASES = [
    ("无扰动",             lambda im: im,                              (0, 1, 2)),
    ("第三视角 平移3%",     lambda im: warp(im, 0.03, 0.02),            (0,)),
    ("第三视角 平移6%",     lambda im: warp(im, 0.06, 0.04),            (0,)),
    ("第三视角 缩放1.08",   lambda im: warp(im, zf=1.08),               (0,)),
    ("仅腕部 平移6%",       lambda im: warp(im, 0.06, 0.04),            (1, 2)),
    ("三路 平移6%",         lambda im: warp(im, 0.06, 0.04),            (0, 1, 2)),
    ("三路 缩放1.08",       lambda im: warp(im, zf=1.08),               (0, 1, 2)),
    ("三路 亮度×0.75",      lambda im: warp(im, br=0.75),               (0, 1, 2)),
]

# val 集必须取自**这个包对应的那次训练**，不能写死三专家那一版 ——
# 四专家训练的 episode 划分不同，用错了等于在训练集上评。
_CK = os.environ.get("ROBUST_CKPT", str(ROOT / "runs/p_third_right_left_e012/best.pt"))
ck = torch.load(_CK, map_location="cuda", weights_only=False)
val_ep = sorted(set(ck["val_ep"]))
lab = np.load(ROOT / "labels/labels.npy")
# ★ 随机抽，不要取「排序后的前 4 集」——实测那样抽出来的子集系统性偏难
#   （≤10帧 50% vs 其余 48 集的 84%），绝对值没法和全量口径比较。
#   固定种子保证多次运行之间可比。
# --all: 用全部留出集。12 集太小 —— 小样本已经两次把结论带偏
# （偏难子集报 50%、其余 48 集 84%），真实场景的数字必须用全量钉死。
ALL = "--all" in sys.argv
rng = np.random.default_rng(0)
pick = []
for eid in ck["experts"]:
    cand = [k for k in val_ep if k // 1000 == eid]
    pick += cand if ALL else list(rng.choice(cand, size=min(4, len(cand)), replace=False))
if ALL:
    # 全量时只跑真实会发生的情形，把机时花在有决策价值的那几档上
    CASES = [c for c in CASES if c[2] == (0,) or c[0] == "无扰动"]
print(f"· 权重 {PKG}")
print(f"· 留出集抽 {len(pick)} 集 × {len(CASES)} 种情形（每集只解码一次）\n")

sws = {int(eid): SubtaskSwitcher(expert=int(eid), pkg=PKG, verbose=False)
       for eid in ck["experts"]}
acc = {c[0]: ([], 0, 0) for c in CASES}

for key in pick:
    eid, ep = int(key // 1000), int(key % 1000)
    ds = DS[eid]
    tbl = pq.read_table(B / ds / f"data/chunk-000/episode_{ep:06d}.parquet").to_pydict()
    st = np.asarray([np.asarray(s, np.float32) for s in tbl["observation.state"]])
    n = len(st)
    vids = []
    for v in VIEWS:                       # ★ 只解一次
        buf = []
        with av.open(str(B / ds / f"videos/chunk-000/{v}/episode_{ep:06d}.mp4")) as c:
            for fr in c.decode(video=0):
                if len(buf) >= n:
                    break
                buf.append(fr.to_ndarray(format="rgb24"))
        vids.append(buf)
    m = lab[(lab[:, 0] == eid) & (lab[:, 1] == ep)]
    truth = {int(s): int(m[m[:, 3] == s][:, 2].max()) for s in np.unique(m[:, 3])}

    for name, fn, which in CASES:
        sw = sws[eid]; sw.reset()
        errs, e_, t_ = acc[name]
        t_ += 1
        for t in list(range(n)) + [n - 1] * 60:
            if sw.finished:
                break
            imgs = {}
            for vi, k in enumerate(KEYS):
                src = vids[vi][t] if t < len(vids[vi]) else vids[vi][-1]
                imgs[k] = fn(src) if vi in which else src
            prev = sw.sub
            r = sw.step(imgs, st[t])
            if r["switched"] and prev in truth:
                errs.append(t - truth[prev])
        acc[name] = (errs, e_ + (1 if sw.finished else 0), t_)

print(f"  {'情形':<18}{'跑完整集':>10}{'≤10帧':>8}{'≤20帧':>8}{'误差中位':>10}")
base = None
for name, _, _ in CASES:
    errs, d_, t_ = acc[name]
    e = np.array(errs) if errs else np.array([999])
    w10 = (np.abs(e) <= 10).mean() * 100
    if base is None:
        base = w10
    print(f"  {name:<18}{d_}/{t_:<8}{w10:>7.0f}%{(np.abs(e)<=20).mean()*100:>7.0f}%"
          f"{np.median(e):>+10.0f}")
print(f"\n  判读：只扰第三视角的那几行才是**真实会发生**的情形"
      f"（腕部相机刚性装在夹爪上，不会独立移位）")
