#!/usr/bin/env python3
"""扰动鲁棒性：相机移位 / 亮度变化 下切换还准不准。

为什么这是上机前最大的未测风险
------------------------------
模型只见过一个录制时段的画面。真机上相机被碰了一下、或者换个时间灯光不同，
特征分布就偏了。前面所有指标都是在**同分布**的留出集上量的，测不到这个。

若结果脆弱，修法是明确的（带增强重训），但那要重新过一遍 SigLIP 编码，
约 1 小时 —— 所以必须先知道到底脆不脆，而不是等上机才发现。

扰动直接加在**原始视频帧**上，再走完整的 SigLIP 编码 → 进度头 → 状态机，
和真机链路一致（不能在缓存特征上做，那样测不出编码器的敏感度）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "<SWITCH_ROOT>/deploy")
from switcher import SIG, SubtaskSwitcher                      # noqa: E402

ROOT = Path("<SWITCH_ROOT>")
B = Path("<DATA_ROOT>/local")
DS = {0: "nero_right_box_pick_ee_v1", 1: "nero_left_box_pick_v2_sub",
      2: "nero_hezi_closing_ee_v1"}
VIEWS = ["observation.images.third_view", "observation.images.right_wrist",
         "observation.images.left_wrist"]

import av                                                       # noqa: E402
import pyarrow.parquet as pq                                    # noqa: E402


def shift(img, fx, fy):
    """平移 + 边缘复制填充，模拟相机被碰移位。"""
    h, w = img.shape[:2]
    dx, dy = int(w * fx), int(h * fy)
    out = np.empty_like(img)
    xs = np.clip(np.arange(w) - dx, 0, w - 1)
    ys = np.clip(np.arange(h) - dy, 0, h - 1)
    out[:] = img[ys][:, xs]
    return out


def zoom(img, f):
    """中心缩放，模拟相机前后位移。"""
    h, w = img.shape[:2]
    ch, cw = int(h / f), int(w / f)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    c = img[y0:y0 + ch, x0:x0 + cw]
    yi = np.clip((np.arange(h) * ch / h).astype(int), 0, ch - 1)
    xi = np.clip((np.arange(w) * cw / w).astype(int), 0, cw - 1)
    return c[yi][:, xi]


PERTURB = {
    "无扰动":          lambda im: im,
    "平移 3%":         lambda im: shift(im, 0.03, 0.02),
    "平移 6%":         lambda im: shift(im, 0.06, 0.04),
    "亮度 ×0.75":      lambda im: np.clip(im.astype(np.float32) * 0.75, 0, 255).astype(np.uint8),
    "亮度 ×1.30":      lambda im: np.clip(im.astype(np.float32) * 1.30, 0, 255).astype(np.uint8),
    "缩放 1.08":       lambda im: zoom(im, 1.08),
    "平移3%+亮度0.8":  lambda im: shift(
        np.clip(im.astype(np.float32) * 0.8, 0, 255).astype(np.uint8), 0.03, 0.02),
}

ck = torch.load(ROOT / "runs/p_third_right_left_e012/best.pt",
                map_location="cuda", weights_only=False)
val_ep = sorted(set(ck["val_ep"]))
lab = np.load(ROOT / "labels/labels.npy")

# 每个专家取 4 集，控制运行时间（要跑 7 种扰动 × 3 视角的完整编码）
pick = []
for eid in (0, 1, 2):
    pick += [k for k in val_ep if k // 1000 == eid][:4]
print(f"· 留出集抽 {len(pick)} 集 × {len(PERTURB)} 种扰动，走完整 SigLIP 编码链路\n")

sws = {eid: SubtaskSwitcher(expert=eid, verbose=False) for eid in (0, 1, 2)}


def frames_of(eid, ep):
    """解出三路视频。腕部是 AV1 —— 必须 PyAV，cv2 会静默失败产出全零。"""
    ds = DS[eid]
    n = len(pq.read_table(B / ds / f"data/chunk-000/episode_{ep:06d}.parquet")
            .to_pydict()["observation.state"])
    out = []
    for v in VIEWS:
        vp = B / ds / f"videos/chunk-000/{v}/episode_{ep:06d}.mp4"
        buf = []
        with av.open(str(vp)) as c:
            for fr in c.decode(video=0):
                if len(buf) >= n:
                    break
                buf.append(fr.to_ndarray(format="rgb24"))
        out.append(buf)
    return out, n


results = {}
for name, fn in PERTURB.items():
    errs, done, tot = [], 0, 0
    for key in pick:
        eid, ep = key // 1000, key % 1000
        vids, n = frames_of(eid, ep)
        st = np.asarray([np.asarray(s, np.float32) for s in
                         pq.read_table(B / DS[eid] / f"data/chunk-000/episode_{ep:06d}.parquet")
                         .to_pydict()["observation.state"]])
        m = lab[(lab[:, 0] == eid) & (lab[:, 1] == ep)]
        truth = {int(s): int(m[m[:, 3] == s][:, 2].max()) for s in np.unique(m[:, 3])}
        sw = sws[eid]; sw.reset(); tot += 1
        seq = list(range(n)) + [n - 1] * 60
        for t in seq:
            if sw.finished:
                break
            imgs = {"third_view": fn(vids[0][t]),
                    "right_wrist": fn(vids[1][t]) if t < len(vids[1]) else vids[1][-1],
                    "left_wrist": fn(vids[2][t]) if t < len(vids[2]) else vids[2][-1]}
            prev = sw.sub
            r = sw.step(imgs, st[t])
            if r["switched"] and prev in truth:
                errs.append(t - truth[prev])
        if sw.finished:
            done += 1
    e = np.array(errs) if errs else np.array([999])
    results[name] = (done, tot, e)
    print(f"  {name:<16}跑完 {done}/{tot}   切换 {len(errs):3d} 次   "
          f"误差中位 {np.median(e):+5.0f}   ≤10帧 {(np.abs(e)<=10).mean()*100:3.0f}%   "
          f"≤20帧 {(np.abs(e)<=20).mean()*100:3.0f}%")

base = results["无扰动"]
print(f"\n  基准（无扰动）：跑完 {base[0]}/{base[1]}，≤10帧 {(np.abs(base[2])<=10).mean()*100:.0f}%")
worst = min((v for k, v in results.items() if k != "无扰动"),
            key=lambda v: (np.abs(v[2]) <= 10).mean())
print(f"  最差扰动：≤10帧 {(np.abs(worst[2])<=10).mean()*100:.0f}%   跑完 {worst[0]}/{worst[1]}")
print(f"\n  判读：跑完率若保持 → 不会卡死，最坏只是切早/切晚；"
      f"跑完率若掉 → 必须带增强重训")
