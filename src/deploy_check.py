#!/usr/bin/env python3
"""部署前的两项硬检查：推理延迟、重试误报。

A. 延迟
   SigLIP2-so400m 是 4 亿参数的视觉塔，每步要跑 1~3 张图。若单步超过控制周期，
   这套方案在真机上就是不可用的 —— 这个数从没测过，属于硬阻塞项。

B. 重试误报（此前标为「无法用现有数据验证」的风险，其实可以构造）
   数据集全是成功演示，但真机会抓空后退回重来。此时腕部近景在重试的接近过程中
   与首次接近很像，模型可能误以为「又快做完了」而提前切走。
   构造方法：在段内 80% 处，把该段 30% 附近的帧插回去若干帧（模拟退回重做），
   然后接回原序列。看两件事：
     · 倒带期间进度是否**回落**（回落=在读世界状态，好；不回落=在读相位，坏）
     · 倒带是否导致**误触发**（在真边界之前就切走）
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "<SWITCH_ROOT>")
from switch_ctl import SIG, SubtaskSwitcher, ROOT     # noqa: E402

RUNS = {"三视角": "p_third_right_left_e012", "仅第三视角": "p_third_e012"}

# ── A. 延迟 ───────────────────────────────────────────────────────────
print("── A. 单步推理延迟（GPU2）──")
for tag, r in RUNS.items():
    sw = SubtaskSwitcher(expert=2, run_dir=ROOT / "runs" / r, verbose=False)
    nv = len(sw.views)
    imgs = {k: np.random.randint(0, 255, (480, 640, 3), np.uint8)
            for k in ("third_view", "right_wrist", "left_wrist")}
    st = np.zeros(20, np.float32)
    for _ in range(5):                       # 预热
        sw.step(imgs, st); sw.reset()
    torch.cuda.synchronize()
    N = 60
    t0 = time.perf_counter()
    for _ in range(N):
        sw.step(imgs, st)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / N * 1000
    print(f"  {tag:<12}{nv} 张图   {dt:6.1f} ms/步   ≈ {1000/dt:5.1f} Hz")
print("  （策略本身通常跑 5~10Hz；只要切换头远快于此就不构成瓶颈）")

# ── B. 重试误报 ───────────────────────────────────────────────────────
print("\n── B. 重试/倒带误报测试 ──")
RUN = ROOT / "runs" / RUNS["三视角"]
ck = torch.load(RUN / "best.pt", map_location="cuda", weights_only=False)
val = set(ck["val_ep"])
lab = np.load(ROOT / "labels/labels.npy")
REW = 40          # 倒带插入多少帧


def trace(eid, ep, sub, rewind=False):
    """跑单段，返回 (进度序列, 首次触发下标, 段长)。rewind=True 时插入倒带。"""
    d = np.load(ROOT / f"feats/e{eid}/ep{ep:04d}.npz")
    ft, st = d["feat"], d["state"]
    m = lab[(lab[:, 0] == eid) & (lab[:, 1] == ep)]
    t = m[m[:, 3] == sub][:, 2].astype(int); t = t[t < len(ft)]
    if len(t) < 60:
        return None
    L = len(t)
    seq = list(t)
    if rewind:                      # 在 80% 处插入 30% 附近的帧
        cut, back = int(L * 0.8), int(L * 0.3)
        seq = seq[:cut] + list(t[back:back + REW]) + seq[cut:]
    sw = SubtaskSwitcher(expert=eid, run_dir=RUN, verbose=False)
    sw.reset(sub=sub)
    vi = [["third", "right", "left"].index(v) for v in sw.views]
    ps, trig = [], None
    for i, ti in enumerate(seq):
        feat = torch.as_tensor(ft[ti, vi].reshape(1, -1).astype(np.float32), device="cuda")
        p, dn, tb = sw._heads(feat, st[ti])
        raw = SIG[sw.cfg["signal"]](p, dn, tb)
        a = sw.cfg["alpha"]
        sw.ema = raw if sw.ema is None else a * raw + (1 - a) * sw.ema
        ps.append(sw.ema)
        sw.hits = sw.hits + 1 if sw.ema > sw.cfg["tau"][sw._key()] else 0
        if trig is None and sw.hits >= sw.cfg["K"]:
            trig = i
    return np.array(ps), trig, L


drops, early_fire, n, sh = [], 0, 0, []
for key in sorted(val):
    eid, ep = key // 1000, key % 1000
    m = lab[(lab[:, 0] == eid) & (lab[:, 1] == ep)]
    if not len(m):
        continue
    for sub in np.unique(m[:, 3]).astype(int):
        base = trace(eid, ep, sub, rewind=False)
        rw = trace(eid, ep, sub, rewind=True)
        if base is None or rw is None:
            continue
        pb, tb_, L = base
        pr, tr_, _ = rw
        cut = int(L * 0.8)
        # 倒带期间进度相对倒带前的变化：负=回落（好）
        pre = pr[cut - 5:cut].mean()
        during = pr[cut:cut + REW].mean()
        drops.append(during - pre)
        # 误触发：倒带版在倒带段内就触发，而基准版此时还没触发
        if tr_ is not None and cut <= tr_ < cut + REW and (tb_ is None or tb_ >= cut):
            early_fire += 1
        if tb_ is not None and tr_ is not None:
            sh.append((tr_ - REW) - tb_)     # 扣掉插入的帧数后的净偏移
        n += 1
    if n >= 120:
        break

dr = np.array(drops)
print(f"  {n} 段，每段在 80% 处插入 {REW} 帧「退回到 30% 位置」的画面")
print(f"  倒带期间进度变化：中位 {np.median(dr):+.4f}   "
      f"回落的段占 {(dr<0).mean()*100:.0f}%")
print(f"    负值 = 进度跟着回落 → 模型在读**世界状态**，重试不会骗到它")
print(f"  倒带段内误触发：{early_fire}/{n} = {early_fire/n*100:.1f}%")
if sh:
    s = np.array(sh)
    print(f"  扣除插入帧后的净触发偏移：中位 {np.median(s):+.0f} 帧  "
          f"|偏移|≤5 占 {(np.abs(s)<=5).mean()*100:.0f}%")
    print(f"    ≈0 表示倒带过后判断完全恢复，没有留下后遗症")
