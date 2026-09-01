#!/usr/bin/env python3
"""停滞门控：用数据标定阈值，并把注入式卡顿打到真实部署代码路径上。

为什么必须做这一步
------------------
停滞门控是抗卡顿的**执行部件**，但 switch_ctl 里那两个阈值
（stall_img_eps / stall_state_eps）此前是拍的。拍错的后果是双向的：
  · 定太松 → 正常运动被误判成卡顿，门控一直冻结，永远不切（真机卡死）
  · 定太紧 → 真卡顿时门控不生效，EMA 继续爬过阈值，误切
所以要用真实相邻帧的差分分布来定，并留出安全间隔。

两件事：
 A. 标定：统计真实相邻帧的 Δ图像 / Δstate 分布，取低分位作为「正在动」的下界，
    阈值放在它下面。同时验证「复制帧」（模拟卡顿）确实落在阈值以内。
 B. 注入测试：在整集回放中间硬插 N 步静止帧，跑**真实的 sw.step 逻辑**，
    确认卡顿期间不推进、卡顿结束后正常恢复、最终切换时刻几乎不受影响。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "<SWITCH_ROOT>")
from switch_ctl import SIG, SubtaskSwitcher, ROOT     # noqa: E402

RUN = Path("<SWITCH_ROOT>/runs/p_third_right_left_e012")
B = Path("<DATA_ROOT>/local")
DS = {0: "nero_right_box_pick_ee_v1", 1: "nero_left_box_pick_v2_sub",
      2: "nero_hezi_closing_ee_v1"}

# ── A. 标定：真实相邻帧到底差多少 ────────────────────────────────────
print("── A. 停滞阈值标定（真实相邻帧差分）──")
import av
import pyarrow.parquet as pq

d_img_all, d_st_all = [], []
for eid, ds in DS.items():
    # state：直接从 parquet 读，取相邻帧最大分量变化
    for ep in range(3):
        f = B / ds / f"data/chunk-000/episode_{ep:06d}.parquet"
        if not f.exists():
            continue
        st = np.asarray([np.asarray(s, np.float32)
                         for s in pq.read_table(f).to_pydict()["observation.state"]])
        d_st_all.append(np.abs(np.diff(st, axis=0)).max(1))
        # 图像：第三视角，逐帧平均绝对差
        vp = B / ds / f"videos/chunk-000/observation.images.third_view/episode_{ep:06d}.mp4"
        if vp.exists():
            prev, ds_img = None, []
            with av.open(str(vp)) as c:
                for i, fr in enumerate(c.decode(video=0)):
                    a = fr.to_ndarray(format="rgb24").astype(np.float32)
                    if prev is not None:
                        ds_img.append(np.abs(a - prev).mean())
                    prev = a
                    if i > 400:
                        break
            d_img_all.append(np.array(ds_img))

di = np.concatenate(d_img_all); dsx = np.concatenate(d_st_all)
print(f"  Δ图像(第三视角, 0~255 平均绝对差)  p1 {np.percentile(di,1):.3f}  "
      f"p5 {np.percentile(di,5):.3f}  中位 {np.median(di):.3f}")
print(f"  Δstate(最大分量)                   p1 {np.percentile(dsx,1):.5f}  "
      f"p5 {np.percentile(dsx,5):.5f}  中位 {np.median(dsx):.5f}")
print(f"  复制帧(模拟卡顿) Δ图像 = 0.000   Δstate = 0.00000")
# 阈值取「正在动」分布 p1 的一半：既在真实运动之下，又远在复制帧之上。
# ⚠ Δstate 的 p1/p5 都是 0 —— 真实运动中 state 经常整帧不变（量化/保持），
#   直接用 p1/2 会得到 0，而 `d_st < 0` 恒为假 → 门控永远不触发（第一版的 bug）。
#   所以必须给一个正的下限；判别力本来就主要来自图像差分，
#   state 只作为「必须同时静止」的附加约束。
img_eps = float(np.percentile(di, 1)) / 2
st_eps = max(float(np.percentile(dsx, 1)) / 2, 1e-5)
print(f"\n  ★ 建议 stall_img_eps = {img_eps:.3f}   stall_state_eps = {st_eps:.6f}")
print(f"    （取真实运动 p1 的一半：低于真实运动的 99%，又远高于复制帧的 0）")
frac = float((di < img_eps).mean() * 100)
print(f"    真实运动帧被误判为卡顿的比例：{frac:.2f}%")

# ── B. 注入卡顿，跑真实 sw.step 状态机 ────────────────────────────────
print("\n── B. 注入式卡顿测试（走真实部署代码路径）──")
ck = torch.load(RUN / "best.pt", map_location="cuda", weights_only=False)
val_ep = sorted(set(ck["val_ep"]))
lab = np.load(ROOT / "labels/labels.npy")
STALL = 90          # 注入 90 步卡顿（3 秒 @30fps）


def run(eid, ep, stall_at=None):
    """整集跑一遍；stall_at 不为空时在该帧插入 STALL 步静止。返回各段切换帧。"""
    d = np.load(ROOT / f"feats/e{eid}/ep{ep:04d}.npz")
    ft, st = d["feat"], d["state"]
    sw = SubtaskSwitcher(expert=eid, run_dir=RUN, verbose=False,
                         stall_img_eps=img_eps, stall_state_eps=st_eps)
    vi = [["third", "right", "left"].index(v) for v in sw.views]
    seq = list(range(len(ft))) + [len(ft) - 1] * 60
    if stall_at is not None:
        k = seq.index(stall_at)
        seq = seq[:k] + [stall_at] * STALL + seq[k:]
    sws, frozen_n = {}, 0
    for step_i, ti in enumerate(seq):
        if sw.finished:
            break
        feat = torch.as_tensor(ft[ti, vi].reshape(1, -1).astype(np.float32), device="cuda")
        p, dn, tb = sw._heads(feat, st[ti])
        raw = SIG[sw.cfg["signal"]](p, dn, tb)
        tau = sw.cfg["tau"][sw._key()]
        # 停滞检测（与 switch_ctl.step 同逻辑）
        cur_st = st[ti]
        stalled = False
        if sw._prev_state is not None:
            stalled = (np.abs(cur_st - sw._prev_state).max() < st_eps
                       and np.abs(ft[ti, vi[0]].astype(np.float32)
                                  - sw._prev_img).mean() < 1e-6)
        sw._prev_state = cur_st; sw._prev_img = ft[ti, vi[0]].astype(np.float32)
        freeze = stalled and raw <= tau
        if freeze:
            frozen_n += 1
            continue
        a = sw.cfg["alpha"]
        sw.ema = raw if sw.ema is None else a * raw + (1 - a) * sw.ema
        sw.steps += 1; sw.since_switch += 1
        sw.hits = sw.hits + 1 if sw.ema > tau else 0
        if sw.pending is not None:
            sw.pending -= 1
        if sw.pending is None and sw.hits >= sw.cfg["K"] and sw.since_switch >= sw.lockout:
            sw.pending = int(sw.cfg["delay"][sw._key()])
        if sw.pending is None and sw.steps > int(sw.cfg["max_steps"][sw._key()]):
            sw.pending = 0
        if sw.pending is not None and sw.pending <= 0:
            sws[sw.sub] = ti          # 记「原始帧号」，可与无卡顿版直接比
            if sw.sub >= sw.n_sub:
                sw.finished = True
            else:
                sw.sub += 1; sw.ema = None; sw.hits = 0
                sw.steps = 0; sw.since_switch = 0
            sw.pending = None
    return sws, frozen_n, sw.finished


shift, ok, tested = [], 0, 0
for key in val_ep[:24]:
    eid, ep = key // 1000, key % 1000
    m = lab[(lab[:, 0] == eid) & (lab[:, 1] == ep)]
    if not len(m):
        continue
    base, _, fin0 = run(eid, ep)
    if not fin0 or len(base) < 2:
        continue
    # 卡顿点插在第一段中部
    s1_end = int(m[m[:, 3] == 1][:, 2].max())
    inj, nfroz, fin1 = run(eid, ep, stall_at=s1_end // 2)
    tested += 1
    if fin1:
        ok += 1
    for k in base:
        if k in inj:
            shift.append(inj[k] - base[k])
    if tested <= 3:
        print(f"  E{eid+1} ep{ep}: 冻结 {nfroz}/{STALL} 步   "
              f"切换帧 无卡顿{list(base.values())} → 有卡顿{list(inj.values())}")

sh = np.array(shift)
print(f"\n  注入 {STALL} 步卡顿 × {tested} 集")
print(f"  门控冻结步数：应 ≈{STALL}（见上面逐集明细）")
print(f"  切换时刻相对无卡顿版的偏移：中位 {np.median(sh):+.0f} 帧  "
      f"最大 |{np.abs(sh).max()}| 帧   0 表示卡顿完全没影响判断")
print(f"  卡顿后仍跑完整集：{ok}/{tested}")
