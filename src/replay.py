#!/usr/bin/env python3
"""串行回放：在留出集上整集跑状态机，量最贴近真机的行为。

和分段评估的区别 —— 这个更严格
------------------------------
分段评估把每一段单独喂进去，起点永远是真实边界。串行回放里，
第 n 段的起点是**上一段实际切走的位置**，误差会累积：早切 20 帧就意味着
下一段带着 20 帧的错误上下文开局。真机就是这样跑的。

同时统计：切满全部子任务的比例（没切满 = 真机会卡住）、每段误差、兜底触发次数。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "<SWITCH_ROOT>")
from switch_ctl import SIG, SubtaskSwitcher, ROOT     # noqa: E402

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "<SWITCH_ROOT>/runs/p_third_right_left_e012")

ck = torch.load(RUN / "best.pt", map_location="cuda", weights_only=False)
val_ep = set(ck["val_ep"])
lab = np.load(ROOT / "labels/labels.npy")

print(f"· {RUN.name}   视角 {ck['views']}   留出 {len(val_ep)} 集\n")
tot_err, done_all, n_ep, n_forced, stuck = [], 0, 0, 0, []
per_expert = {}

for eid in ck["experts"]:
    sw = SubtaskSwitcher(expert=eid, run_dir=RUN, verbose=False)
    vi = [["third", "right", "left"].index(v) for v in sw.views]
    errs_e, ok_e, n_e = [], 0, 0
    for ep in range(100):
        if eid * 1000 + ep not in val_ep:
            continue
        f = ROOT / f"feats/e{eid}/ep{ep:04d}.npz"
        if not f.exists():
            continue
        d = np.load(f); ft, st = d["feat"], d["state"]
        m = lab[(lab[:, 0] == eid) & (lab[:, 1] == ep)]
        truth = {int(s): int(m[m[:, 3] == s][:, 2].max()) for s in np.unique(m[:, 3])}

        sw.reset()
        n_ep += 1; n_e += 1
        # 末帧保持 PAD 步：最后一个子任务的真实边界就是整集最后一帧，
        # 触发后还要等 D 步延迟补偿，帧就用完了 —— 真机不会在那一刻停止推流，
        # 机械臂会保持在末位姿继续出图。不补这一段会把「已经切对了」误记成卡住。
        PAD = 60
        for t in range(len(ft) + PAD):
            ti = min(t, len(ft) - 1)
            if sw.finished:
                break
            feat = torch.as_tensor(ft[ti, vi].reshape(1, -1).astype(np.float32), device="cuda")
            p, dn, tb = sw._heads(feat, st[ti])
            raw = SIG[sw.cfg["signal"]](p, dn, tb)
            a = sw.cfg["alpha"]
            sw.ema = raw if sw.ema is None else a * raw + (1 - a) * sw.ema
            sw.steps += 1; sw.since_switch += 1
            sw.hits = sw.hits + 1 if sw.ema > sw.cfg["tau"][sw._key()] else 0
            if sw.pending is not None:
                sw.pending -= 1
            if (sw.pending is None and sw.hits >= sw.cfg["K"]
                    and sw.since_switch >= sw.lockout):
                sw.pending = int(sw.cfg["delay"][sw._key()])
            forced = False
            if sw.pending is None and sw.steps > int(sw.cfg["max_steps"][sw._key()]):
                sw.pending, forced = 0, True
            if sw.pending is not None and sw.pending <= 0:
                if forced:
                    n_forced += 1
                b = truth.get(sw.sub)
                if b is not None:
                    errs_e.append(t - b); tot_err.append(t - b)
                if sw.sub >= sw.n_sub:
                    sw.finished = True
                else:
                    sw.sub += 1; sw.ema = None; sw.hits = 0
                    sw.steps = 0; sw.since_switch = 0
                sw.pending = None
        if sw.finished:
            done_all += 1; ok_e += 1
        else:
            stuck.append((eid, ep, sw.sub))
    per_expert[eid] = (np.array(errs_e), ok_e, n_e)

e = np.array(tot_err)
name = {0: "E1 右臂抓放", 1: "E2 左臂抓放", 2: "E3 封箱中段", 3: "E4 封箱末段"}
print(f"  {'专家':<12}{'跑完整集':>10}{'切换数':>8}{'误差中位':>10}{'≤10帧':>8}{'≤20帧':>8}")
for eid, (ee, ok, ne) in per_expert.items():
    if not len(ee):
        continue
    print(f"  {name[eid]:<12}{ok}/{ne:<8}{len(ee):>8}{np.median(ee):>+10.0f}"
          f"{(np.abs(ee)<=10).mean()*100:>7.0f}%{(np.abs(ee)<=20).mean()*100:>7.0f}%")
print(f"\n  合计 {n_ep} 集   跑完整集 {done_all}/{n_ep} = {done_all/n_ep*100:.0f}%"
      f"   兜底触发 {n_forced} 次")
print(f"  切换时刻误差：中位 {np.median(e):+.0f}  均值 {e.mean():+.1f}  "
      f"|误差|≤10 占 {(np.abs(e)<=10).mean()*100:.0f}%   ≤20 占 {(np.abs(e)<=20).mean()*100:.0f}%")
if stuck:
    print(f"  ⚠ 没跑完的集（会卡在该子任务）: {stuck[:8]}")
