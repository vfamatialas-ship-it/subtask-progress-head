#!/usr/bin/env python3
"""从四个专家的数据集导出「子任务进度」训练标签。

标签怎么来的：数据集里 prompt_index 是逐帧记录的，同一个子任务是一段连续游程，
所以子任务边界不用人工标 —— 游程的起止就是 subtask_start / subtask_end。
    p_t = (t - start) / (end - start)      ∈ [0,1]
    done_t = 1  if  t ∈ [end-4, end]       （5 帧软窗口；单帧正样本太少，BCE 学不动）

同时导出 event_code 里的 STAGE_COMPLETED / SUBTASK_DONE 帧作为**事件锚点**，
用来做「线性 progress vs 事件分段 progress」的对照实验（用户方案第六节要比的那个）。

⚠ 不导出任何与时间轴有关的特征（帧号、已执行时长）。
   部署时网络卡顿会让机械臂停在原地，一旦模型依赖时间就会继续漂移、误触发切换。
   进度必须只由「当前观测」决定，卡住时输入不变、输出自然不动。
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

B = Path("<DATA_ROOT>/local")
EXPERTS = {                       # expert_id : (数据集, 说明)
    0: ("nero_right_box_pick_ee_v1", "E1 右臂抓放"),
    1: ("nero_left_box_pick_v2_sub", "E2 左臂抓放"),
    2: ("nero_hezi_closing_ee_v1",   "E3 封箱中段(7子任务)"),
    3: ("nero_stage56_flap_closing_ee_v2", "E4 封箱末段(7子任务)"),
}
DONE_WIN = 5                       # done_t 的软窗口宽度（帧）


def runs(seq):
    """把逐帧的 prompt_index 压成 [(值, 起, 止)] 的游程。止是闭区间。"""
    out, i = [], 0
    for k, g in itertools.groupby(seq):
        n = len(list(g))
        out.append((k, i, i + n - 1))
        i += n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="<SWITCH_ROOT>/labels")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    rows, stats = [], {}
    for eid, (ds, name) in EXPERTS.items():
        files = sorted(glob.glob(str(B / ds / "data/chunk-000/*.parquet")))
        if not files:
            print(f"  ✘ {name}: 找不到 parquet"); continue
        n_ep = n_fr = n_sub = n_evt = 0
        durs, order_bad = [], 0
        for ep_i, f in enumerate(files):
            d = pq.read_table(f).to_pydict()
            pidx = d.get("prompt_index")
            if pidx is None:
                print(f"  ✘ {name}: 没有 prompt_index 列"); break
            ev = d.get("event_name", [None] * len(pidx))
            rr = runs(pidx)
            # 子任务必须单调递增；不递增说明这一集有回退/重做，先记下来别混进训练集
            ks = [k for k, _, _ in rr]
            if ks != sorted(ks):
                order_bad += 1
                continue
            for si, (k, s, e) in enumerate(rr):
                L = e - s
                if L < 5:                     # 太短的游程多半是标注抖动，丢掉
                    continue
                durs.append(L + 1)
                for t in range(s, e + 1):
                    rows.append((
                        eid, ep_i, t,
                        int(k),                              # subtask_index（1 起）
                        si,                                  # 该子任务是本集第几段
                        len(rr),                             # 本集共几段
                        (t - s) / L,                         # p_t 线性
                        1 if t >= e - (DONE_WIN - 1) else 0,  # done_t 软窗口
                        1 if (ev[t] in ("STAGE_COMPLETED", "SUBTASK_DONE")) else 0,  # 事件锚点
                    ))
                n_sub += 1
            n_evt += sum(1 for x in ev if x in ("STAGE_COMPLETED", "SUBTASK_DONE"))
            n_ep += 1; n_fr += len(pidx)
        stats[name] = dict(dataset=ds, episodes=n_ep, frames=n_fr, subtasks=n_sub,
                           events=n_evt, order_violation=order_bad,
                           dur_median=int(np.median(durs)) if durs else 0)
        print(f"  ✔ {name}: {n_ep}集 {n_fr}帧 {n_sub}段  事件锚点{n_evt}  "
              f"中位段长{stats[name]['dur_median']}帧  顺序异常{order_bad}集")

    arr = np.array(rows, dtype=np.float32)
    np.save(out / "labels.npy", arr)
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n· 共 {len(arr)} 帧 → {out}/labels.npy")
    print(f"  列: expert_id, episode, frame, subtask_index, seg_i, n_seg, p_t, done_t, event_anchor")
    if len(arr):
        print(f"  done_t 正样本占比 {arr[:,7].mean()*100:.1f}%   "
              f"事件锚点占比 {arr[:,8].mean()*100:.2f}%")


if __name__ == "__main__":
    main()
