#!/usr/bin/env python3
"""诊断：模型在「真实段末」到底输出多少？决定该改阈值、改标签还是改结构。

想知道三件事：
 1. 真段末的 p 分布 —— 若中位只有 0.6，那 τ=0.85 必然大面积漏切，是**校准问题**
 2. p 在段内的可分性 —— 段末的 p 和段中的 p 重叠多少（重叠大=信号本身弱，改阈值没用）
 3. done 概率在段末 vs 段中 —— 同上
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, "<SWITCH_ROOT>/tools")
FEAT = Path("<SWITCH_ROOT>/feats")
LAB = Path("<SWITCH_ROOT>/labels/labels.npy")

import importlib.util
spec = importlib.util.spec_from_file_location("tp", "<SWITCH_ROOT>/tools/train_progress.py")
tp = importlib.util.module_from_spec(spec); spec.loader.exec_module(tp)

ck = torch.load(sys.argv[1] if len(sys.argv) > 1
                else "<SWITCH_ROOT>/runs/p_third_e2/best.pt",
                map_location="cuda", weights_only=False)
net = tp.ProgressHead(len(ck["views"])).cuda(); net.load_state_dict(ck["model"]); net.eval()
vi = [{"third": 0, "right": 1, "left": 2}[v] for v in ck["views"]]
val_ep = set(ck["val_ep"])
lab = np.load(LAB)


@torch.no_grad()
def pred(ft, st, eid, sub):
    x = torch.as_tensor(ft[:, vi].reshape(len(ft), -1).astype(np.float32), device="cuda")
    s = np.zeros((len(ft), 20), np.float32); s[:, :st.shape[1]] = st
    p, d = net(x, torch.as_tensor(s, device="cuda"),
               torch.full((len(ft),), eid, dtype=torch.long, device="cuda"),
               torch.full((len(ft),), sub, dtype=torch.long, device="cuda"))
    return p.cpu().numpy(), torch.sigmoid(d).cpu().numpy()


p_end, p_mid, d_end, d_mid, lens, peak = [], [], [], [], [], []
per_sub = {}
for eid in ck["experts"]:
    m = lab[lab[:, 0] == eid]
    by_ep = {}
    for r in m:
        by_ep.setdefault(int(r[1]), []).append(r)
    for ep, rows in sorted(by_ep.items()):
        if eid * 1000 + ep not in val_ep:
            continue
        f = FEAT / f"e{eid}" / f"ep{ep:04d}.npz"
        if not f.exists():
            continue
        dd = np.load(f); ft, st = dd["feat"], dd["state"]
        rows = np.array(rows, dtype=np.float32)
        for sub in np.unique(rows[:, 3]).astype(int):
            t = rows[rows[:, 3] == sub][:, 2].astype(int); t = t[t < len(ft)]
            if len(t) < 20: continue
            p, d = pred(ft[t], st[t], eid, sub)
            L = len(t); lens.append(L)
            p_end.append(p[-5:].mean());  p_mid.append(p[L//3:2*L//3].mean())
            d_end.append(d[-5:].mean());  d_mid.append(d[L//3:2*L//3].mean())
            peak.append(p.max())
            per_sub.setdefault(int(sub), []).append((p[-5:].mean(), p.max(), d[-5:].mean()))

f = lambda a: f"中位{np.median(a):.3f} p10 {np.percentile(a,10):.3f} p90 {np.percentile(a,90):.3f}"
print(f"· {len(lens)} 段，中位段长 {np.median(lens):.0f} 帧\n")
print(f"  真段末 p     {f(p_end)}")
print(f"  段中部 p     {f(p_mid)}")
print(f"  段内 p 峰值  {f(peak)}   ← τ 必须低于这个的 p10 才不漏切")
print(f"  真段末 done  {f(d_end)}")
print(f"  段中部 done  {f(d_mid)}")
ov = np.mean(np.array(p_mid)[:, None] > np.array(p_end)[None, :])
print(f"\n  段中 p > 段末 p 的比例 {ov*100:.1f}%  ← 越高说明信号本身越不可分")
print(f"  段末 p 与段中 p 的间隔 {np.median(p_end)-np.median(p_mid):+.3f}")
print(f"\n  按子任务看段末 p：")
for k in sorted(per_sub):
    v = np.array(per_sub[k])
    print(f"    sub{k}: n={len(v):3d}  段末p 中位{np.median(v[:,0]):.3f}  "
          f"峰值p10 {np.percentile(v[:,1],10):.3f}  段末done 中位{np.median(v[:,2]):.3f}")
