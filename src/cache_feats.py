#!/usr/bin/env python3
"""把三路 RGB 用 SigLIP2 编码一遍并缓存，之后训进度头就是纯 MLP，几十秒一轮。

为什么先缓存：视觉塔冻结，同一帧的特征永远不变，每轮重算是纯浪费。
188k 帧 × 3 视角一次性算完约 1 小时，之后可以反复试 loss/结构/超参而不再碰 GPU 重活。

同时把 state 也一并存下来（各专家维度不同：10/16/20/20，分开存，不做 padding ——
不同专家同一维下标的语义不同，硬塞进同一张量会互相污染）。
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

B = Path("<DATA_ROOT>/local")
SIGLIP = "<WORK_ROOT>/RLinf_nero/hf_models/siglip2-so400m-patch14-224"
EXPERTS = {
    0: ("nero_right_box_pick_ee_v1", "E1"),
    1: ("nero_left_box_pick_v2_sub", "E2"),
    2: ("nero_hezi_closing_ee_v1",   "E3"),
    # E4 用 v2：旧的 nero_hezi_stage56_ee_v1 每集只有 2 段（prompt_index 从 5 起），
    # v2 是完整的 7 个子任务、69 集、顺序异常 0。
    3: ("nero_stage56_flap_closing_ee_v2", "E4"),
}
VIEWS = ["observation.images.third_view",
         "observation.images.right_wrist",
         "observation.images.left_wrist"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="<SWITCH_ROOT>/feats")
    ap.add_argument("--experts", default="0,1,2")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--grid", type=int, default=0,
                    help="非 0 时额外缓存 GxG 网格池化的 patch token（而不是只存全局向量）。"
                         "pooler_output 把整张图压成一个向量，"
                         "「盒子在不在夹爪里」这类**空间局部**判据会被抹掉。"
                         "G=2 时每视角 4x1152，存储约是全局向量的 4 倍。")
    ap.add_argument("--aug", type=int, default=0,
                    help="几何增强的随机种子偏移；>0 时对每集用一组随机的平移/缩放/亮度")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # ★ 用 PyAV 而不是 cv2：两路腕部相机是 AV1，服务器的 cv2 解不了 ——
    #   它静默返回失败、特征全成零，模型等于只看第三视角（第一版就踩了这个坑）。
    #   PyAV 18.1 自带 libdav1d，H.264 / AV1 都能解。
    import av
    import pyarrow.parquet as pq
    from transformers import SiglipVisionModel, SiglipImageProcessor

    dev = "cuda"
    proc = SiglipImageProcessor.from_pretrained(SIGLIP)
    vis = SiglipVisionModel.from_pretrained(SIGLIP, torch_dtype=torch.float16).to(dev).eval()
    mean = torch.tensor(proc.image_mean, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(proc.image_std, device=dev).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode(frames):                       # frames: list[HxWx3 uint8 RGB]
        x = torch.from_numpy(np.stack(frames)).to(dev)
        x = x.permute(0, 3, 1, 2).float() / 255.0
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = ((x - mean) / std).half()
        out = vis(pixel_values=x)
        if not a.grid:
            return out.pooler_output.float().cpu().numpy()          # (B, 1152)
        # patch token 排成 16x16（224/14），池化成 GxG 再摊平
        h = out.last_hidden_state                                    # (B, 256, 1152)
        B_, N, Cc = h.shape
        side = int(N ** 0.5)
        h = h.transpose(1, 2).reshape(B_, Cc, side, side)
        g = torch.nn.functional.adaptive_avg_pool2d(h, a.grid)       # (B, C, G, G)
        return g.reshape(B_, -1).float().cpu().numpy()               # (B, G*G*1152)

    # 扰动测试结论：亮度几乎无影响（SigLIP 归一化吃掉了），但几何扰动伤得很重 ——
    # 平移 3% 就让「切换时刻 ≤10 帧」从 68% 掉到 51%，缩放让 3/12 集卡死。
    # 所以增强以几何为主，亮度只作轻微陪衬。
    # 每**集**用一组固定的随机参数（而不是逐帧随机）：真机上相机位姿在一次运行内
    # 是不变的，逐帧抖动模拟的是不存在的情况，反而会让模型去学"忽略高频抖动"。
    def make_aug(seed):
        r = np.random.default_rng(seed)
        fx, fy = r.uniform(-0.05, 0.05, 2)
        zf = r.uniform(0.94, 1.08)
        br = r.uniform(0.8, 1.25)

        def f(im):
            # 缩放 + 平移用一条坐标映射搞定，边缘 clip 复制。
            # 别写成「先 crop 再 resize」：zf<1 时 crop 尺寸大于原图，
            # 起点算出负数、切片切成空数组（第一版就是这么崩的）。
            h, w = im.shape[:2]
            ys = np.clip(((np.arange(h) - h / 2) / zf + h / 2 - fy * h)
                         .round().astype(int), 0, h - 1)
            xs = np.clip(((np.arange(w) - w / 2) / zf + w / 2 - fx * w)
                         .round().astype(int), 0, w - 1)
            im = im[ys][:, xs]
            return np.clip(im.astype(np.float32) * br, 0, 255).astype(np.uint8)

        return f

    for eid in [int(x) for x in a.experts.split(",")]:
        ds, tag = EXPERTS[eid]
        dst = out / f"e{eid}"
        dst.mkdir(exist_ok=True)
        eps = sorted(glob.glob(str(B / ds / "data/chunk-000/*.parquet")))
        print(f"── {tag} {ds}  {len(eps)} 集")
        for ep_i, pf in enumerate(eps):
            fo = dst / f"ep{ep_i:04d}.npz"
            if fo.exists():
                continue
            d = pq.read_table(pf).to_pydict()
            n = len(d["observation.state"])
            state = np.asarray([np.asarray(s, dtype=np.float32) for s in d["observation.state"]])

            aug = make_aug(a.aug * 100003 + eid * 1009 + ep_i) if a.aug else None
            DIM = 1152 * (a.grid ** 2 if a.grid else 1)
            feats = np.zeros((n, len(VIEWS), DIM), dtype=np.float16)
            ok_view = []
            for vi, v in enumerate(VIEWS):
                vp = B / ds / f"videos/chunk-000/{v}/episode_{ep_i:06d}.mp4"
                if not vp.exists():
                    ok_view.append(False)
                    continue
                buf, idx, t = [], [], 0
                try:
                    with av.open(str(vp)) as c:
                        for fr in c.decode(video=0):
                            if t >= n:
                                break
                            im = fr.to_ndarray(format="rgb24")
                            buf.append(aug(im) if aug else im)
                            idx.append(t)
                            t += 1
                            if len(buf) == a.batch:
                                feats[idx, vi] = encode(buf).astype(np.float16)
                                buf, idx = [], []
                        if buf:
                            feats[idx, vi] = encode(buf).astype(np.float16)
                except Exception as e:
                    print("   x decode fail", v, ep_i, type(e).__name__, e, flush=True)
                ok_view.append(t > 0)
                if 0 < t < n:
                    print("   ! short decode", v, ep_i, t, "/", n, flush=True)
            caps = ok_view

            np.savez_compressed(fo, feat=feats, state=state.astype(np.float32),
                                view_mask=np.array(caps))
            if ep_i % 10 == 0:
                print(f"   {tag} {ep_i+1}/{len(eps)}  {n}帧", flush=True)
        print(f"  ✔ {tag} 完成 → {dst}")

    meta = {str(k): {"dataset": v[0], "tag": v[1]} for k, v in EXPERTS.items()}
    (out / "meta.json").write_text(json.dumps(
        {"experts": meta, "views": VIEWS,
         "dim": 1152 * (a.grid ** 2 if a.grid else 1), "grid": a.grid, "encoder": SIGLIP},
        ensure_ascii=False, indent=2))
    print(f"\n· 特征缓存 → {out}")


if __name__ == "__main__":
    main()
