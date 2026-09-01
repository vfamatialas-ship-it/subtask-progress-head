#!/usr/bin/env python3
"""把切换头打成自包含的部署包：拷到机器人那台机器上就能跑。

现在的 switch_ctl.py 硬编码了三条本机路径 —— 权重、deploy_cfg、以及
**别人目录下**的 SigLIP2（<WORK_ROOT>/...）。换台机器就起不来。

包里只放视觉塔（vision tower），不放 text model：4.3G 的原始目录里
文本塔占了一大半，而这套方案从不做文本编码。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

SRC_RUN = Path("<SWITCH_ROOT>/runs/p_third_right_left_e012")
SIGLIP = "<WORK_ROOT>/RLinf_nero/hf_models/siglip2-so400m-patch14-224"
OUT = Path("<SWITCH_ROOT>/deploy")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "siglip2_vision").mkdir(exist_ok=True)

# ── 只导出视觉塔 ──────────────────────────────────────────────────────
from transformers import SiglipVisionModel, SiglipImageProcessor

vis = SiglipVisionModel.from_pretrained(SIGLIP, torch_dtype=torch.float16)
vis.save_pretrained(OUT / "siglip2_vision", safe_serialization=True)
SiglipImageProcessor.from_pretrained(SIGLIP).save_pretrained(OUT / "siglip2_vision")
print(f"· 视觉塔 → {OUT/'siglip2_vision'}")

# ── 权重与决策参数 ────────────────────────────────────────────────────
ck = torch.load(SRC_RUN / "best.pt", map_location="cpu", weights_only=False)
torch.save({k: ck[k] for k in ("model", "views", "experts", "state_dim", "n_sub")},
           OUT / "progress_head.pt")
shutil.copy(SRC_RUN / "deploy_cfg.json", OUT / "deploy_cfg.json")
print(f"· 进度头权重 + 决策参数 → {OUT}")

sz = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 2**30
print(f"· 包大小 {sz:.2f} G")
print(f"\n拷贝方式：  scp -r <USER>:{OUT} /你的机器人目录/")
