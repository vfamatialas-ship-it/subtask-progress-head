#!/usr/bin/env python3
"""子任务自动切换控制器 —— 自包含部署版。

把整个 deploy/ 目录拷到机器人那台机器上，`from switcher import SubtaskSwitcher`
即可，所有路径都相对本文件解析，不依赖训练机上的任何目录。

用法::

    from switcher import SubtaskSwitcher

    sw = SubtaskSwitcher(expert=2)          # 0=E1右臂 1=E2左臂 2=E3封箱中段 3=E4封箱末段
    while True:
        obs = robot.get_obs()
        r = sw.step({"third_view":  obs.third_rgb,     # HxWx3 uint8 RGB
                     "right_wrist": obs.right_rgb,
                     "left_wrist":  obs.left_rgb},
                    obs.state)                          # 1-D float 关节/末端
        action = policy.infer(obs, prompt=r["prompt"])  # ← 用返回的 prompt
        robot.exec(action)
        if r["finished"]:            # 该专家做完了 → 上层计数器决定下一个专家
            break

    # 专家级调度：不需要分类器，用确定性计数器
    #   for cycle in range(6):  run_expert(0 if cycle%2==0 else 1)   # E1/E2 交替
    #   run_expert(2)   # 封箱中段
    #   run_expert(3)   # 封箱末段（仅四专家包 deploy4 有 E4）

设计的第一约束是**抗卡顿**
--------------------------
1. 模型**只吃当前帧** —— 不喂帧号、不喂已执行时长、不用任何时序模块。
   卡住时输入不变 → 输出不变 → 进度天然不推进。实测静止 30 帧漂移 0.0000。
2. **停滞门控**：state 几乎没变时冻结 EMA 与所有计数器，且只在信号还没过阈值时
   冻结 —— 信号已过线的静止说明「做完了停那儿」，该切就切；
   要防的是「卡顿期间 EMA 慢慢爬过阈值」这种假阳性。
3. 延迟补偿 D 按**推理步**计而非墙钟时间，同样受门控冻结。

决策链::

    SigLIP2 视觉塔 → 进度头 → 信号 → EMA(α) → 连续 K 步过 τ → 再等 D 步 → 切
                                                  ↑ 三者都受停滞门控冻结

τ / D / 兜底步数按 (专家, 子任务) 在**训练集**上标定，存在 deploy_cfg.json。
单步实测 7.1 ms（3 张图，Blackwell），约 140 Hz，远快于策略本身的 5~10 Hz。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent

# 各专家的 state 维度与子任务数**从权重文件里读**，不写死在这里 ——
# 写死会让同一份 switcher.py 只能配一种权重：三专家权重配四专家的常量，
# ProgressHead 会多建一组 state 编码器、专家嵌入也多一行，load_state_dict 直接报错。
STATE_DIM_FALLBACK = {0: 10, 1: 16, 2: 20, 3: 20}   # E1 末端 / E2 关节 / E3 / E4
N_SUB_MAX = 7
VIEW_KEY = {"third": "third_view", "right": "right_wrist", "left": "left_wrist"}

# prompt_index → 提示词。⚠ tasks.jsonl 里的 task_index **不是执行顺序**
# （E3 的执行序是 task_index [0,6,5,2,1,4,3]），这里已按 prompt_index 排好。
PROMPTS = {
    0: {
        1: "Use the right arm to grasp a box from the right-side area.",
        2: "Use the right arm to place the grasped box into an empty spot of the middle packing carton.",
    },
    1: {
        1: "Use the left arm to grasp a box from the left-side area.",
        2: "Use the left arm to place the grasped box into an empty spot of the middle packing carton.",
    },
    2: {
        1: "Move the right arm toward the lower part of the box's right side flap and brace the middle of the right side flap.",
        2: "Move the left arm toward the lower part of the box's left side flap and brace the middle of the left side flap.",
        3: "Use both arms at the same time to fold the left and right side flaps inward toward the carton opening, then press the flaps down until they lie flat.",
        4: "Move both arms away from the carton flaps and withdraw to a safe position, clearing space for the subsequent lower-flap pushing actions.",
        5: "Move the right arm toward the front part of the box's right lower flap and gently push that area.",
        6: "Move the left arm under the open flap, brace the left edge of the box's lower flap from the left-arm viewpoint, and gently push it.",
        7: "Move the right arm under the open flap, brace the left edge of the box's lower flap from the right-arm viewpoint, and gently push it.",
    },
    3: {  # E4 封箱末段。直接取自数据集的 prompt_text 列 ——
          # 这个数据集自带逐帧提示词，不必像 E3 那样做 task_index → 执行顺序的映射
        1: "Raise the right arm, move it over the flaps, and bring it close to the top surface of the carton.",
        2: "Open the right gripper and lower it so that the two fingers brace the front and rear flaps.",
        3: "Use the left arm to lift the left flap from the left side until it is nearly vertical, while the right arm keeps bracing the front and rear flaps.",
        4: "Use the left arm to fold the left flap inward and close it, while the right arm keeps bracing the front and rear flaps.",
        5: "Retract the right arm from above the carton and clear the flap area.",
        6: "Use the right arm to lift the right flap from the right side until it is nearly vertical.",
        7: "Use the right arm to fold the right flap inward and close it.",
    },
}

SIG = {
    "p":    lambda p, d, t: p,
    "done": lambda p, d, t: d,
    "ttb":  lambda p, d, t: t,
    "both": lambda p, d, t: min(p, d),
    "pt":   lambda p, d, t: min(p, t),
    "all":  lambda p, d, t: min(p, d, t),
}


class ProgressHead(nn.Module):
    """图像特征 + state + (专家, 子任务) 条件 → 进度 p / 段末 done / 距边界 ttb。

    state 按专家各配一个编码器：三个专家维度不同(10/16/20)，且同一下标语义不同，
    padding 成同一张量会互相污染。
    """

    def __init__(self, n_view, state_dim=None, d_img=1152, d_hid=512, d_cond=64):
        super().__init__()
        self.state_dim = dict(state_dim or STATE_DIM_FALLBACK)
        self.img = nn.Sequential(nn.LayerNorm(d_img * n_view),
                                 nn.Linear(d_img * n_view, d_hid), nn.GELU())
        self.state_enc = nn.ModuleDict({
            str(k): nn.Sequential(nn.LayerNorm(v), nn.Linear(v, 128), nn.GELU())
            for k, v in self.state_dim.items()})
        self.emb_expert = nn.Embedding(len(self.state_dim), d_cond)
        self.emb_sub = nn.Embedding(N_SUB_MAX + 1, d_cond)
        self.trunk = nn.Sequential(
            nn.Linear(d_hid + 128 + 2 * d_cond, d_hid), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d_hid, 256), nn.GELU())
        self.head_p = nn.Linear(256, 1)
        self.head_done = nn.Linear(256, 1)
        self.head_ttb = nn.Linear(256, 1)

    def forward(self, img, state, eid, sub):
        h = self.img(img)
        s = torch.zeros(img.shape[0], 128, device=img.device, dtype=h.dtype)
        for k, dim in self.state_dim.items():
            m = eid == k
            if m.any():
                s[m] = self.state_enc[str(k)](state[m, :dim]).to(h.dtype)
        z = self.trunk(torch.cat([h, s, self.emb_expert(eid), self.emb_sub(sub)], -1))
        return (self.head_p(z).squeeze(-1), self.head_done(z).squeeze(-1),
                self.head_ttb(z).squeeze(-1))


class SubtaskSwitcher:
    """单帧推理 + 停滞门控 + 迟滞 + 漏切兜底。"""

    def __init__(self, expert: int, pkg=HERE, device="cuda",
                 stall_state_eps=1e-5, stall_img_eps=0.032, stall_use_image=False,
                 lockout=10, verbose=True):
        """
        stall_state_eps  state 最大分量变化低于此值算「没动」。**真机主判据** ——
                         网络卡顿时机械臂停住、编码器读数恒定，这是最直接的信号。
        stall_img_eps    画面平均逐像素差(0~255)低于此值算「没动」。
                         ⚠ 默认**不启用**。0.032 是在压缩视频上标定的 ——
                         H.264 把静止区域量化成完全相同的块，噪声被抹掉了。
                         真机实时相机的传感器噪声就能让静止画面的逐像素差超过 1，
                         用这个阈值门控**永远不会触发**。要用先跑 calibrate_stall()。
        lockout          切换后多少步内不允许再切（迟滞，防抖动）
        """
        pkg = Path(pkg)
        self.cfg = json.loads((pkg / "deploy_cfg.json").read_text())
        self.expert, self.device, self.verbose = expert, device, verbose
        self.stall_state_eps, self.stall_img_eps = stall_state_eps, stall_img_eps
        self.stall_use_image, self.lockout = stall_use_image, lockout
        # 先查包里到底有哪些专家，再取配置 —— 否则不支持的 expert 会先在
        # cfg["n_sub"] 上抛 KeyError，报错信息看不出真正原因
        ck = torch.load(pkg / "progress_head.pt", map_location=device, weights_only=False)
        sd = {int(k): int(v) for k, v in ck.get("state_dim", STATE_DIM_FALLBACK).items()}
        have = sorted(set(sd) & {int(k) for k in self.cfg["n_sub"]})
        if expert not in have:
            raise ValueError(f"这个包只含专家 {[e+1 for e in have]}（E 编号），"
                             f"不支持 expert={expert}（E{expert+1}）")
        self.n_sub = int(self.cfg["n_sub"][str(expert)])
        self.views = self.cfg["views"]
        self.state_dim = sd[expert]
        self.net = ProgressHead(len(self.views), state_dim=sd).to(device)
        self.net.load_state_dict(ck["model"]); self.net.eval()

        from transformers import SiglipImageProcessor, SiglipVisionModel
        mp = pkg / "siglip2_vision"
        proc = SiglipImageProcessor.from_pretrained(mp)
        self.vis = SiglipVisionModel.from_pretrained(
            mp, torch_dtype=torch.float16).to(device).eval()
        self.mean = torch.tensor(proc.image_mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(proc.image_std, device=device).view(1, 3, 1, 1)
        self.reset()

    def reset(self, sub: int = 1):
        self.sub = sub
        self.ema = None
        self.hits = 0          # 连续过阈值的步数
        self.pending = None    # 已触发，正在等 D 步
        self.steps = 0         # 本子任务已走的步数（受门控冻结）
        self.since_switch = 10 ** 9
        self._prev_img = self._prev_state = None
        self.finished = False

    def _key(self):
        return f"{self.expert}_{self.sub}"

    @property
    def prompt(self) -> str:
        return PROMPTS[self.expert][self.sub]

    @torch.no_grad()
    def _forward(self, images, state):
        frames = [images[VIEW_KEY[v]] for v in self.views]
        x = torch.from_numpy(np.stack(frames)).to(self.device)
        x = x.permute(0, 3, 1, 2).float() / 255.0
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear",
                                            align_corners=False)
        x = ((x - self.mean) / self.std).half()
        feat = self.vis(pixel_values=x).pooler_output.float().reshape(1, -1)
        s = torch.zeros(1, 20, device=self.device)
        s[0, :self.state_dim] = torch.as_tensor(
            np.asarray(state, np.float32)[:self.state_dim], device=self.device)
        o = self.net(feat, s, torch.tensor([self.expert], device=self.device),
                     torch.tensor([self.sub], device=self.device))
        return float(o[0]), float(torch.sigmoid(o[1])), float(1.0 - o[2])

    def step(self, images: dict, state) -> dict:
        if self.finished:
            return dict(prompt=self.prompt, subtask=self.sub, progress=1.0,
                        switched=False, stalled=False, frozen=False,
                        forced=False, finished=True)

        p, done, ttb = self._forward(images, state)
        raw = SIG[self.cfg["signal"]](p, done, ttb)
        tau = self.cfg["tau"][self._key()]

        cur_state = np.asarray(state, np.float32)
        cur_img = images[VIEW_KEY[self.views[0]]].astype(np.float32)
        stalled = False
        if self._prev_state is not None:
            stalled = float(np.abs(cur_state - self._prev_state).max()) < self.stall_state_eps
            if self.stall_use_image:
                stalled = stalled and float(
                    np.abs(cur_img - self._prev_img).mean()) < self.stall_img_eps
        self._prev_state, self._prev_img = cur_state, cur_img

        # ★ 只在「信号尚未过线」时冻结
        freeze = stalled and raw <= tau
        if not freeze:
            a = self.cfg["alpha"]
            self.ema = raw if self.ema is None else a * raw + (1 - a) * self.ema
            self.steps += 1
            self.since_switch += 1
            self.hits = self.hits + 1 if self.ema > tau else 0
            if self.pending is not None:
                self.pending -= 1

        switched = forced = False
        if (self.pending is None and self.hits >= self.cfg["K"]
                and self.since_switch >= self.lockout):
            self.pending = int(self.cfg["delay"][self._key()])
        if (self.pending is None and not freeze
                and self.steps > int(self.cfg["max_steps"][self._key()])):
            self.pending, forced = 0, True
            if self.verbose:
                print(f"[switch] ⚠ E{self.expert+1}s{self.sub} 超过 "
                      f"{self.cfg['max_steps'][self._key()]} 步未触发，强制推进")

        if self.pending is not None and self.pending <= 0:
            switched = True
            if self.sub >= self.n_sub:
                self.finished = True
                if self.verbose:
                    print(f"[switch] E{self.expert+1} 全部 {self.n_sub} 个子任务完成")
            else:
                if self.verbose:
                    print(f"[switch] E{self.expert+1}: s{self.sub} → s{self.sub+1}"
                          f"{' (兜底)' if forced else ''}  ({self.steps} 步)")
                self.sub += 1
                self.ema = None; self.hits = 0
                self.steps = 0; self.since_switch = 0
            self.pending = None

        return dict(prompt=self.prompt, subtask=self.sub, progress=p, signal=raw,
                    ema=self.ema, tau=tau, hits=self.hits, pending=self.pending,
                    switched=switched, stalled=stalled, frozen=freeze,
                    forced=forced, finished=self.finished)


def calibrate_stall(still_frames, still_states, moving_frames, moving_states):
    """在**真机**上标定停滞阈值。

    采两段数据：一段机械臂完全静止（断开控制让它停住即可），一段正常运动。
    阈值取「运动分布 p1」与「静止分布 p99」的几何中点。

    若两个分布重叠（静止 p99 > 运动 p1），说明该通道在你的硬件上区分不了停滞 ——
    相机噪声大时图像通道常常如此，此时就只用 state。
    """
    def diffs(fr, st):
        fr = np.asarray(fr, np.float32); st = np.asarray(st, np.float32)
        return (np.abs(np.diff(fr, axis=0)).mean((1, 2, 3)),
                np.abs(np.diff(st, axis=0)).max(1))

    si, ss = diffs(still_frames, still_states)
    mi, ms = diffs(moving_frames, moving_states)
    out = []
    for name, s_, m_ in (("图像", si, mi), ("state", ss, ms)):
        lo, hi = np.percentile(s_, 99), np.percentile(m_, 1)
        if lo >= hi:
            print(f"  ⚠ {name} 通道区分不了停滞：静止 p99={lo:.4f} ≥ 运动 p1={hi:.4f}，"
                  f"建议不要用这个通道")
            out.append(None)
        else:
            eps = float(np.sqrt(max(lo, 1e-9) * hi))
            print(f"  {name}: 静止 p99={lo:.4f}  运动 p1={hi:.4f}  → 阈值 {eps:.5f}")
            out.append(eps)
    return tuple(out)
