# 数据说明

## labels.npy
从四个数据集的 `prompt_index` 逐帧游程直接导出，**零人工标注**。
列：`expert_id, episode, frame, subtask_index, seg_i, n_seg, p_t, done_t, event_anchor`

| 专家 | 数据集 | 集数 | 帧数 | 段数 | 顺序异常 |
|---|---|---|---|---|---|
| E1 右臂抓放 | nero_right_box_pick_ee_v1 | 100 | 41,034 | 200 | 0 |
| E2 左臂抓放 | nero_left_box_pick_v2_sub | 100 | 38,087 | 200 | 0 |
| E3 封箱中段 | nero_hezi_closing_ee_v1 | 100 | 87,649 | 700 | 0 |
| E4 封箱末段 | nero_stage56_flap_closing_ee_v2 | 69 | 70,187 | 483 | 0 |

合计 236,957 帧，事件锚点数与段数完全对齐。

## SigLIP2 特征缓存（未拷，约 2 G，可重建）
在 `<SWITCH_ROOT>/feats/`（干净）与 `feats_aug1/`（几何增强）。
```bash
CUDA_VISIBLE_DEVICES=2 python tools/cache_feats.py --experts 2,0,1,3
CUDA_VISIBLE_DEVICES=2 python tools/cache_feats.py --experts 2,0,1,3 --aug 1 --out feats_aug1
```
⚠ 解码必须用 PyAV：两路腕部是 AV1，cv2 会**静默**失败并产出全零特征。
