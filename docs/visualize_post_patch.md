# 后解耦：单帧文本 / patch 可视化

入口：`tools/visualize_post_patch.py`。仅推理，不训练，不切换当前工作区，不修改 SAV 配置。

## 按视频真实类别画完整 8 帧（用户指定）

同一个 Query 视频 `20220731_33_1113_11.mp4` 的实际视频级标签是
`stand`（站立）、`look_forward`（朝前看）、`bend`（弯腰），不是 episode 的 5 个候选类。

增加 `--video-labels`，输出目录：`output/sav_post_patch_viz_a5134b8_ep0_q0_true_labels/`。

- `all_frames_cosine.png`：原图 + 3 个真实类别，每行完整 8 帧，展示纯文本/Post cosine。
- `all_frames_weights.png`：同一布局，展示对应的纯文本 softmax 权重。
- `class_stand_8frames.png`、`class_look_forward_8frames.png`、`class_bend_8frames.png`：
  每类单独一图，3 行依次为原图、cosine、softmax，每行 8 帧。

这里统一画纯文本诊断分数，不冒充实际 Support 融合 Query 权重；`bend` 不在原 episode
候选类中，不为它虚构 Support 原型。真实标签仅在正常前向后用于选择展示行，不修改模型输入条件。
各类别和 8 帧共用色标。`patch_scores.csv` 和 `patch_scores.npz` 保存全部 3×8×256 个数值。

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/home/duanwei/miniconda3/envs/trokens/bin/python \
  /data3/duanwei/trokens-main/tools/visualize_post_patch.py \
  --gpu 3 --episode 0 --query 0 --frame 3 --video-labels \
  --output /data3/duanwei/trokens-main/output/sav_post_patch_viz_a5134b8_ep0_q0_true_labels_repeat
```

## 更多视频（真实标签，不是 episode 候选类别）

批量入口：`tools/visualize_post_patch_batch.py`。输出目录：
`output/sav_post_patch_viz_a5134b8_more_videos/`。
`index.html` 提供相似度总览、softmax 总览、每类 8 帧大图及 CSV 链接；
`video_contact_sheet.png` 是 5 个新增视频的原始 8 帧预览。

样本从 seed=1 的前 20 个测试 episode（100 个 Query 候选）按标签覆盖选择：
先最大化新增类别数，再最大化与已选组合的最小 Jaccard 距离，同分取最早 episode/query。
排除原视频和重复标签组合，不看模型分数或图像效果。新增 5 个加原视频共覆盖全部 15 类。

| 输出子目录 | 该视频自身的标签 |
| --- | --- |
| `01_ep0_q2` | sit, look_sideways, touch_sth, turn_around |
| `02_ep0_q4` | sit, read, flip_books, take_notes |
| `03_ep2_q4` | sit, look_forward, hands_down, answer_questions |
| `04_ep11_q2` | sit, look_sideways, raise_hand, talk_with_others |
| `05_ep4_q1` | sit, look_sideways, applaud |

**同一视频内**各类别、各帧共用色标；不同视频的色标上下界可能不同，跨视频不要只比较颜色。
每个视频的原始分数、色标范围、权重 SHA256、prompt 与视频路径都单独保存在结果目录。
新增图不覆盖此前原视频的结果。

```bash
/home/duanwei/miniconda3/envs/trokens/bin/python \
  /data3/duanwei/trokens-main/tools/visualize_post_patch_batch.py \
  --gpu 2 --samples 0:2 0:4 2:4 11:2 4:1 \
  --output /data3/duanwei/trokens-main/output/sav_post_patch_viz_a5134b8_more_videos_repeat
```

GPU 编号需按运行时空闲情况选择；批量任务串行推理，不启动训练。

## 权重来源

使用提交 `a5134b8a69e49a2c4bc3c5ace2bd58e40cc957c3` 的代码临时快照，以及
`output/sav_semantic_relation_v3/checkpoints/best_by_run/checkpoint_best_run_20261011_133133.pyth`。
该归档的 run ID、epoch 20 与 `test_20261011_193326.log` 对齐。
同目录的通用 `checkpoint_best.pyth` 后来已被其他实验覆盖，不用于此图。

所有 65 个已保存模型 tensor 都已加载；645 个未保存在训练 checkpoint 中的冻结
DinoTxt tower tensor 由本地预训练文件加载，与历史测试日志一致。

## 已生成示例

输出：`output/sav_post_patch_viz_a5134b8_ep0_q0/`。
固定 seed=1、test episode=0、query=0，没有根据预测好坏筛选。
完整输入仍是 8 帧，默认重点展示第 4 个采样帧（`--frame 3`）。

- `single_frame_focus.png`：原图与 5 个 episode 类别的实际 Query 路由，便于快速查看。
- `single_frame.png`：详细对照；每行一个类别，依次为原图、纯文本/Post cosine、纯文本 softmax、融合 Support 后 cosine、实际 Query softmax。
- `frame_00.png` 到 `frame_07.png`：逐帧详细对照。
- `all_frames.png`：8 帧 × 5 类的实际 Query 权重总览。
- `supports.png`：参与该 episode 的 Support 图像及 episode 内标签。
- `patch_scores.csv` / `patch_scores.npz`：原始数值、坐标、可见性、原跟踪点编号及 transport 后的权重。
- `provenance.json`：提交、checkpoint SHA256、样本、完整 prompt 组、标签、数值校验结果。
- `checkpoint_config.yaml`：checkpoint 保存的完整配置。

## 如何读图

1. Support 匹配原型用纯文本在 Post 特征上 readout；Query 实际使用融合后的
   `normalize(text + normalize(mean_t(Support prototype)))`，温度为 0.04。
   纯文本 Query 图是用于比较的诊断图，不冒充实际 Query 分布。
2. 256 个 token 是轨迹采样位置，按各帧 `pred_tracks` 叠图，不能直接 reshape 为 16×16 网格。
   点可能重合、不可见或超出画面；原始 CSV 全部保留。图不做空间插值或平滑。
3. 所有类别和帧共用同一线性色标。青色圈是 top-8 展示辅助，不代表模型硬选了 8 个点。
   `x` 表示跟踪器判为不可见；历史 `QUERY_MASK_MODE=all` 仍允许这些 token 参与主路由。
4. 实际 Query softmax 图是 m/rho 验证前的条件权重；rho 对可见性重新掩码，
   m/rho 再影响帧级 transport。`query_transport_weights` 单独保存在数值文件中。
5. 标签是视频级标签，不是该帧的动作或区域真值。即使某类不存在，softmax 仍然归一化；
   高亮和集中程度不能单独证明动作存在或定位正确。Post token 已经混合时空上下文。

该示例中重算权重及聚合原型与真实 forward 的最大绝对误差均为 0。
第 4 帧 `stand` 的 top-8 权重和约 65.0%，`read` 约 59.4%，但该视频标签不含 `read`。
这只说明两种条件的关注位置/集中程度不同，不是定位准确率指标。

## 重跑 / 更换样本

先确认目标 GPU 空闲。指定新的输出目录（已有目录会拒绝覆盖）：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/home/duanwei/miniconda3/envs/trokens/bin/python \
  /data3/duanwei/trokens-main/tools/visualize_post_patch.py \
  --gpu 3 --episode 0 --query 0 --frame 3 \
  --output /data3/duanwei/trokens-main/output/sav_post_patch_viz_a5134b8_ep0_q0_repeat
```

`--episode`、`--query` 和 `--frame` 都从 0 开始；5-way 1-shot 时 query 通常为 0–4，frame 为 0–7。
程序会自动提取历史代码，并在结束时清理自己的临时代码目录。
