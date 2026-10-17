# 默认模型与数据集配置

代码基准：`a5134b8a69e49a2c4bc3c5ace2bd58e40cc957c3`。

## 数据流

1. 冻结的 DinoTxt 提供视觉和类别文本特征，读取已有轨迹缓存。
2. 位置编码、CrossMotion、HoD 和 Pointformer 生成 Post 特征。
3. 类别文本构造 Support 帧原型；文本与 Support 视觉原型融合后，路由 Query。
4. Raw 特征提供语义质量 `m`，Post 特征与正负 Support 的度量提供独特性 `rho`。
5. `g = m * rho` 在双向 BiMHM 的帧匹配归约前控制匹配质量，得到 Query 类别分数。

模型只使用 Support 标签和 episode 候选类别。Query 真值用于损失与指标计算。

| 文件 | 职责 |
| --- | --- |
| `trokens/models/pointformer.py` | 特征提取、运动融合、时空注意力与类别路由 |
| `trokens/models/query_evidence_verifier.py` | 质量、独特性和证据门控 |
| `trokens/models/query_matching.py` | Support 参考、帧状态质量、双向匹配 |
| `tools/few_shot_multilabel.py` | 多标签监督、损失和诊断导出 |

## SAV 配置

入口为 `configs/trokens/sav.yaml`，默认输出为 `output/sav`。

- 训练协议：5-way、1-shot，每类 6 个训练 Query；每轮 2000 个训练 episode，10000 个测试 episode，20 轮训练。
- 轨迹：每段 8 帧、256 个点，空间路由采用 `QUERY_MASK_MODE: all`；运动和证据模块仍使用真实可见性。
- 路由：`FRAME_SOFTMAX_TAU: 0.04`，Support 文本与视觉融合保持开启。
- 消融：`QUALITY_ENABLE` 和 `UNIQUENESS_ENABLE` 独立控制两个判断。
- 监督：基础匹配损失权重 0.60，证据门控后的匹配损失权重 0.40，两个 MIL 损失权重各为 0.05。
- 优化：AdamW、cosine，学习率 `1e-4`，5 轮 warmup，梯度累积 2 步。
- 显存选项：`QUERY_CHUNK_SIZE`、`AMP_ENABLE` 和 `ACTIVATION_CHECKPOINT` 调整执行方式，不引入额外模型分支。

每个进程每步处理一个 episode；有效全局 episode 数为进程数乘以梯度累积步数。通用 `TRAIN/TEST.BATCH_SIZE` 不控制这里的 episode 大小。

SAV YAML 已移除旧评分阈值、Top-K 校准、旧模型选择项、关闭的时序细化、未使用的 AUG 配置，以及数据加载器不读取的通用视频预处理选项。RandAugment 由 `DATA.USE_RAND_AUGMENT` 独立控制，仍然开启。

共享配置保留其他数据集所需的 DINO、CLIP 等入口。关闭的时序相似度细化、文本对齐、相关性生成轨迹、HoD 时序金字塔、MixUp/CutMix 和额外编码器占位分支已从代码中移除。

## 运行与检查点

```bash
/home/duanwei/miniconda3/envs/trokens/bin/python tools/run_local.py \
  --cfg configs/trokens/sav.yaml --gpus 0
```

实际 GPU 数由启动参数决定，训练前按实验协议设置 `GRAD_ACCUM_STEPS`。

默认路径保留原参数名及检查点结构标识，可以加载对应结构的已有权重。旧启发式或其他结构的权重不能用于直接续训或评估；检查点加载会明确拒绝不完整或不匹配的证据头。

## TinyVIRAT 配置

入口为 `configs/trokens/tinyvirat.yaml`，输出为 `output/tinyvirat`。旧启发式路径已随清理删除，入口改用相同的默认质量／独特性模块；其旧实验结果和旧检查点不代表当前配置，需要重新训练。

配置按 SAV 的标准移除未使用与关闭的选项，保留 TinyVIRAT 的数据路径、21 类描述及顺序、类别划分和 3-shot 协议。空间路由使用 `QUERY_MASK_MODE: visibility`，测试使用 `DROP_EMPTY_QUERY_ROWS: True`。RandAugment 仍由 `DATA.USE_RAND_AUGMENT` 控制。
