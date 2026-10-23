# 默认模型与数据集配置

当前 SAV 为[双轴单文本拼接第 1 版](text_concat_st.md)（2026-10-23）。
历史前置对照仍保留在 `sav_label_pre_temporal*.yaml`；TinyVIRAT 未改结构。

## 数据流

1. 冻结的 DinoTxt 提供视觉和类别文本特征，读取已有轨迹缓存。
2. 保留位置编码、CrossMotion、HoD；Support、Query 按每个候选类使用相同原始文本。
3. 时间轴拼 1 个文本，空间轴拼 1 个文本，进行原有串行时空 self-attention。
4. 复用空间 self-attention 的文本→视觉权重，聚合最终 Post 视觉 patch；正 Support 类内平均。
5. 关闭质量 `m`、独特性 `rho` 及 transport，直接用原 BiMHM 得到 Query 类别分数。

模型只使用 Support 标签和 episode 候选类别。Query 真值用于损失与指标计算。

| 文件 | 职责 |
| --- | --- |
| `trokens/models/pointformer.py` | 特征提取、运动融合、时空注意力与类别路由 |
| `trokens/models/text_concat_conditioning.py` | 当前双轴文本拼接、空间聚合、Support/Query 匹配 |
| `trokens/models/query_evidence_verifier.py` | 历史对照及其他数据集的质量、独特性判断 |
| `trokens/models/query_matching.py` | Support 参考、帧状态质量、双向匹配 |
| `tools/few_shot_multilabel.py` | 多标签监督、损失和诊断导出 |

## SAV 配置

入口为 `configs/trokens/sav.yaml`，默认输出为 `output/sav_text_concat_t1_s1_attention_readout_m_rho_off`。
原 32 份版本保留为 `configs/trokens/sav_text_concat_s32.yaml`，两种 token 数量的 checkpoint 不可混用。

- 训练协议：5-way、1-shot，每类 6 个训练 Query；每轮 2000 个训练 episode，10000 个测试 episode，20 轮训练。
- 轨迹：每段 8 帧、256 个点，空间读出采用 `QUERY_MASK_MODE: all`；运动仍使用真实可见性。
- 编码：启用 `TEXT_CONCAT_ST`，关闭旧前置路由及 Support/text 融合，不额外计算相似度。
- 消融：本版 `QUALITY_ENABLE=False`、`UNIQUENESS_ENABLE=False`，不构建两个判断头。
- 监督：全局分类 BCE + 匹配 BCE；保留的 `.60/.40` 匹配项因 logits 相同而等价于一份 BCE；两个 MIL 停用。
- 优化：AdamW、cosine，学习率 `1e-4`，5 轮 warmup，梯度累积 2 步。
- 显存选项：`TEXT_CONCAT_ST` 内的类别/视频分块和完整块 activation checkpoint 调整执行方式。
- 提速：缓存冻结文本塔输出，保留投影和提示词平均逻辑；默认分块 2×4、完整块 checkpoint 不变。
  可选 `sav_text_concat_s1_fast.yaml` 使用 5×4 分块；[GPU 3 短测](text_concat_speed_gpu3.md)
  约减少 5% 单文本版本的计算耗时，峰值 allocated 从 10.15 增至 12.09 GiB。

每个进程每步处理一个 episode；有效全局 episode 数为进程数乘以梯度累积步数。通用 `TRAIN/TEST.BATCH_SIZE` 不控制这里的 episode 大小。

SAV YAML 已移除旧评分阈值、Top-K 校准、旧模型选择项、关闭的时序细化、未使用的 AUG 配置，以及数据加载器不读取的通用视频预处理选项。RandAugment 由 `DATA.USE_RAND_AUGMENT` 独立控制，仍然开启。

共享配置保留其他数据集所需的 DINO、CLIP 等入口。关闭的时序相似度细化、文本对齐、相关性生成轨迹、HoD 时序金字塔、MixUp/CutMix 和额外编码器占位分支已从代码中移除。

## 运行与检查点

```bash
/home/duanwei/miniconda3/envs/trokens/bin/python tools/run_local.py \
  --cfg configs/trokens/sav.yaml --gpus 0
```

实际 GPU 数由启动参数决定，训练前按实验协议设置 `GRAD_ACCUM_STEPS`。

当前路径保存结构及文本数量签名。仅可加载同结构的完整 checkpoint；不能用旧前置路由
checkpoint 直接续训或评估。新实验应从头训练。

## TinyVIRAT 配置

入口为 `configs/trokens/tinyvirat.yaml`，输出为 `output/tinyvirat`。旧启发式路径已随清理删除，入口改用相同的默认质量／独特性模块；其旧实验结果和旧检查点不代表当前配置，需要重新训练。

配置按 SAV 的标准移除未使用与关闭的选项，保留 TinyVIRAT 的数据路径、21 类描述及顺序、类别划分和 3-shot 协议。空间路由使用 `QUERY_MASK_MODE: visibility`，测试使用 `DROP_EMPTY_QUERY_ROWS: True`。RandAugment 仍由 `DATA.USE_RAND_AUGMENT` 控制。
