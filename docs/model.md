# 默认模型与数据集配置

当前 SAV 使用[统一的四项消融开关](sav_ablation.md)，默认 Support / Query 文本解耦及 m/ρ 全关闭。
历史前置对照仍保留在 `sav_label_pre_temporal*.yaml`；TinyVIRAT 未改结构。

## 默认全关闭组合的数据流

1. 冻结的 DinoTxt 提供视觉特征，读取已有轨迹缓存；不调用类别文本编码。
2. 保留位置编码、CrossMotion、HoD；Support / Query 都不使用文本。
3. 每个视频只运行一次普通时空 self-attention，两侧共享参数，不复制类别分支。
4. 两侧每帧对 Norm 后的视觉 patch 求均值，正 Support 类内平均；视觉 CLS 单独用于全局分类。
5. 一份 Query 特征与全部 Support 类原型进行 BiMHM 匹配，保留 `10*S-2` 分数尺度；
   不构建 m/ρ 判断头，不做其匹配修正或辅助损失。

模型只使用 Support 标签和 episode 候选类别。Query 真值用于损失与指标计算。

| 文件 | 职责 |
| --- | --- |
| `trokens/models/pointformer.py` | 特征提取、运动融合、时空注意力与类别路由 |
| `trokens/models/text_concat_conditioning.py` | episode 接口、文本拼接/通用编码消融、空间聚合和匹配 |
| `trokens/models/query_evidence_verifier.py` | Raw 质量、Post 独特性判断及匹配修正 |
| `trokens/models/query_matching.py` | Support 参考、帧状态质量、双向匹配 |
| `tools/few_shot_multilabel.py` | 多标签监督、损失和诊断导出 |

## SAV 配置

入口为 `configs/trokens/sav.yaml`，默认输出为 `output/sav`。
独立消融入口 `configs/trokens/sav_shared_support_query.yaml` 输出 `output/sav_shared_support_query_m_rho_off`。
原仅 Support 拼接版保留在 `configs/trokens/sav_text_concat_shared_query.yaml`。
原双侧拼接 m/ρ 开启版保留在 `configs/trokens/sav_text_concat_s1_m_rho_on.yaml`，仍输出 `output/savo`。
原单文本关闭版固定在 `configs/trokens/sav_text_concat_s1_base.yaml`，有效配置与切换前一致。
原 32 份版本保留为 `configs/trokens/sav_text_concat_s32.yaml`，两种 token 数量的 checkpoint 不可混用。

- 训练协议：5-way、1-shot，每类 6 个训练 Query；每轮 2000 个训练 episode，10000 个测试 episode，20 轮训练。
- 轨迹：每段 8 帧、256 个点，空间读出采用 `QUERY_MASK_MODE: all`；运动仍使用真实可见性。
- 编码：`DECOUPLING.QUERY_ENABLE` / `SUPPORT_ENABLE` 独立控制两侧；当前均关闭，使用通用编码。
- 判断：`DECOUPLING.M_ENABLE` / `RHO_ENABLE` 当前均关闭，匹配修正及两个 MIL 权重自动归零。
- 监督：global BCE + base BCE；base/final 分数相同，不重复计算 final BCE。
- 优化：AdamW、cosine，学习率 `1e-4`，5 轮 warmup，梯度累积 2 步。
- 执行：每块 8 视频，关闭完整块 checkpoint，仅保留 MLP/readout 重算。
  双侧通用模式下不运行类别分块、文本编码/缓存或文本权重读出；配置中的相关旧值不生效。
  原 fast 配置仍固定为双侧拼接、m/ρ 关闭版。

内部 `TEXT_CONCAT_ST.ENABLE`、shared 和 loss 标志由四个公开开关统一生成，不在 sav 中重复配置。
这是当前框架的无解耦 baseline，仍使用 `10*S-2` 和 base BCE，未回退到历史 ReLU/负双向距离打分。

每个进程每步处理一个 episode；有效全局 episode 数为进程数乘以梯度累积步数。通用 `TRAIN/TEST.BATCH_SIZE` 不控制这里的 episode 大小。

SAV YAML 已移除旧评分阈值、Top-K 校准、旧模型选择项、关闭的时序细化、未使用的 AUG 配置，以及数据加载器不读取的通用视频预处理选项。RandAugment 由 `DATA.USE_RAND_AUGMENT` 独立控制，仍然开启。

共享配置保留其他数据集所需的 DINO、CLIP 等入口。关闭的时序相似度细化、文本对齐、相关性生成轨迹、HoD 时序金字塔、MixUp/CutMix 和额外编码器占位分支已从代码中移除。

## 运行与检查点

```bash
/home/duanwei/miniconda3/envs/trokens/bin/python tools/run_local.py \
  --cfg configs/trokens/sav.yaml --gpus 0
```

实际 GPU 数由启动参数决定，训练前按实验协议设置 `GRAD_ACCUM_STEPS`。

当前双侧通用路径签名为 `[5,0,0,1]`。真正续训/评估必须加载同结构完整参数；
不能直接加载旧前置路由或文本拼接 v1/v2/v3/v4 checkpoint，本组配置默认从头训练。
原 v2 的显式 v1 初始化规则仍仅适用于保留的 m/ρ 开启版，不适用于当前双侧通用路径。

## TinyVIRAT 配置

入口为 `configs/trokens/tinyvirat.yaml`，输出为 `output/tinyvirat`。旧启发式路径已随清理删除，入口改用相同的默认质量／独特性模块；其旧实验结果和旧检查点不代表当前配置，需要重新训练。

配置按 SAV 的标准移除未使用与关闭的选项，保留 TinyVIRAT 的数据路径、21 类描述及顺序、类别划分和 3-shot 协议。空间路由使用 `QUERY_MASK_MODE: visibility`，测试使用 `DROP_EMPTY_QUERY_ROWS: True`。RandAugment 仍由 `DATA.USE_RAND_AUGMENT` 控制。
