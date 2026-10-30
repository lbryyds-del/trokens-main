# 默认模型与数据集配置

当前 SAV 与 TinyVIRAT 使用[统一的四项消融开关](sav_ablation.md)，默认 Support / Query 文本解耦及 m/ρ(p) 全关闭，HoD / CTD（CrossMotion）全开启。
两份主配置的公共训练/模型设置一致；历史对照配置未修改。

## 默认无解耦、双运动模块数据流

1. 冻结的 DinoTxt 提供视觉特征，读取已有轨迹缓存；不加载类别文本编码器。
2. 外观特征叠加 CrossMotion、HoD 运动特征，保留位置编码。
3. Support / Query 均走原始 `pt_forward`，不拼文本、不复制类别分支。
4. 正 Support 类内先平均视觉 patch；Support 原型和 Query 各自经过 ReLU，再做空间均值。
5. 使用原始负双向最近帧距离匹配，不使用 `10*S-2`；保留全局分类和原始匹配 BCE，
   不创建 m/ρ 判断头，不做匹配修正或辅助损失。

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
- 轨迹：每段 8 帧、256 个点，空间读出采用 `QUERY_MASK_MODE: all`；轨迹缓存和独立可见性信息保留。
- 编码：`DECOUPLING.QUERY_ENABLE` / `SUPPORT_ENABLE` 独立控制两侧，当前均关闭。
- 判断：`DECOUPLING.M_ENABLE` / `RHO_ENABLE` 当前均关闭，transport 和两个辅助权重自动归零。
- 运动：HoD 和 CTD（CrossMotion）均开启。
- 监督：`L_global + L_original_q2s_BCE`，无重复 final BCE。
- 优化：AdamW、cosine，学习率 `1e-4`，5 轮 warmup，梯度累积 2 步。
- 执行：原始支路不使用 `TEXT_CONCAT_ST` 的分块/checkpoint 或文本缓存设置。
  历史 fast 配置仍固定为双侧拼接、m/ρ 关闭版。

内部 `TEXT_CONCAT_ST.ENABLE`、shared 和 loss 标志由四个公开开关统一生成，不在 sav 中重复配置。
当前为无解耦、含运动模块的原始匹配 baseline；不代表复现历史整套训练协议。详见消融说明。

每个进程每步处理一个 episode；有效全局 episode 数为进程数乘以梯度累积步数。通用 `TRAIN/TEST.BATCH_SIZE` 不控制这里的 episode 大小。

SAV YAML 已移除旧评分阈值、Top-K 校准、旧模型选择项、关闭的时序细化、未使用的 AUG 配置，以及数据加载器不读取的通用视频预处理选项。RandAugment 由 `DATA.USE_RAND_AUGMENT` 独立控制，仍然开启。

共享配置保留其他数据集所需的 DINO、CLIP 等入口。关闭的时序相似度细化、文本对齐、相关性生成轨迹、HoD 时序金字塔、MixUp/CutMix 和额外编码器占位分支已从代码中移除。

## 运行与检查点

```bash
/home/duanwei/miniconda3/envs/trokens/bin/python tools/run_local.py \
  --cfg configs/trokens/sav.yaml --gpus 0,1 \
  --output-dir output/sav_no_decoupling_hod_ctd_1shot
```

实际 GPU 数由启动参数决定，训练前按实验协议设置 `GRAD_ACCUM_STEPS`。

真正续训/评估必须加载同结构完整参数，角色/判断开关及架构签名需要一致。
本组配置默认从头训练，不自动恢复旧 checkpoint；新实验请使用未占用的独立输出目录。
当前不应把文本解耦、m/ρ 开启或关闭运动模块的模型权重直接当作同结构续训。

## TinyVIRAT 配置

入口为 `configs/trokens/tinyvirat.yaml`，默认输出仍为 `output/tinyvirat`。
Tiny 已对齐 SAV 的 5-way 1-shot、双侧文本解耦及 m/ρ 关闭、HoD/CTD 开启设置。
每轮 2000 个训练 episode、10000 个测试 episode、20 轮及学习率策略不变。

保留 TinyVIRAT 的数据/轨迹路径、21 类及顺序、12/9 类划分和 105 条独立文本描述。
空间读出使用 `QUERY_MASK_MODE: all`，测试仍为 `DROP_EMPTY_QUERY_ROWS: True`，SAV 仍为 False。
旧前置路由/Support 文本融合配置已被统一开关替代；RandAugment 仍由 `DATA.USE_RAND_AUGMENT` 控制。
