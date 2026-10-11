# SAV：学习型质量 / 独特性核验器

## 当前版本：v3 语义度量约束的小型关系质量头

SAV 当前默认是 `QUALITY_MODE: semantic_relation` + `UNIQUENESS_MODE: metric`，
输出目录为 `output/sav_semantic_relation_v3`。只替换质量 m，保留 v2 独特性、
主路由、匹配器和损失。完整修改方案、公式、checkpoint 兼容边界与验证记录见
[semantic_relation_quality_v3.md](semantic_relation_quality_v3.md)。

以下保留 v1/v2 历史设计与实验记录；其中的“当前/本版”均指当时的版本。
复现 v2 必须显式设 `QUALITY_MODE: semantic_attention`，不能直接沿用当前 SAV 默认质量头。

## 历史版本 v1 / v2

v2 时 `configs/trokens/sav.yaml` 默认使用
`FEW_SHOT.QUERY_CLASS_MATCHABILITY.BACKEND: learned_dual_verifier`，
以及 `LEARNED_VERIFIER.UNIQUENESS_MODE: metric`（v2）。
**当前实验只改独特性 rho，质量 m 沿用 v1**。新路径不调用旧 Top-K、Support
阈值、手工分数校准链、语义分差回退或旧全局诊断；也不做新旧分数混合或蒸馏。
v2 保留直接余弦作为关系证据，学习聚合温度与校准尺度；不是恢复整套手工判断。
旧代码仅供 `BACKEND: legacy` 和未迁移的数据集使用，TinyVIRAT 保持不变。

## 数据流与保留的核心

- DinoTxt、Pointformer、Text+Support 候选路由、每类每帧视觉原型不变。
- 质量 `m`：位置编码 / 运动融合之前的 Raw DinoTxt 与原类别文本，经共享
  投影及语义 MLP；学习型空间池化构造每个 Support 的类别条件 Raw 帧参考。
  Query 分别读取正 / 负 Raw 参考，残差融合后池化、sigmoid 输出 `[Q,K,T]`。
- 独特性 `rho`（v2）：归一化的 Post Query patch 与目标类条件下的每个
  Support 帧参考直接算余弦，按已知 Support 标签划分正 / 负集合；两侧用共享的
  可学习温度做计数归一化 log-mean-exp，再沿**现有** `pi` 池化，正值可学习
  尺度校准差值。不再有 Post 降维投影、QKV 或关系 MLP；详见末尾 v2 公式。
  `UNIQUENESS_MODE: relation_attention` 保留 v1 的注意力 + 关系 MLP 实现及权重格式。
- 保留 `target=m*rho`、`confuser=m*(1-rho)`、`null=1-m`，总和为 1。
  仅 target 质量进入原来的、时间 max 之前的双向 BiMHM 抑制；保留
  one-sided 规则，拒绝不能提高原本为负的相似度。
- 核验器只输出门控，不把文本、注意力参考或核验后的特征写回视觉原型。

这版的质量聚合与原始粘贴草案有一个明确细化：使用独立的**学习型 Raw
空间池化**，不复用 Post 候选 `pi`，也不保留 Raw Top-K。正负质量参考采用
共享注意力分别读取、按固定正负槽位融合，而非混合参考的类型嵌入。
独特性使用共享正负关系评分，交换正负参考时 `rho` 变为 `1-rho`（允许浮点舍入误差）。
这些结构有助于保留分工，但不保证 `m/rho` 已是校准后的真实概率，也不保证
两条分支不会学习到冗余信息。

## 训练和数值边界

质量头规模：128 维、4 heads、1 层参考注意力、FFN 256；D=1024 时 v2 共
460,803 个核验器参数，其中独特性只有 2 个共享标量（v1 合计 822,657）。
当前 SAV 按 Query 16 个一组分块，关闭 activation
checkpoint，开启核验器 AMP（跟随外层 CUDA autocast）。Linear、MLP、attention
矩阵乘使用 FP16；特征/LN 归一化、masked softmax / log-mean-exp、加权汇聚、关系差值、门控及
MIL 保留 FP32。CPU 与关闭外层 AMP 时仍用 FP32。

`ACTIVATION_CHECKPOINT` 仅控制中间激活的保存/重计算：关闭后保留反向所需的中间
结果，不再反向重算，用显存换速度。**不影响梯度计算、权重更新或模型文件保存**；
SAV 的 `TRAIN.CHECKPOINT_PERIOD: 1` 未修改，仍每个 epoch 保存训练 checkpoint。

默认 Q2S 目标：

```text
L_q2s = 0.60 * BCE(base_logits, video_labels)
      + 0.40 * BCE(verified_logits, video_labels)
      + 0.05 * MIL(quality_frame_logits, video_labels)
      + 0.05 * MIL(relation_frame_logits, video_labels)
```

MIL 的视频 logit 为有效帧的 `logsumexp(logits)-log(frame_count)`，不用
Top-K 或额外温度。它不是把正视频的每一帧都标为正。原有主分类损失、
可选 Support 文本对齐及外层 `Q2S_LOSS_LAMBDA` 没有改动。

`DETACH_INPUTS: True` 在核验器编码/度量之前截断上游 Raw / Post / 文本 / `pi`
梯度；质量编码/注意力/输出头与 rho 的两个标量正常学习。verified loss 仍可
沿原相似度 `C` 更新主干，不等于冻结 Pointformer。新模块矩阵参数 WD=0.01，
bias/LayerNorm WD=0，使用主学习率；主干原 WD=0.5 不变。

门控在训练前 5% 的计划 optimizer updates 从 0 线性升至 1，考虑梯度累积，
不在每个 epoch 重置；两个 MIL 从首步参与。验证 / 测试始终为完整门控。
独立开关 `QUALITY_ENABLE` / `UNIQUENESS_ENABLE` 决定是否构建对应网络和损失；
关闭的因子为 1，两者都关闭时原匹配 logits 严格不变。

有效性与监督：

- Query 标签不进入核验器，只用于视频级损失；正负参考仅依据 Support 标签。
- 核验证据使用 `point_mask & visibility`，不改变 SAV 主路由的 all-True mask。
- 当前 SAV 标签为完整 0/1。核验器额外接受 `episode_label_known_mask` 或
  非 0..1 / 非有限标签作为未知；未知 Support 不能充当负参考。
  Q2S/MIL 也忽略未知视频标签。这不代表整个数据集/主分类流程已经迁移为部分标注训练。
- 缺少有效正参考或负参考时，对应门控回到 1，该类核验 MIL 不监督。
- 无有效观测的 Query 帧：质量开启时 `m=0`，`rho=1`，该帧不进入 MIL。
  只有这些结构性掩码回退，不再根据旧分数高低决定回退。

## 如何跑实验

沿用原来的启动器，加载 `configs/trokens/sav.yaml` 即启用双学习分支。
当前默认输出为 **`output/sav_metric_rho_v2`**，与历史 `output/sav` 分开。
重复实验仍应设置新的 `OUTPUT_DIR`，避免日志/checkpoint 混放。
例如，两张空闲 GPU、每卡累积两步仍为 4 episodes/update：

```bash
CUDA_VISIBLE_DEVICES=0,1 WANDB_MODE=offline \
  /home/duanwei/miniconda3/envs/trokens/bin/python tools/run_net.py \
  --cfg configs/trokens/sav.yaml \
  NUM_GPUS 2 SOLVER.GRAD_ACCUM_STEPS 2 \
  FEW_SHOT.QUERY_CLASS_MATCHABILITY.LEARNED_VERIFIER.QUALITY_MODE semantic_attention \
  OUTPUT_DIR output/sav_metric_rho_v2
```

这是启动示例，交付时没有执行正式训练。运行前自行确认 GPU 可用，并为重复实验
更换输出路径。其他 episode/epoch/数据路径设置沿用现有 SAV 配置。

- 从头训练：当前默认 `TRAIN.AUTO_RESUME=False`、`TRAIN.CHECKPOINT_FILE_PATH=""`。
- 用旧模型初始化主干：显式指定 `TRAIN.CHECKPOINT_FILE_PATH`。旧权重如果完整
  缺少新核验器，允许初始化新头，同时强制重置 epoch/optimizer/scaler，不当成续训。
- v1 学习型权重不能直接加载成 v2（包括显式初始化）；会明确提示版本不兼容。
  此版推荐保持空 checkpoint 路径重新训练，没有实现 v1→v2 部分迁移。
- 续训：使用同一模式下含完整头的 checkpoint；输出目录自动恢复不会静默接受旧模型。
- 测试 / `VAL_ONLY`：必须有完整的新头 checkpoint；仅旧权重或部分/形状不匹配
  的新头会报错，避免把随机核验器当作已训练模型测试。
- 复现旧算法：显式设 `BACKEND legacy` 并搭配旧 checkpoint。当前 YAML 中旧
  判断字段只服务该后端。底部 transport strength / unmatched cost / one-sided /
  dual-logit loss 字段仍是两个后端共用的匹配与损失配置。
- 评估/续训之前的学习型 v1：设 `LEARNED_VERIFIER.QUALITY_MODE semantic_attention`
  和 `LEARNED_VERIFIER.UNIQUENESS_MODE relation_attention`
  并显式指定对应的 v1 checkpoint；不要把 v2 输出目录里的权重混用。

主要监控项：`q2s_quality_mil_loss`、`q2s_relation_mil_loss`、
`q2s_verifier_transport_strength`；以及正/负 Query 分开的 `verifier_quality`、
`verifier_rho`、`verifier_target_mass/confuser_mass/null_mass`、参考可用率。
v2 另记录 `verifier_metric_temperature`、`verifier_metric_scale`，便于确认聚合/校准
是否实际学习。预测 CSV 保留质量、两头 logit 和三类质量字段；不伪造旧阈值 / margin 字段。

## 验证记录与限制

以下两阶段为历史 v1 的实现及测速记录，不代表 v2 精度。v2 的验证记录见文末。

专项测试涵盖八种判断开关/时间路由组合、禁止旧评分函数、Query 标签独立、
Support 排列不变性、正负交换对称性、主路由保持、真实 forward 的 Raw 提取位置、
detach 与梯度、缺参考/无效帧/未知标签、MIL、checkpoint/优化器、CPU/CUDA AMP，
以及两进程 Gloo DDP + activation checkpoint 的多次反向。
首版全量测试 **148 passed**；`git diff --check` 和改动 Python 文件的语法编译通过。

```bash
CUDA_VISIBLE_DEVICES=0 /home/duanwei/miniconda3/envs/trokens/bin/python \
  -m pytest -q tests
```

2026-10-10 在 A100 40GB / PyTorch 2.0.1 上的**合成特征路径** FP16 外层
forward+backward 检查：Q=30、K=5、T=8、N=256、D=1024，1-shot 峰值已分配
显存约 2677 MiB，5-shot 约 4592 MiB；新模块所有参数梯度有限且非零。
这包含合成输入、路由/匹配及核验器，**不包含 DinoTxt / Pointformer 全模型和
Adam 状态**，不是完整训练显存估计，也不是性能加速或精度提升的证据。

去手工化不等于没有分数，也不保证更快；现在分数由特征关系网络学习。
效果需在匹配 seeds/episodes 的新训练下对比 Base/Novel/HM，并检查双头是否
饱和或冗余。本次没有正式训练或测量识别准确率。

## 第一阶段：Attention 等价加速版（2026-10-10）

该阶段优化仅发生在 `query_evidence_verifier.py` 内，SAV 配置未改动：Support 的
`K/V` 每个 forward 只投影一次，独特性 Query 的 `Q` 只投影一次，再向类别及
正负槽位广播；类别、帧、正负读取都是 batch 维，softmax 仍只在各自的 Support
reference 维归一化，没有跨类别交互或正负混合。Query attention 也不再把
`chunk*T*N` 压成一条长序列，减少 FP32 attention backward 的长轴归约。

该阶段保留 FP32、hidden=128、heads=4、chunk=4、activation checkpoint=True、
`m/rho/g`、MIL、detach、mask 及全部主路由/BiMHM 逻辑；参数名称、形状、注册顺序
和 architecture_version 不变，已训练的学习型核验器 checkpoint 可继续加载。
浮点归约顺序会变化，不承诺整次训练逐 bit 相同。

空闲 A100 40GB、PyTorch 2.0.1+cu117、外层 FP16、TF32=False，改动前后交替计时，
没有 optimizer.step：

| 范围 | 改前 | 改后 | 减少 |
|---|---:|---:|---:|
| 1-shot 合成路由+核验+Q2S | 863.1 ms | 273.0 ms | 68.4% |
| 5-shot 合成路由+核验+Q2S | 995.9 ms | 418.5 ms | 58.0% |
| 1-shot 真实 SAV 整模型+损失 | 1568.4 ms | 968.1 ms | 38.3% |

真实 episode forward 为 723.3→636.9 ms，backward 为 842.6→330.9 ms；峰值已分配
显存为 16404→16668 MiB。该计时不含数据读取、H2D、DDP、GradScaler 和优化器更新，
因此完整两卡步速仍需恢复后实测。真实 episode 输出/门控/损失最大误差 `4.77e-7`，
核验器梯度最大误差 `2.24e-8`。

新增等价性与逐参数梯度测试，覆盖开关、detach、checkpoint、无效观测、缺负参考、
CPU/CUDA AMP、投影复用及 DDP；该阶段全量 **187 passed**。冻结的旧学习型实现仅用于
`tests/reference_query_verifier.py` 测试和测速，不会由生产训练调用。

可复测：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  /home/duanwei/miniconda3/envs/trokens/bin/python tools/benchmark_query_verifier.py \
  --mode episode --steps 8 --variants sequential batched_fp32
```

当前运行中的 Python 进程不会热加载新代码；新启动/恢复训练才会使用加速实现。

## 第二阶段：chunk=16、关闭激活重计算、选择性 AMP（2026-10-10）

当前 SAV 已同步启用：

```yaml
FEW_SHOT:
  QUERY_CLASS_MATCHABILITY:
    LEARNED_VERIFIER:
      QUERY_CHUNK_SIZE: 16
      ACTIVATION_CHECKPOINT: False
      AMP_ENABLE: True
```

该开关继承训练已有的 CUDA AMP 与动态 GradScaler；不转换参数存储类型，不改变
优化器、损失权重、detach、参考集合、mask/fallback、正负独立 softmax 或 `g=m*rho`。
LayerNorm 子类仍保留原参数名称/形状/顺序与版本号，学习型核验器 checkpoint 兼容。
全局默认仍为 chunk=4/checkpoint=True/AMP=False，其他数据集配置不被自动切换。

AMP attention 显式拆成低精度 QK/AV 矩阵乘与 FP32 masked softmax，空参考仍增加
dummy key 并将该槽输出归零。不能只凭取消外层 AMP 禁用，就假定 attention 内部
softmax 一定保留 FP32；实现对照了当前使用的
[PyTorch 2.0.1 math attention 源码](https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/native/transformers/attention.cpp#L769)。

同一 A100 40GB、相同权重/输入/dropout、外层 FP16、TF32=False、2 次预热后交替
计时 8 次，报告前后向中位数。真实 episode 使用预训练 DinoTxt 与同一组随机初始化
Pointformer/核验器，**不是已训练 checkpoint 的准确率评估**：

| 实现 | 1-shot 合成路由前后向 | 1-shot 真实整模型前后向 | 整模型峰值已分配显存 |
|---|---:|---:|---:|
| 初版顺序学习型、FP32、chunk=4、checkpoint 开 | 833.2 ms | 1546.6 ms | 16409 MiB |
| 批处理 attention、FP32、chunk=4、checkpoint 开 | 268.1 ms | 972.8 ms | 16673 MiB |
| 再设 chunk=16、checkpoint 关，仍 FP32 | 203.6 ms | 908.9 ms | 25813 MiB |
| 再开启选择性 AMP（当前 SAV） | 124.8 ms | 829.2 ms | 23141 MiB |

当前版相对初版真实整模型耗时减少 **46.4%**、吞吐 **1.87x**；相对上一阶段再减少
**14.8%**。峰值显存约 **22.6 GiB**，比初版增加约 **6.6 GiB**。该数据不含数据
读取/H2D、DDP、优化器状态/更新及 GradScaler 的 unscale/update；完整两卡训练仍需
新启动/恢复后实测。不要把单卡已分配显存等同于完整 DDP 训练的显存占用。

5-shot 仅测合成路由（6 次、loss_scale=65536）：947.8→265.7 ms，峰值
4602→16549 MiB；**尚未测完整 5-shot 训练显存**，不能沿用 1-shot 的余量结论。

真实 episode 相对初版 FP32 的输出/门控/损失最大绝对误差为 `5.54e-4`，核验器
参数梯度最大绝对误差 `1.03e-4`，所有已有模型梯度有限。混合精度有舍入误差，
不再是严格的 FP32 数值等价；不承诺整次训练 bitwise 一致或识别准确率不变。

新增检查覆盖完整/尾部 chunk、FP16 计算与 FP32 归一化/输出边界、AMP 关闭回退、
detach 开关、checkpoint 开关、缺参考、全无效观测、非有限输入清理，以及
GradScaler(scale=65536) 的核验器参数/输入梯度检查。保留第一阶段的严格 FP32
逐参数等价测试。完整路由另在 scale=1024 下检查门控与有限梯度。全量测试
**205 passed**（17.37s）；`git diff --check` 与改动 Python 文件语法编译通过。

测速脚本默认固定 `loss_scale=1024`，只 backward、不更新模型。直接不缩放做 FP16
反向会造成小梯度下溢，不适合拿来代替 AMP 训练检查；真实整模型测试在 65536 下，
**未改动的初版 Pointformer 分支也会溢出**，因此统一用 1024 做匹配比较。这不修改
正式训练已有的动态 GradScaler 策略，也不是强制正式训练采用固定 scale。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  /home/duanwei/miniconda3/envs/trokens/bin/python tools/benchmark_query_verifier.py \
  --mode episode --steps 8 --loss-scale 1024
```

如显存不足，先恢复 `ACTIVATION_CHECKPOINT: True`；只减小 chunk 未必能充分降低
训练保存的总激活量。要回到上一阶段的纯 FP32 验证实现，同时设置 `AMP_ENABLE: False`、
`QUERY_CHUNK_SIZE: 4`、`ACTIVATION_CHECKPOINT: True`。以上均不影响每 epoch 保存模型。

## 第三阶段：metric rho v2，单独改独特性

这一步是**算法修改**，不是 v1 的等价加速：根据同 checkpoint 消融定位的独特性
问题，去掉它的自由特征关系网络，直接保留原 Post 空间里的匹配证据。
质量 m 的结构、初始化与计算完全不变，主路由、参考构造、损失权重、detach、
transport warmup、one-sided BiMHM 及 AMP/chunk/checkpoint 设置不变。
重新训练后，共享主干和质量参数的轨迹仍可能因 rho 改变而变化，不承诺权重始终一样。

对每个 Query patch i 和同一目标类的有效 Support 帧参考 j：

```text
s_ij       = cosine(PostQuery_i, PostSupportReference_j)
tau        = softplus(temperature_raw) + 1e-4     # 初始 0.10，可学习
a          = softplus(scale_raw) + 1e-4           # 初始 10.0，可学习
r_i^sign   = tau * [logsumexp_j(s_ij / tau) - log(|R_sign|)]
z_sign     = a * sum_i pi_i * r_i^sign
rho        = sigmoid(z_positive - z_negative)
g          = m * rho
```

这里采用 log-mean-exp 平滑聚合，而不是额外的 MLP 或固定 Top-K。
它对每项有效相似度单调递增，梯度权重为 `softmax(s/tau)`；网络学习应该偏向
最匹配参考还是更均匀地使用参考。减去 `log(|R_sign|)`，使相同证据不会仅因一侧
参考数量更多而多出 `tau*log(count)` 偏置。正负使用同一 tau/a，不设 additive bias，
交换正负得到相反 logit。参数是跨类别共享的两个标量，不是按 Base 类记忆的分类器。

正负集合各自展平有效的 Support×帧，沿用 v1 的参考集合和现有 pi，不新增
视频级二次聚合。不同 Support 若有效帧数不同，仍按其有效参考帧计权；这一步未引入
Support 等权或时间建模。缺一侧参考时 rho=1 且不算对应 MIL；无效 Query 帧 rho=1。
直接余弦 GEMM 跟随 AMP，归一化、log-mean-exp、标量参数变换、pi 汇聚和门控均 FP32。

SAV 在 v2 实验中使用以下设置；当前默认版本已见文首：

```yaml
OUTPUT_DIR: output/sav_metric_rho_v2
FEW_SHOT:
  QUERY_CLASS_MATCHABILITY:
    BACKEND: learned_dual_verifier
    LEARNED_VERIFIER:
      QUALITY_MODE: semantic_attention
      UNIQUENESS_MODE: metric
      METRIC_TEMPERATURE_INIT: 0.10
      METRIC_SCALE_INIT: 10.0
      DETACH_INPUTS: True
      QUERY_CHUNK_SIZE: 16
      ACTIVATION_CHECKPOINT: False
      AMP_ENABLE: True
```

这些初始化值不是训练中固定不变的阈值。两个标量进入优化器（WD=0），同时接收
原 relation MIL 和 verified loss；新日志记录实际温度/尺度，单测验证一次更新确实改变它们。
本版没有残差 MLP、类别竞争、Query 梯度解锁、质量头锚定或新损失超参数。

### 启动及 checkpoint 边界

按前文命令启动新训练即可，当前配置保持 `AUTO_RESUME=False`、checkpoint 路径为空。
历史文件未删除/改写；新输出目录只有正式启动后才会创建。若启动器覆盖 OUTPUT_DIR，
请也使用独立的新路径，并保持原 GPU 数、有效 episodes/update、seed、shot 和测试协议。

新版 rho 使用 architecture_version=2，旧学习型为 1；没有实现二者之间的部分权重迁移。
不能直接用 epoch20 的 v1 权重冒充 v2 测试或自动续训。要评估原学习型实验，覆盖：

```text
FEW_SHOT.QUERY_CLASS_MATCHABILITY.LEARNED_VERIFIER.QUALITY_MODE semantic_attention
FEW_SHOT.QUERY_CLASS_MATCHABILITY.LEARNED_VERIFIER.UNIQUENESS_MODE relation_attention
TEST.CHECKPOINT_FILE_PATH <原v1权重路径>
```

要回到原手工实验，使用 `BACKEND legacy` 及对应手工 checkpoint。仅改配置不会
把已训练 v1/v2/手工 checkpoint 变成另一种模型。

### 本轮验证

全量测试 **236 passed（20.16s）**；`git diff --check` 与修改文件语法编译通过。
全量测试覆盖度量公式、单调性、参考计数归一化、Support/类别排列、正负交换、无跨类
耦合、无 Query 标签输入、缺参考/未知标签/无效帧、可学习标量更新、完整/尾 chunk、
detach、activation checkpoint、AMP/FP32 梯度和两进程 DDP；保留 v1 对冻结顺序
实现的逐参数等价测试。v2 与 v1 共用权重时，质量输出及原路由张量一致。

A100 40GB / PyTorch 2.0.1，chunk=16、checkpoint=False、AMP=True、TF32=False，
两次预热后交替测量八次，固定 GradScaler scale=1024，只 forward/backward：

| 范围 | 原优化后 v1 | metric v2 | 耗时减少 |
|---|---:|---:|---:|
| 真实 SAV 1-shot 整模型 + 损失 | 823.1 ms | 786.6 ms | 4.4% |
| 合成 5-shot 路由 + 核验 + Q2S | 268.1 ms | 190.0 ms | 29.1% |

真实整模型峰值已分配显存 23136→19516 MiB（约 22.6→19.1 GiB），合成 5-shot
16544→11048 MiB。真实测速仍使用同一缓存 episode、预训练 DinoTxt 和随机初始化
Pointformer/质量头，不是识别准确率评估。新旧 rho/gate/loss/梯度本来就不同；测速脚本
只对比不变的质量输出和主路由（本次最大误差为 0），另检查所有已有梯度有限。
计时不含数据加载/H2D、DDP、优化器状态及 step；5-shot 结果不能当完整训练显存。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  /home/duanwei/miniconda3/envs/trokens/bin/python tools/benchmark_query_verifier.py \
  --mode episode --variants batched_amp metric_amp --steps 8
```

尚未正式训练；本次只证明实现、边界、梯度和计算开销，没有证明 Novel/HM 提升。
训练完成后按匹配 checkpoint 选择规则/seed/episodes 比较手工、学习型 v1 和 v2 的
Base/Novel/HM，再检查正负 rho/g 分布、温度/尺度。先判断这次单独改 rho 是否有效，
再决定是否动质量 m；不要同时变更 detach 或损失权重混淆归因。
