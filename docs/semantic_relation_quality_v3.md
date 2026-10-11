# SAV v3：语义度量约束的 Support 关系质量头

## 修改范围

只替换质量 m 的内部估计。独特性沿用 v2 的 Post cosine / metric rho，
DinoTxt、Pointformer、Text+Support 主候选路由、每类每帧视觉原型、BiMHM、
`target=m*rho`、`confuser=m*(1-rho)`、`null=1-m` 全部保留。
TinyVIRAT 的 legacy 后端和全局默认不变。

这不是与 v2 数值等价的实现优化，也不是把原手工算法的四个常数换成参数。
新增网络学习逐 patch 的语义池化修正、Query 条件的参考读取、正负关系解释；
原始余弦证据始终有直接通道，关系修正受固定幅度约束。

## 具体数据流

### 1. 原始语义锚点

输入为位置编码、运动融合、Pointformer 之前的 Raw DinoTxt，以及未经过
Support 融合的原类别文本。先在 FP32 中归一化：

`s[k,q,t,n] = cosine(raw_query[q,t,n], text[k])`

新的 64 维投影 P 只用于辅助关系，不用于替换这项余弦计算。

### 2. 有界的质量专用空间池化

共享小 MLP 输入 `[P(x)-P(e), P(x)*P(e), s]`，输出一个空间 logit 修正：

`d_pool = 0.5 * tanh(pool_mlp(...))`

`w = masked_softmax(s / 0.07 + d_pool, over=patches)`

同一个池化器用于 Query 和 Support。w 仅服务质量核验，不覆盖主路由 pi。
每个 Support、每个类别、每帧保留一个由原始 Raw 特征加权得到并归一化的参考 r。
不做 Top-K，也不提前把一类所有 Support 帧合并成一个均值。

### 3. Query 条件的正负参考读取

每个 Query patch 分别读取该类的正、负 Support 帧集合。有效身份仅来自已知
Support 标签；未知标签不是负例。读取 logits：

`cosine(x,r) / 0.10 + 0.5 * tanh(dot(P(x)-P(e), P(r)-P(e)) / sqrt(64))`

两侧分别 masked softmax。读取 P(r) 向量，同时保留加权的原始 Query–reference
余弦与 reference–text 余弦。注意力不是最终质量；全体参考都弱时，归一化权重
仍然存在，因此后面必须保留绝对语义证据。

实现上先读取各侧关系参考，再对逐 patch 的关系向量做 MLP，不为每个
Query–reference 对展开一个完整高维 MLP，避免 `[K,Q,T,N,R,D]` 的激活开销。
没有把 Query patch 提前压成帧向量，也没有新增时间模块或跨类别竞争。

### 4. 有界关系判断与最终门控

关系 MLP 输入：

- `P(x)-P(e)`；
- `P(x)-read_pos`、`P(x)-read_neg`；
- `P(x)*read_pos`、`P(x)*read_neg`；
- 原始 s、正/负读取的 Query–reference cosine、正/负读取的 reference–text cosine。

MLP 为 `325 -> 128 -> 1`（输入 FP32 LayerNorm，中间 GELU）。输出逐 patch 的
`r = tanh(MLP(...))`，而不是直接替换原始证据的任意分类 logit：

```text
E[q,k,t]     = sum_n w[q,k,t,n] * s[q,k,t,n]
Delta[q,k,t] = relation_strength * 0.10 * sum_n w[q,k,t,n] * r[q,k,t,n]
logit_m     = a * (E + Delta) + b
m           = sigmoid(logit_m)
a           = softplus(scale_raw) + 1e-4
```

固定界保证 `abs(Delta) <= 0.10`（在原始余弦单位中），不是说概率 m 最多变化 0.10。
a/b 共享、可学习，初始 10/0；所有网络参数跨类别共享，不提供类别 ID。
池化与关系输出层以小非零权重初始化，使修正较小且首步低层网络也能收到梯度。
两个温度与三个修正上限是明确的结构超参数，保存为 buffer，不被优化器更新。
恢复 checkpoint 时使用权重中保存的这些数值；不能靠改 YAML 偷换已训练模型的界。

这套结构限制了语义偏移，但不能保证全局单调、双头语义可识别或 Novel 提升。
网络仍可能弱化 Support 或使修正饱和，必须用机制实验检查。

## 训练、数值与异常边界

保持原 Q2S 目标：`0.60 L_base + 0.40 L_verified + 0.05 L_quality_MIL + 0.05 L_rho_MIL`。
原主分类/文本对齐损失不变。没有手工分数蒸馏或新的监督标签，正视频不被当作
每帧/每个 patch 都正；Query 标签只进入视频级损失，不进入核验器前向。

`DETACH_INPUTS=True` 保持核验器输入梯度隔离，网络内部与 Support 读取正常学习。
verified loss 仍沿主相似度更新主模型，不能把这一配置称作冻结整个主干。
保持 5% transport warmup、chunk=16、activation checkpoint=False、AMP=True。

原始 Raw 余弦、参考原特征聚合、归一化、masked softmax、最终聚合与门控使用 FP32；
小投影、MLP、低维关系矩阵乘跟随外层 AMP。Raw 高维余弦通过单次 GEMM 比较
共享 Query 与全部类别参考，避免广播复制 K 份高维 Query。

缺少有效正/负参考时：有效 Query 帧 m=1，不参与质量 MIL。
无有效 Query patch 时：m=0，不参与 MIL。全无效参考 masked softmax 有显式安全分支。
未恢复旧“Support 均值间隔 <= 0 就令 m=1”的规则；当前 v2 也没有该规则。

## 配置、兼容与运行

SAV 已默认设置：

```yaml
OUTPUT_DIR: output/sav_semantic_relation_v3
FEW_SHOT:
  QUERY_CLASS_MATCHABILITY:
    LEARNED_VERIFIER:
      QUALITY_MODE: semantic_relation
      QUALITY_HIDDEN_DIM: 64
      QUALITY_PATCH_TEMPERATURE: 0.07
      QUALITY_REFERENCE_TEMPERATURE: 0.10
      QUALITY_MAX_POOL_DELTA: 0.5
      QUALITY_MAX_REFERENCE_DELTA: 0.5
      QUALITY_MAX_EVIDENCE_DELTA: 0.10
      QUALITY_SCALE_INIT: 10.0
      QUALITY_BIAS_INIT: 0.0
      QUALITY_RELATION_STRENGTH: 1.0
      UNIQUENESS_MODE: metric
```

`HIDDEN_DIM/NUM_HEADS/FFN_DIM` 是旧质量或旧 rho 注意力设置；新质量头使用独立的
QUALITY_HIDDEN_DIM，不使用旧 128 维四头 QKV。

- v3 从头训练：现有启动方式即可；默认 AUTO_RESUME=False、checkpoint 路径为空。
  如果启动器覆盖 OUTPUT_DIR，请也使用新的独立目录，不覆盖已有实验。
- v2 对照：`QUALITY_MODE semantic_attention` + `UNIQUENESS_MODE metric`，匹配 v2 权重。
- v1 对照：`QUALITY_MODE semantic_attention` + `UNIQUENESS_MODE relation_attention`，匹配 v1 权重。
- 原手工对照：`BACKEND legacy`，匹配手工权重。
- v3 的版本号是 3；新质量配旧 attention rho 的可选组合为版本 4。质量关闭时
  沿用实际 rho 的版本。跨版本加载/续训/评估明确报错，不允许随机新头冒充已训练头。
  未实现 v2→v3 部分迁移。完整缺少核验器的 legacy 主干初始化仍按原规则重置训练状态。

## 诊断与机制验证

新增按正/负 Query 分开的训练/验证指标：

- `verifier_quality_anchor_*`：原始语义锚点 E；
- `verifier_quality_correction_*` / `correction_abs_*`：关系修正的方向与幅度；
- `verifier_quality_correction_saturation_*`：质量权重落在 `abs(tanh)>0.95` patch 的比例；
- `verifier_quality_pool_shift_*`：空间池化修正幅度；
- `verifier_quality_scale/bias/relation_strength/max_evidence_delta`：校准和控制量。

测试 CSV 同步增加 E、Delta、abs(Delta)、饱和率的每视频/每类均值。
这些是机制诊断，不是额外质量标注或真实概率校准证据。

同一训练好的 v3 checkpoint 可通过 `QUALITY_RELATION_STRENGTH 0.0` 关闭**最终关系修正**，
保留学习到的 Raw 空间池化。它不是“完全无网络”或“恢复手工算法”的对照。
QUALITY_ENABLE=False 会改变构建的头与版本，不能直接配完整 v3 checkpoint；
本轮同 checkpoint 的机制控制只提供上述 relation_strength 开关。

评估应保持 seed/episodes、shot、有效 episodes/update 与原实验一致，报告 Base/Novel/HM。
同 checkpoint 关闭修正仅说明模型是否依赖它；是否有泛化收益需要配套重新训练对照。
实际训练后还应检查替换/打乱 Support 是否使正确类别证据按预期变化，不能用未训练网络
“数值发生变化”的单测冒充学到了可迁移关系。

## 本轮验证记录（2026-10-11）

未启动正式训练，不提供新模型准确率结论。测试与测速未修改既有实验权重。
全量测试 **319 passed（34.09s）**；修改文件语法编译和 `git diff --check` 通过。

代码验证覆盖原始余弦锚点、精确最终公式、修正上限、空间掩码、原特征参考聚合、
重复参考集合不变性、Support 内容/身份干预、关闭关系修正、无 Post/pi 捷径、
逐参数 MIL 梯度及更新、诊断导出、旧评分链禁用、Query 标签隔离、主路由/rho 不变、
checkpoint 版本保护、AMP/FP32、无效/缺失参考、activation checkpoint 和两进程 DDP。

测速环境：A100 40GB、PyTorch 2.0.1+cu117、TF32=False，Q=30/S=5/K=5/T=8/N=256/D=1024。
新旧均 chunk=16、checkpoint=False、AMP=True。预热 2 次，交替测量，报告中位数。

| 测试范围 | v2 旧网络质量 + metric rho | v3 有界关系质量 + 同一 rho |
|---|---:|---:|
| 核验器参数量 | 460,803 | 116,690 |
| 合成路由＋核验＋Q2S，8 次，前向＋反向 | 91.481 ms | 87.327 ms |
| 上述峰值已分配显存 | 5793.554 MiB | 4697.762 MiB |
| 同一真实 SAV episode，6 次，整模型前向＋反向 | 768.365 ms | 767.628 ms |
| 上述峰值已分配显存 | 19510.339 MiB | 18413.762 MiB |

合成路径约快 4.5%；真实 episode 基本持平（0.1% 不应解读为实质加速），已分配
峰值约少 1097 MiB。不是完整训练吞吐：不含数据加载、H2D、DDP、优化器更新、
验证/存盘时间；缓存的同一 episode 与随机初始化新头不构成识别精度证据。
合成路径在 loss scale=65536、真实 episode 在 1024 下，全模型已产生的梯度及
全部核验器参数梯度均有限。基线路由/独特性比较的最大输出差为 0。

另在完整 Q=30、D=1024 合成路径比较 v3 FP32/AMP：输出最大绝对误差 `1.92e-5`，
核验器梯度最大绝对误差 `6.99e-6`，loss scale=65536；逐参数有限性检查通过。

补充 5-shot 合成检查（S=25，其余形状不变，预热 1 次、测量 3 次）：
v2/v3 前向＋反向分别为 192.090/167.424 ms，峰值 11047.964/9086.792 MiB；
loss scale=65536 梯度有限，主路由与 rho 输出差为 0。这仍不是 5-shot 整模型吞吐。

可复测（先确认对应 GPU 空闲）：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  /home/duanwei/miniconda3/envs/trokens/bin/python -m pytest -q tests

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  /home/duanwei/miniconda3/envs/trokens/bin/python tools/benchmark_query_verifier.py \
  --mode route --steps 8 --variants metric_amp semantic_amp --loss-scale 65536

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  /home/duanwei/miniconda3/envs/trokens/bin/python tools/benchmark_query_verifier.py \
  --mode episode --steps 6 --variants metric_amp semantic_amp --loss-scale 1024
```

测速脚本不会 optimizer.step，也不会写训练 checkpoint。新架构间只比较主路由和
未修改头的一致性；只有同一架构的 FP32/AMP 或实现版本才比较全部输出与梯度。
