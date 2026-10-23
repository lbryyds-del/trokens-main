# SAV 前置标签条件建模：时间＋空间注意力偏置

> 2026-10-23：以下为保留的前置实验记录，仅对应 `sav_label_pre_temporal*.yaml`。
> 当前默认 `sav.yaml` 已改为[双轴文本拼接第 1 版](text_concat_st.md)，关闭 m/ρ。
> 下文关于 sav 默认入口的描述及验证数字属于当时版本。

## 当前：主匹配复用前置权重（v29–v32，2026-10-22）

前置路由一次返回 `g` 和原始 patch softmax 概率 `a`。`g` 仍生成时间/空间注意力偏置；
`a` 直接复用于同一视频、同一候选类、同一帧的 Post patch 聚合：`F = sum_n(a_n * Post_n)`。
使用的是 softmax 概率，不是归一化后的门控 `g`，没有第二次主匹配相似度/softmax，
没有 detach 复用权重；空 mask 的概率及聚合特征均为零，聚合内部保持 FP32。

- Support：原文本生成的前置权重聚合 Post，已知正例均值得到主匹配原型。
- Query：Raw 正 Support＋文本生成的前置权重聚合 Post；不再用 Post 条件重新读主匹配特征。
- rho：保留独立的 Post 正 Support 匹配原型＋文本条件、逐 Support 正负帧参考、
  Post Query 核验权重和原度量公式。原型随主聚合变化，因此 rho 数值不保证不变。
- m、前置条件/偏置、时空编码、全局分类、BiMHM 和损失定义均未修改；非前置基线不受影响。

移除 `POST_SUPPORT_READOUT` 选择开关。原 post_readout YAML 文件名保留为 fast 的输出目录别名，
不再代表另一种主读出结构。输出目录新增 `reuse_pre_weights`。
投影版本为 v29/v30（无/有分类），cosine 为 v31/v32；常用 cosine 配置是 v32。
旧 v1–v12、v21–v28 不能直接续训/评估，同 scorer 仅可显式重置初始化；跨 scorer 仍拒绝。
没有启动训练。下文的旧权重来源、版本号、测试数量及性能记录仅适用于对应历史实现。

本版验证：CPU 全量 `441 passed, 74 skipped`；覆盖一次前置打分、直接加权聚合、
路由梯度、mask、类别/视频分块、checkpoint 和两进程 CPU DDP。
12 组固定权重对照确认编码特征及 m/全局分类输出保持一致，4 组 CPU BF16 前向/反向有限。
rho 度量函数与原版 AST 一致，独立 Post 核验 oracle 通过；dry-run 与 diff 检查通过。
未验证真实数据精度或 GPU 显存/吞吐。

### 默认配置已集中到 sav.yaml

`configs/trokens/sav.yaml` 显式写入全部前置参数：ENABLE=True、SCORING=cosine、
温度 .04、门控强度 .25、时间/空间偏置各 1、5 类 × 8 视频分块、Raw m 批处理、
MLP/读出 checkpoint 开启，外层及 verifier checkpoint 关闭。全局分类权重仍为 1。
Support/Query 主 Post 聚合固定复用前置权重；ρ 保留独立 Post+文本核验，不再添加选择开关。
`sav_label_pre_temporal_cosine.yaml` 是直接继承 sav 的兼容别名；原投影对照配置显式覆盖
scorer 和 checkpoint/批处理选项，保持原有有效配置。m、ρ、数据和优化参数未改。

新输出目录为 `output/sav_label_pre_temporal_cosine_relation_decision_reuse_pre_weights_post_rho_st_bias_cls`。
默认启动器使用 sav；可先只检查命令（不会训练）：

```bash
/home/duanwei/miniconda3/envs/trokens/bin/python tools/run_local.py --dry-run
```

配置修改只对新启动生效，已有训练进程不会自动切换。本次未启动或重启训练。
集中后再次验证：sav 的完整有效配置与集中前 cosine 配置完全一致，五个已有前置配置
的有效值均未变化；CPU 全量 `443 passed, 74 skipped`，默认启动 dry-run 与 diff 检查通过。

## 历史：恢复原版 Post rho 证据链（v21–v28，2026-10-22）

恢复 `a5134b8` 的 rho 逻辑，不回退整个编码器或质量头：

1. Support Post 由原文本加权读出；已知正 Support 平均得到匹配原型 `P_post`。
2. `P_post` 按原融合函数与文本生成 `L_post`，保留有效帧规则和 `VISUAL_DETACH`。
3. 同一个 `L_post` 读取全部 Support 的逐帧参考（正负均保留），并生成 Query Post patch 权重。
4. Query Post patch 与正/负 Support 帧参考分别算余弦，集合内做数量归一化的
   log-mean-exp，再按 Query 权重汇总正负差、缩放并 sigmoid，得到逐帧 rho。
   rho 的度量函数、缺失参考回退、`m*rho`、主匹配与五项损失均未修改。

Support 编码仍用纯文本；Query 编码仍用 Raw 正 Support＋文本，不新增编码分支。
Support Post 仅缓存当前类别块，视频分块仍约束每次编码；外层 checkpoint 包住两次读出，
不把完整 Support Post 缓存作为其输出。没有额外的一遍 Support 编码。

`POST_SUPPORT_READOUT` 仍只选择主匹配 Query 的读出条件：True 时复用 Post 权重给 rho；
False 时保留 Raw 条件主匹配，但 rho 单独使用 Post 条件权重。rho 两端始终对齐。
`sav.yaml` 后置基线本就使用这条证据链，本次不改默认架构；五个既有前置配置的输出目录
加入 `post_rho`，避免与原实验混淆，未新增开关/配置文件，也未启动训练。

版本为投影路由 v21–v24、直接余弦 v25–v28（原版本号加 16）；cosine 配置为 v28。
旧 v1–v12 不可直接续训/评估；同一 scorer 只允许显式 `CHECKPOINT_EPOCH_RESET=True`
初始化并重置 epoch/optimizer/scaler。v13–v20 属于其他实验，拒绝加载。
跨 scorer 仍禁止迁移；同新版完整 checkpoint 可正常续训。可控对照应从头训练。

本次 CPU 回归：`480 passed, 74 skipped`（隐藏 CUDA）；含两进程 CPU DDP 检查。
另以相同权重对照修改前代码的 12 组条件，编码次数/特征、主匹配原型/权重/分数、m 与
全局分类输出及这些独立分支的梯度均保持一致，rho 按预期变化。
rho 两个度量函数与 `a5134b8` 的 AST 一致；新增独立计算检查覆盖共享 Post 条件、
逐 Support 参考、Query 权重、正负汇总及无效帧回退。启动器 dry-run 和 diff 检查通过。
这些只验证实现与边界，不代表真实数据精度、GPU 显存或收敛结果。

## 以下为历史 v5–v12 记录

下文的版本号、Raw rho 参考、旧权重加载方式、性能与测试数字均描述历史实现，
不代表上述当前版本；历史启动命令若使用，必须另选包含 `post_rho` 的新输出目录。

原对照配置为 **v6**：Support 纯文本、Query 正 Support Raw＋文本，
在原 `Temporal → Spatial → MLP → Norm` 串行块内修改注意力分数。
旧的时间 Value 乘法门控实现已替换，不保留旧模式开关，不新增并行编码分支。
原 `sav.yaml` 基线仍保留；以下三个前置配置全部使用同一新版模型。
旧 SVG/PNG 流程图已移出仓库，临时备份位于 `/tmp/trokens-value-gate-diagrams.RCkqsg/`。
已有训练输出和 checkpoint 未删除。

新增独立的 [前置直接余弦权重对照（v12）](label_cosine_pre_temporal.md)：
保留 v8 的前置时间/空间偏置与 Post Support 最终读出，只移除路由 Wq/Wk 投影。
使用 `sav_label_pre_temporal_cosine.yaml`；原有配置仍使用可学习投影，不受影响。

## 本次单项对照：Post Support 条件读出（v8）

新配置 `configs/trokens/sav_label_pre_temporal_post_readout.yaml` 继承 fast，
只开启 `FEW_SHOT.LABEL_PRE_TEMPORAL.POST_SUPPORT_READOUT=True`，并使用新的输出目录。
旧 fast 配置仍默认 False，便于继续评估已训练 v6，旧的 Value 乘法门控没有恢复。

```text
正 Support Raw + 原文本 → L_raw → Query 时间 / 空间注意力偏置
Support 原文本 e_c      → Support 时间 / 空间注意力偏置
                                  │
                     原串行 Temporal → Spatial → MLP → Norm
                                  │
                    ┌─────────────┴──────────────┐
              Support Post                  Query Post
                    │                           │
               原文本读出                       │
                    │                           │
             正 Support 平均 → P_post,c          │
                    │              │            │
                    │        与原文本融合        │
                    │              ↓            │
                    │           L_post ──→ Query 最终加权读出
                    │                           │
                    └──────────── 匹配 ─────────┘
                              原匹配 + m/rho 核验
```

- 前置 Query 条件仍然是 `L_raw`；只把最终 Query 读出的条件换成 `L_post`。
  `P_post,c` 是原有纯文本读出、已知正 Support 平均得到的 `[K,T,D]` 匹配原型，
  不是 Raw 摘要，也不是 rho 的逐 Support 参考特征。Query 真值不参与两种条件。
- 复用原 `_fuse_episode_text_with_support_visual`：有效帧均值、视觉/文本各自归一化、
  按原权重融合后归一化；无有效正 Support 时退回文本。
  保留 `VISUAL_DETACH=True`，仅切断融合条件的视觉梯度，**不 detach 匹配原型**。
- 复用已算出的 Support Post，不增加 Transformer、投影参数或全局分支。
  Support 编码/匹配读出、Query 编码、256 条轨迹、全局分类和五项损失均不改。
- m 的 Raw 路径不改；rho 的原有正/负 Support 参考仍按 `L_raw` 读出，公式和损失不改。
  rho 使用的 Query 假设权重跟随新的最终 Query 读出，因此 **rho 数值并非被锁定不变**。
- 编码前的条件不依赖 Post。执行上原本每个类别块就先算 Support、再算 Query，
  直接在这两步之间融合已有原型，不新增一遍 Support 编码。
- 这是检验“Raw 条件直接读 Post Query 是否损害匹配”的消融，尚不能保证 Novel 提升。

双卡从头训练（与上一轮保持 1-shot、seed=1、2 卡 × 累积 2；不要续训 v6）：

```bash
cd /data3/duanwei/trokens-main
/home/duanwei/miniconda3/envs/trokens/bin/python tools/run_local.py \
  --cfg configs/trokens/sav_label_pre_temporal_post_readout.yaml \
  --gpus 0,1 \
  --output-dir /data3/duanwei/trokens-main/output/sav_label_pre_temporal_post_readout_st_bias_cls_2gpu \
  -- SOLVER.GRAD_ACCUM_STEPS 2 TRAIN.AUTO_RESUME False
```

请先确认 0/1 空闲；若此新目录已有正式结果，另换新目录。
启动配置应包含 `POST_SUPPORT_READOUT: True`、`CLASS_LOSS_LAMBDA: 1.0`、
`K_SHOT: 1`、`GRAD_ACCUM_STEPS: 2`、`AUTO_RESUME: False`、空 `CHECKPOINT_FILE_PATH`。
不要把加载 v6 后微调的结果当成同初始条件对照。

## 数据流与边界

以下为 v6 对照；v8 仅替换上述最终 Query 读出条件。

```text
Support / Query 视频 + 点轨迹
             │
         冻结 DinoTxt
             │
     ┌───────┴──────────────────────────────┐
     │                                      │
Raw 轨迹特征 R                         主编码输入 X
无本模型位置/运动信息             外观 + 位置 + HoD + CrossMotion
     │                                      │
     ├─ 原文本加权正 Support Raw              │
     │         ↓                            │
     │    Raw 类别摘要 + 文本                 │
     │         ↓                            │
     │    Query 条件 L_c                     │
     │                                      │
     ├─ Support：文本 e_c → g_S,c             │
     └─ Query：条件 L_c → g_Q,c               │
                    │                       │
                    │         共享时间 QKV / Softmax 前基础分数
                    │                       │
                    ├→ 时间 Key 偏置 → 各类别时间 Softmax → 原始 V
                    │                       │
                    │              时间投影 + 残差
                    │                       │
                    │       从时间输出计算空间 QKV（参数共享）
                    │                       │
                    └→ 空间 Key 偏置 → 各类别空间 Softmax → 原始 V
                                            │
                                    空间残差 → MLP → Norm
                                            │
                           各视频、各候选类的 Post 特征
                                            │
               ┌────────────────────────────┼───────────────────┐
               │                            │                   │
       Support 纯文本读出           Query 融合条件读出      类别条件全局特征
               │                            │                   │
        正 Support 求平均            Query 类别帧特征       类别等权平均
               │                            │                   │
        匹配原型 P_c ────────────────────────┘              原全局分类头
                              │                                 │
                    原始匹配 + m/rho 核验后匹配                   │
                              └─────────── 五项损失 ─────────────┘
```

- 每类保留全部 256 条轨迹，不把 Label→Patch 权重直接 AV 池化为单个主编码 token。
- Query 条件只使用候选类别文本与已知正 Support 的可见 Raw 证据；无有效正例时退回文本。
  Query 真值仅用于前向后的监督与报告，绝不参与条件或分支选择。
- Raw 摘要只生成 Query 条件，不是最终匹配原型；匹配原型仍由正 Support 的 Post 特征产生。
- Support 主编码和匹配读出均为纯文本；rho 的逐 Support 参考读出仍使用融合条件。
- m 保持原始文本＋Raw 证据，rho 保持同候选类的 Post Query 与正/负 Support 比较。
  `m*rho`、BiMHM、warmup、单侧约束、采样器和已见类别筛选均不变。
- 全局分类复用各类别条件 CLS（或原配置指定的 patch 均值），按类别等权平均后经过
  `pre_logits → head_drop → head`；没有额外无条件编码。分类监督覆盖 Support 和 Query。

## 时间和空间的具体修改

保留原 Label Query / Raw Key 投影及 patch 维 Softmax，初始投影为单位矩阵：

```text
alpha[c,t,n] = masked_softmax_n(cos(Wq(condition_c), Wk(R[t,n])) / temperature)
g[c,t,n] = 1 + gamma * tanh(N_valid * alpha[c,t,n] - 1)
```

默认 temperature=0.04、gamma=0.25。这是正值先验 g，不是直接使用 softmax 概率，
也没有换成 sigmoid 路由器。均匀权重对应 g=1；无效位置给中性先验，原 mask 仍然生效。

时间（同轨迹，沿被关注时间 s 归一化）：

```text
A_T[c,m,t,s] = softmax_s(Q_T[m,t] K_T[m,s]^T / sqrt(d_h)
                       + lambda_T * log(g[c,s,m] + eps))
Y_T = A_T @ V_T
```

空间（同一帧，沿被关注 patch n 归一化）：

```text
A_S[c,t,m,n] = softmax_n(Q_S[c,t,m] K_S[c,t,n]^T / sqrt(d_h)
                       + lambda_S * log(g[c,t,n] + eps))
Y_S = A_S @ V_S
```

时间、空间使用同一个逐帧逐轨迹先验 g，但分别作用于不同 Key 轴，跨 head 和 query 位置广播。
两处 Value 都不再乘 g。空间 QKV 来源于类别条件时间输出，因此**共享参数但不共享计算结果**。
时间 QKV／基础分数只计算一次，最终时间注意力矩阵必须按类别分别 Softmax。

CLS key 先验为 1，CLS query 同样按标签重新关注 patch，原拼接、残差和逐帧平均不变。
实现使用 `log((g+1e-6)/(1+1e-6))`，等价于对全部 key 减去相同常数，
不改变公式的 Softmax，并保证 g=1 精确对应零偏置。log 用 FP32 计算后转换为分数 dtype，
不将完整空间矩阵提升为 FP32。attention dropout 位于逐类 Softmax 后。

固定配置 `FEW_SHOT.LABEL_PRE_TEMPORAL`：

```yaml
TEMPORAL_BIAS_SCALE: 1.0
SPATIAL_BIAS_SCALE: 1.0
GATE_STRENGTH: 0.25
```

两个 SCALE 分别控制 lambda_T、lambda_S，均不是新增可学习参数。置零可单独消融一处偏置；
两者都为零或 GATE_STRENGTH=0 时恢复原串行编码计算，但下游类别条件读出仍在，
因此不等于把整个模型恢复为旧基线。非零先验下空间 patch 对 CLS 的相对权重也会变化；
不能把“所有 patch 的同一非零偏置”误称为含中性 CLS 时的完全中性操作。

## 损失保持不变

```text
L = L_global + 0.60 L_base + 0.40 L_verified + 0.05 L_m + 0.05 L_rho
```

`cls_loss` 为全局分类；`q2s_loss` 包含后四项。两个外层权重都为 1。
`CLASS_LOSS_LAMBDA=0` 仍是显式的无全局分类消融，不是默认设置。
损失定义不变不表示数值必然相同，因为时空建模后的特征已经变化。

## 配置、显存与启动

- `sav_label_pre_temporal.yaml`：1 类 × 4 视频，外层 checkpoint。
- `sav_label_pre_temporal_parallel.yaml`：5 类 × 8 视频，Raw m 批量化，保留外层 checkpoint。
- `sav_label_pre_temporal_fast.yaml`：5 类 × 8 视频，关闭外层／核验器 checkpoint，
  仅保留 MLP 与文本读出 checkpoint。不是时间与空间并行分支。

三者模型、参数、条件来源和损失相同，只改变执行分块与重计算策略。
类别折叠进 batch，不跨类别 attention，不长期保存完整 `[B,K,T,N,D]`。
当前仍限定 depth=1、divided_space_time、冻结 DinoTxt、点轨迹输入、无 adaptive pooling。
新机制的吞吐和显存应以当前实测为准，不能引用旧 Value 门控版本的性能结果。

双卡新训练示例（仅提供命令，等待指定卡空闲后手动执行）：

```bash
cd /data3/duanwei/trokens-main
/home/duanwei/miniconda3/envs/trokens/bin/python tools/run_local.py \
  --cfg configs/trokens/sav_label_pre_temporal_fast.yaml \
  --gpus 0,1 \
  --output-dir /data3/duanwei/trokens-main/output/sav_label_pre_temporal_fast_st_bias_cls_2gpu \
  -- SOLVER.GRAD_ACCUM_STEPS 2
```

此例每次更新为 2 卡 × 每卡累积 2 = 4 episodes，不按普通 batch size 缩放。
新目录包含 `_st_bias_cls`，不移动或覆盖已有训练目录，运行中的旧进程不会自动变成新模型。

## Checkpoint 保护

Raw 读出对照含全局分类为 v6、无分类为 v5；Post 读出含分类为 v8、无分类为 v7。
跨这些语义版本的权重（包括旧 v1–v4）只允许显式初始化：同时指定 `TRAIN.CHECKPOINT_FILE_PATH` 和
`TRAIN.CHECKPOINT_EPOCH_RESET True`，重置 epoch/optimizer/scaler，保留目标版本标记。
不允许把旧语义当新版 resume/eval；部分缺失权重仍拒绝加载。
同版本完整 checkpoint 可恢复 optimizer/scaler/epoch，三个吞吐配置间可切换。
保留版本识别仅为防误载，不表示旧模型实现仍可选择运行。

## v8 本次验证（2026-10-18）

- 完整回归（可见 GPU 0/1）：`406 passed in 116.72s`，无跳过。
  覆盖独立 Post 原型/融合/Query 读出 oracle、Raw 编码不变、Support 参考/m/全局输出不变、
  Query 真值隔离、已知正例限制、空 mask/回退、VISUAL_DETACH 与匹配梯度、
  不完整类别块/分块/checkpoint 等价、FP16/BF16、CPU/CUDA 双进程累积和参数同步。
  新配置也通过原五项损失的独立数值与梯度 oracle，以及 v6→v8 显式初始化/v8 续训保护。
- 真实 SAV＋本地冻结 DinoTxt 单卡短测（GPU 2）：`[35,5,8,256,1024]`、fast、FP16、累积 2，
  4 次尝试得到 2 次成功更新。前两次 cross-motion bias/router Key 梯度溢出，
  GradScaler 安全跳过、scale 65536→32768→16384，随后两次梯度有限、路由和分类头均更新。
  峰值 allocated 31.86 GiB / reserved 36.02 GiB，无 OOM。
- 真实双卡短测（GPU 0/1），每卡同样完整尺寸、FP16、累积 2：4 次尝试得到 2 次成功更新。
  第 1/3 次 AMP 安全跳过、第 2/4 次梯度有限且路由/分类头均更新；scale 最终 16384。
  每次尝试后全部可训练参数跨卡严格相等；每卡峰值 allocated 31.98 GiB / reserved 36.13 GiB，
  无 OOM。没有调整初始 GradScaler scale，也不把跳过的尝试计作成功更新。
- 未启动正式训练、未保存训练 checkpoint；短测不证明 Novel 提升或长程稳定性。

复现本次新模式（请先确认所选卡空闲）：

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 \
  /home/duanwei/miniconda3/envs/trokens/bin/python -m pytest -q tests

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 \
  /home/duanwei/miniconda3/envs/trokens/bin/python tools/smoke_label_pre_temporal.py \
  --cfg configs/trokens/sav_label_pre_temporal_post_readout.yaml \
  --steps 2 --grad-accum 2 --real-data

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 \
  /home/duanwei/miniconda3/envs/trokens/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 tools/smoke_label_pre_temporal.py \
  --cfg configs/trokens/sav_label_pre_temporal_post_readout.yaml \
  --steps 2 --grad-accum 2 --real-data
```

## v6 历史验证

- 新增独立公式 oracle：时间／空间 Key 轴、原始 V、串行数据流、CLS 与空 mask，
  输出及梯度对照；两处偏置分别能影响结果并训练路由投影。
- 中性先验／双零 lambda 恢复原编码；Query 真值隔离、条件来源、类别与视频重排、
  分块／checkpoint 等价、五项损失及 checkpoint 初始化／恢复保护均覆盖。
- 完整回归：`341 passed, 7 skipped in 61.26s`。包含 CPU 双进程 DDP 梯度累积与参数同步，
  单卡 CUDA FP16/BF16。该次只暴露空闲 GPU 2，7 项双 CUDA GPU 测试跳过。
- 随后确认物理 GPU 0/1 已空闲，补测全部 7 项 CUDA-DDP 与更新后的配置检查：
  `8 passed, 157 deselected in 44.80s`。验证 FP16、外层／选择性 checkpoint、
  首步 warmup、梯度累积、分类头更新与跨卡参数严格相等；累计覆盖 348 个独立测试项。
- 完整尺寸合成 AMP 短测：物理 GPU 2，`[35,5,8,256,1024]`、fast、FP16、累积 2，
  两次尝试均成功更新；分类头和路由投影均更新，无非有限梯度、无 AMP 跳步。
  峰值 allocated 28.40 GiB、reserved 32.52 GiB；不包含真实冻结骨干。
- 同卡完整尺寸真实 SAV＋本地 DinoTxt、相同 fast/FP16/累积 2：4 次尝试得到 2 次成功更新。
  前两次 `label_pre_temporal.key.weight` 梯度溢出，GradScaler 安全跳过更新，
  scale 从 65536 → 32768 → 16384；之后连续两次梯度有限、路由与分类头均成功更新。
  峰值 allocated 31.86 GiB、reserved 36.02 GiB，无 OOM。没有通过修改初始 scale 掩盖跳步；
  该结果不保证长程稳定性，完整尺寸真实双卡训练尚未验证。

可复现检查（请按实际空闲情况选择 GPU）：

```bash
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 \
  /home/duanwei/miniconda3/envs/trokens/bin/python -m pytest -q tests

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 \
  /home/duanwei/miniconda3/envs/trokens/bin/python tools/smoke_label_pre_temporal.py \
  --cfg configs/trokens/sav_label_pre_temporal_fast.yaml --steps 2 --grad-accum 2
```

smoke 使用完整实际 Pointformer、HoD/CrossMotion、核验器、分类头和优化器。
默认替换冻结视觉／文本塔为合成输入；增加 `--real-data` 才包含真实 SAV 与本地 DinoTxt。
不下载、不保存 checkpoint、不跑正式 epoch；短测不能证明长程稳定性或精度提升。
