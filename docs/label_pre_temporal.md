# SAV 前置标签条件建模：时间＋空间注意力偏置

原对照配置为 **v6**：Support 纯文本、Query 正 Support Raw＋文本，
在原 `Temporal → Spatial → MLP → Norm` 串行块内修改注意力分数。
旧的时间 Value 乘法门控实现已替换，不保留旧模式开关，不新增并行编码分支。
原 `sav.yaml` 基线仍保留；以下三个前置配置全部使用同一新版模型。
旧 SVG/PNG 流程图已移出仓库，临时备份位于 `/tmp/trokens-value-gate-diagrams.RCkqsg/`。
已有训练输出和 checkpoint 未删除。

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
