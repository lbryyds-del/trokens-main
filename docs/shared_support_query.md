# Support / Query 均关闭文本解耦（m/ρ 全关闭）

本文记录 v5 双侧关闭实验；当前 `sav.yaml` 的编辑方式已统一为
[四项消融开关及自动损失联动](sav_ablation.md)。下文旧 shared 开关为内部派生值，
不再手动填写。独立历史配置仍可运行，当前默认四项也均关闭。

默认入口：`configs/trokens/sav.yaml`，输出 `output/sav`。
独立消融入口：`configs/trokens/sav_shared_support_query.yaml`，输出
`output/sav_shared_support_query_m_rho_off`；编码和损失与当前默认一致，使用旧配置格式。

## 编码、聚合与监督

- Support / Query 共享原有普通时空 Transformer，每个视频只编码一次；
  不拼文本、不编码候选类别文本、不按类别复制视觉特征，不使用文本注意力权重。
- 两侧对最终 Norm 后的每帧有效视觉 patch 求均值，输出 `[B,T,D]`。
  当前 `QUERY_MASK_MODE=all`，即每帧平均 256 个视觉 patch；CLS 不进入此聚合。
- 已知正标签 Support 类内平均形成 `[K,T,D]` 类原型。
  多标签 Support 可参与多个类原型，但不需要多次编码；未知标签不作正例。
- Query 只有一份 `[Q,T,D]` 特征，对所有 Support 类原型匹配，得到 `[Q,K]`；
  没有 K 路条件 Query 各自只与同类 Support 匹配。
- 全局分类复用同一次普通编码的视觉 CLS，不是 patch 均值或文本加权向量。
  Support / Query 都保留原有 seen 类全局分类监督，没有额外跑一遍 CLS 编码。
- m、ρ 判断头、匹配修正、两个 MIL 辅助损失全部关闭。保留未知标签 mask。
- appearance、HoD、CrossMotion、位置编码、采样、优化器和训练协议不变。
  保留每块 8 视频、MLP/readout checkpoint；完整块 checkpoint 仍关闭。

对帧间余弦矩阵 `C[q,k,t,s]`，保留当前打分和损失：

```text
S[q,k] = 0.5 * (mean_t max_s C[q,k,t,s] + mean_s max_t C[q,k,t,s])
z_final = z_base = 10 * S - 2
L = L_global + L_base
```

`L_base` 仍是 episode 内的少样本匹配 BCE，不是全局 seen 类分类损失。
不重复计算 final BCE，不引入 m/ρ 先验或修正。

## 与历史 Trokens baseline 的区别

结构上已回到“通用视觉时空编码 → patch 均值 → Support 类原型匹配”。
但这是当前框架的**双侧无解耦对照**，不是原始 Trokens 的逐项数值复现：
历史 `process_patch_tokens` 在空间均值前做 ReLU，使用其余弦分母规则，
计算双向最近帧距离的时间求和并取负。这里未额外回退这些打分细节，
保留当前归一化、余弦双向最大相似度均值及 `10*S-2`，避免同时更换训练打分。

## 配置开关

```yaml
FEW_SHOT:
  TEXT_CONCAT_ST:
    ENABLE: True
    SHARED_QUERY_ENABLE: True
    SHARED_SUPPORT_ENABLE: True
  QUERY_CLASS_MATCHABILITY:
    QUALITY_ENABLE: False
    UNIQUENESS_ENABLE: False
    ABSOLUTE_MASS_TRANSPORT_STRENGTH: 0.0
    BASE_ONLY_LOSS_ENABLE: True
    DUAL_LOGIT_LOSS_ENABLE: False
    BASE_LOGIT_LOSS_WEIGHT: 1.0
    VERIFIED_LOGIT_LOSS_WEIGHT: 0.0
    LEARNED_VERIFIER:
      SEMANTIC_MIL_WEIGHT: 0.0
      RELATION_MIL_WEIGHT: 0.0
```

`TEXT_CONCAT_ST.ENABLE=True` 保留 episode/损失接口，**不表示仍拼文本**。
两个 shared 开关绕过所有文本编码/拼接/权重聚合。原 token 数、文本 READOUT、
类别分块、冻结文本缓存值保留但不参与该分支。冻结特征提取器初始化不变。
`POT_ROUTE.QUERY_PARTIAL_ENABLE=True` 也是旧接口名，不代表多路 Query；
`QUERY_CLASS_MATCHABILITY.ENABLE=True` 保留未知标签 mask，不启用已关闭的 m/ρ。

独立历史对照不继承当前可编辑的 `sav.yaml`：

| 配置 | Support | Query | m/ρ |
| --- | --- | --- | --- |
| `sav_shared_support_query.yaml` | 通用 | 通用 | 关闭 |
| `sav_text_concat_shared_query.yaml` | 文本拼接 | 通用 | 关闭 |
| `sav_text_concat_s1_base.yaml` | 文本拼接 | 文本拼接 | 关闭 |
| `sav_text_concat_s1_m_rho_on.yaml` | 文本拼接 | 文本拼接 | 开启 |

## 启动与检查点

新结构签名 `[5,0,0,1]`（v4 保留给此前独立无文本 Query CLS 实验）。
不把旧 v1/v2/v3/v4 checkpoint 直接当作本版续训/评估。配置从头训练：
`AUTO_RESUME=False`、`NEW_TRAIN=True`、空 `CHECKPOINT_FILE_PATH`。

只检查启动命令：

```bash
/home/duanwei/miniconda3/envs/trokens/bin/python tools/run_local.py \
  --cfg configs/trokens/sav_shared_support_query.yaml --gpus 0,1 --dry-run
```

确认 GPU 可用后手动训练（独立输出目录）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/duanwei/miniconda3/envs/trokens/bin/python tools/run_local.py \
  --cfg configs/trokens/sav_shared_support_query.yaml --gpus 0,1
```

此次修改不自动启动、停止或重启正式训练。已有进程不会自动切换到新配置。
单元测试/CPU DDP 仅验证实现，不代表真实数据精度、GPU 峰值显存或提速收益。

## v5 切换时的历史验证

- 隐藏 GPU 的全套回归：`564 passed, 82 skipped`；未执行 CUDA 专用用例。
- 新分支验证普通 `pt_forward` 对照、均值/Support 原型与逐对匹配公式、
  不调用文本/解耦/mρ、无 Query 真值泄漏、每视频单次编码、全局 CLS、
  空 mask、多标签/未知标签、分块与 checkpoint 前后向、双进程 CPU DDP。
- 原有 18 份合并配置对照：除默认 `sav.yaml` 启用新增 Support shared 开关外，
  其余配置仅增加默认关闭的同名字段，无其他有效值改变。
  新独立配置与 `sav.yaml` 仅输出目录不同。
- 两个入口的双卡启动器 `--dry-run` 与 `git diff --check` 通过。
  未运行正式训练或 GPU 性能测试。
