# 前置直接余弦权重对照

> 2026-10-23：本文为历史前置实验记录。该结构请使用
> `sav_label_pre_temporal_cosine.yaml`；当前 `sav.yaml` 已切换为
> [双轴文本拼接第 1 版](text_concat_st.md)，不再是本文配置的别名。

## 当前版本 v32（2026-10-22）

Support 和 Query 主匹配直接复用各自前置路由的 patch softmax 权重聚合 Post 特征。
前置先验及时间/空间编码不变；不再为主聚合计算一次 Post 文本相似度，也不用门控 `g` 当权重。
rho 保留独立的 Post 原型＋文本核验流程；m 不改。主原型变化会传递给 rho 条件。
详见 [当前权重复用说明](label_pre_temporal.md)。

完整配置已集中到 `sav.yaml`，默认启动直接使用本方案；
`sav_label_pre_temporal_cosine.yaml` 仅作兼容别名，解析后的配置完全相同。
输出目录包含 `reuse_pre_weights_post_rho`。
`POST_SUPPORT_READOUT` 已移除，旧检查点不可直接续训或评估，公平对照应从头训练。
默认已启用前置解耦；m/ρ 的判断逻辑和训练参数不变，未启动或重启训练。

## 历史版本 v28（2026-10-22）

前置编码、主匹配与质量头保持当前实现，仅恢复 `a5134b8` 的 Post rho 证据来源：
Post 正 Support 匹配原型＋文本生成同一个 `L_post`，读取逐 Support 正负帧参考，
并生成 Query 的 patch 核验权重。正负余弦比较、平滑汇总、sigmoid 与 `m*rho` 不变。
详情和完整 checkpoint 边界见 [当前 Post rho 说明](label_pre_temporal.md)。

现有 `sav_label_pre_temporal_cosine.yaml` 输出目录已加入 `post_rho`，版本从 v12 升为 v28。
旧 v12 不可直接续训/评估；同 scorer 权重仅允许显式重置初始化。公平对照应从头训练。
未改变 `sav.yaml` 的默认后置架构，也未启动训练。

## 以下为 v12 历史记录

下文的 Raw rho 参考、版本/加载约定以及实测结果属于历史实现，不能作为当前版本的验证。
历史命令若使用，必须另选包含 `post_rho` 的新输出目录。

本轮对照基于 v8，恢复旧式“直接相似度 → patch softmax”的权重生成方式，
不把解耦挪回时空建模之后，不退回已删除的 Value 乘法门控。

## 唯一研究变量

```text
v8：a = masked_softmax_n(cos(Wq(condition), Wk(Raw)) / tau)
v12：a = masked_softmax_n(cos(condition, Raw) / tau)
```

新模式不包含可学习的路由 Q/K 矩阵，直接在 DinoTxt 原始语义空间计算相似度。
归一化沿特征 D，softmax 沿每帧 patch N，不沿类别 K；同一轨迹仍可服务多个标签。
实现采用旧式安全归一化（eps=1e-12）与 [-1,1] 余弦截断，权重运算保留 FP32。
空 mask 行输出零权重、中性先验，不引入 NaN。

没有删除 Transformer 自身的时间/空间 QKV，也没有将相似度换成未归一化点积。
这不是标准 cross-attention 的 AV 聚合：不会提前把 256 条轨迹合并为一个 token。

## 保持不变的数据流

```text
冻结 DinoTxt → Raw 轨迹 R + 主编码输入 X（含原位置/运动信息）
                      │
Support 条件 = 原文本 e_c
Query 条件   = 原文本加权正 Support Raw，再与文本融合的 L_raw,c
                      │
       条件与各 Raw 轨迹直接算余弦相似度
                      ↓
          patch softmax → 权重 a[b,c,t,n]
                      ↓
      g = 1 + 0.25 * tanh(N_valid * a - 1)
                      ↓
        时间/空间注意力分数加入 log(g) 偏置
                      ↓
         原串行 Temporal → Spatial → MLP → Norm
                      ↓
       Support Post 纯文本读出 → 正 Support 平均 → P_post,c
                                                    │
                                          与文本融合 → L_post,c
                                                    │
                           Query Post ← L_post,c 加权读出
                                      ↓
                         原主匹配 + m/rho 核验后匹配
```

- 仍在主时空编码前计算标签先验，影响时间/空间建模，不只是修改最终匹配分数。
- 共享计算时间 QKV/基础分数，空间从类别条件时间输出继续计算，时空串行不变。
- 全部 256 条轨迹保留；不增加编码分支，不修改 mask、运动模块、采样、优化策略。
- 最终 Query 继续使用 v8 的 Post Support 原型＋文本读出，不恢复 v6 的 Raw 条件最终读出。
- m 保持原 Raw 路径，rho 的 Support 参考仍按 Raw 融合条件读出，核验定义不变。
- 全局分类复用类别条件全局特征并等权平均，五项损失保持：
  `L_global + 0.60 L_base + 0.40 L_verified + 0.05 L_m + 0.05 L_rho`。
- 不使用 Query 真值生成条件。Support 标签仅筛选已知正例；无有效正例时原融合逻辑回退文本。
- 固定输入/条件下直接余弦路由不能通过优化 Wq/Wk 改变；模型仍训练时空编码、运动模块、
  分类头和核验器。episode、Raw 特征和条件变化时权重仍会改变，并非所有视频共用固定权重。

## 配置与启动

`configs/trokens/sav_label_pre_temporal_cosine.yaml` 继承 v8，仅覆盖：

```yaml
OUTPUT_DIR: output/sav_label_pre_temporal_cosine_st_bias_cls
FEW_SHOT:
  LABEL_PRE_TEMPORAL:
    SCORING: cosine
```

原配置默认 `SCORING: projected_cosine`，保持 v6/v8 行为。新配置保留两处偏置系数 1、
tau=.04、门控强度 .25、Post 读出、5 类 × 8 视频分块、MLP/读出 checkpoint。
不在这轮同时改变偏置强度或关闭时间/空间条件。

从头训练，继续使用 data3（先确认卡空闲；已有结果的目录不要重复使用）：

```bash
cd /data3/duanwei/trokens-main
/home/duanwei/miniconda3/envs/trokens/bin/python tools/run_local.py \
  --cfg configs/trokens/sav_label_pre_temporal_cosine.yaml \
  --gpus 0,1 \
  --output-dir /data3/duanwei/trokens-main/output/sav_label_pre_temporal_cosine_st_bias_cls_2gpu \
  -- SOLVER.GRAD_ACCUM_STEPS 2 TRAIN.AUTO_RESUME False
```

日志配置核对：`SCORING: cosine`、`POST_SUPPORT_READOUT: True`、两处 `BIAS_SCALE: 1.0`、
`CLASS_LOSS_LAMBDA: 1.0`、`K_SHOT: 1`、`NUM_GPUS: 2`、`GRAD_ACCUM_STEPS: 2`，
空 `TRAIN.CHECKPOINT_FILE_PATH`，启动提示 `Start from scratch.`。
2 卡 × 每卡累积 2，仍然每次更新 4 episodes；不缩放原学习率/训练 episode 数。

## Checkpoint 边界

直接余弦版本为 v9/v10（Raw 最终读出，无/有分类）、v11/v12（Post 最终读出，无/有分类）。
本次新配置为 v12；不含路由 query/key 参数，只保存 router 版本 buffer。
v5-v8 原投影版本保持原参数名与版本号；已有对照 checkpoint 仍可用原配置加载。

两种 SCORING 之间禁止误恢复、误评估，也不提供隐式删除/添加 Q/K 的权重初始化迁移。
必须独立从头训练；同版本完整 checkpoint 可正常恢复 epoch、optimizer、scaler。
同一 SCORING 内切换分类/最终读出语义仍需显式 `CHECKPOINT_EPOCH_RESET=True` 初始化。

## 已完成的验证

全量回归测试已通过：`439 passed in 142.72s`。

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 \
  /home/duanwei/miniconda3/envs/trokens/bin/python -m pytest -q tests
```

覆盖直接余弦权重与旧公式对齐、mask/空正例、梯度、两处前置偏置、五项损失、
checkpoint 版本隔离，以及 CUDA 混合精度和双进程 DDP。
启动器 dry-run 与 `git diff --check` 通过。
按用户要求，未启动正式训练，也未运行真实 SAV 数据的双卡 smoke；上述通过结果不代表精度提升。

## 检验目标

本实验判断：原始语义空间相似度的前置权重，是否比可学习投影后的前置权重更利于 Novel。
不是测试“前置 vs 后置”本身，也不预先宣称直接相似度一定更好。
优先比较核验前 Novel、最终 Base/Novel/HM，以及正负样本读出权重与 rho；保持同训练协议。
