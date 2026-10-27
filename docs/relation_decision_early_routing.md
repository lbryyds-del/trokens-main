# 提前解耦版本：同步直接学习式质量判断

> 历史临时 smoke 工具已清理，下文测速命令仅作记录。
> 当前入口与检查方式见[统一消融说明](sav_ablation.md)。

> 本文记录历史质量头实验，仍由 `sav_label_pre_temporal*.yaml` 保留。
> 2026-10-23 起，当前 `sav.yaml` 使用[双轴文本拼接第 1 版](text_concat_st.md)，
> 不启用质量/独特性头；本文的默认入口及性能描述不适用于新版本。

## 修改边界

基于当前 `restore/2514d8c` / `ad32ef3`，只将已启用的质量判断从
`semantic_relation` 替换为前一实验分支 `experiment/quality-relation-decision-9.1`
（`017be3f`）中的 `relation_decision`。两套质量实现与该来源在 AST 层面完全相同，
不是重新设计另一种质量网络。

- 旧判断：Raw 语义锚点 + 有界 Support 关系修正，再进行尺度/偏置校准。
- 新判断：保留相同 Raw/text/Support 证据读取，经局部关系编码、空间池化和决策网络，
  直接输出质量 logit，再用 sigmoid 得到 `m`。
- 余弦相似度仍用于读取先验和输入特征，但不再作为必须加到最终 logit 的固定锚点。
- 新头维度沿用前一实验：隐藏 64、局部关系 32、决策隐藏 16。

不修改 `pointformer.py`、`label_conditioned_routing.py`、`attention.py` 或
`query_matching.py`。保留提前的直接余弦路由、时间后空间建模、Post Support 最终读取、
metric 独特性、`g=m*rho`、BiMHM 前门控、现有五项损失及其权重、mask、
AMP 和分块/checkpoint 设置。

质量接口兼容提前路径的两种执行方式：逐类/逐视频计算，以及
`BATCH_RAW_QUALITY=True` 时按类别组预计算 Raw 质量。只使用 Support 标签，
不向前向计算引入 Query 真值；原型内容仍由原路由形成，不由新质量头重写。

## 配置和启动

公共质量选项直接写在 `configs/trokens/sav.yaml`：

```yaml
FEW_SHOT:
  QUERY_CLASS_MATCHABILITY:
    LEARNED_VERIFIER:
      QUALITY_MODE: relation_decision
      QUALITY_HIDDEN_DIM: 64
      QUALITY_DECISION_LOCAL_DIM: 32
      QUALITY_DECISION_READOUT_DIM: 16
```

现有五个 `sav_label_pre_temporal*.yaml` 均通过继承使用新质量头。
逐项比较解析后的配置，除了质量模式和输出目录，没有改变原有配置值；
新增维度与来源实验一致。TinyVIRAT 等未显式选择新模式的配置继续使用旧 v3。

重要：`sav.yaml` 本身仍是公共基础配置，**单独启动它不会开启提前解耦**。
当前直接余弦提前解耦版本应明确使用以下入口（GPU 选择需按运行时空闲情况调整）：

```bash
cd /data3/duanwei/trokens-main
/home/duanwei/miniconda3/envs/trokens/bin/python tools/run_local.py \
  --cfg configs/trokens/sav_label_pre_temporal_cosine.yaml --gpus 0,1
```

默认输出目录：
`output/sav_label_pre_temporal_cosine_relation_decision_st_bias_cls`。
其他提前解耦配置也各自使用含 `relation_decision` 的新目录，避免覆盖旧实验。
没有修改启动器默认 GPU 或默认配置；不带 `--cfg` 的旧命令仍会选择 `sav.yaml`。

## 权重边界

提前路由版本继续为 v12，新质量核验器为 v5；这是两个独立的架构标识。
旧质量 v3 权重不能直接作为新质量 v5 的恢复训练或测试权重。
前一后置解耦实验的完整 checkpoint 也不能直接当作提前解耦 checkpoint。
保留检查并拒绝不兼容加载，不能静默随机初始化缺失头后继续评估。

本次同步的是实现，不是把旧实验训练出的质量权重直接装入新框架。
当前配置 `AUTO_RESUME=False`、`CHECKPOINT_FILE_PATH=""`、`NEW_TRAIN=True`，
用于新一轮训练对比；没有启动正式训练，也没有修改现有 checkpoint。

## 验证范围

- 质量专项与 AMP 回归首次运行：104 passed。
- 最终全量回归：`CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python -m pytest -q tests`，
  **496 passed，162.81 秒**；`git diff --check` 通过。
- 新增提前路径组合测试：固定其他权重时，Query/Support 原型、基础 logits、
  独特性输出、全局分类输出不变；核对 Raw 质量与后置路径相同。
- 覆盖 Query 标签隔离、质量/独特性四组合、串行/批量等价、外层/选择性 checkpoint、
  有限梯度、双卡 AMP/DDP 同步、同架构保存恢复及跨架构拒绝。
- 启动命令 dry-run 通过；这不是训练启动。
- GPU 0/1 实际尺寸合成 smoke 通过：每卡 35 视频、5 类、8 帧、256 轨迹、1024 维，
  FP16、梯度累积 2、连续 2 次成功参数更新；没有跳过更新、没有非有限梯度，
  两卡训练参数完全一致。运行输出确认 `routing_version=12`、`routing_scoring=cosine`、
  `quality_mode=relation_decision`、`verifier_version=5`。
  包含真实时空/运动/核验/损失模块，冻结骨干输入使用合成特征；不是实数据精度测试。

合成 smoke 命令（只在空闲卡运行，不保存权重）：

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 \
  /home/duanwei/miniconda3/envs/trokens/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 tools/smoke_label_pre_temporal.py \
  --cfg configs/trokens/sav_label_pre_temporal_cosine.yaml --steps 2 --grad-accum 2
```

新增质量诊断使用 `quality_semantic_evidence` 和 `quality_relation_feature_norm`，
不再伪造新头不存在的 anchor/correction 分解；训练指标和逐 Query/类别 CSV 均支持新字段。
这些测试验证实现与训练通路，不代表提前解耦组合的 mAP 已经得到验证。
