# SAV 第 1 版：双轴单文本拼接，关闭 m / ρ

本文以下为 v1 历史记录。2026-10-24 起，v1 对照入口固定为
`configs/trokens/sav_text_concat_s1_base.yaml`（有效配置与切换前的 sav 完全相同），
原 fast/S32 配置仍关闭 m/ρ。当前 `sav.yaml` 已切换到
[v2：Raw m / 同空间权重 Post ρ](text_concat_verifier.md)，使用独立输出目录。
下文提到的“当前 sav”及测速结果均指 v1，不应套用到 v2。

当前入口为 `configs/trokens/sav.yaml`。旧实验仍使用
`sav_label_pre_temporal*.yaml`，不会随当前入口切换到新结构。
2026-10-23 将空间文本副本从 32 改为 1；原 32 份配置保留在
`sav_text_concat_s32.yaml`，不覆盖原实验输出，也不停止已有训练进程。

## 每个视频、每个候选类别的计算

Support、Query 完全使用同一条编码路径和原始类别文本（保留原有 DinoTxt
提示词编码及平均方式）。不使用 Query 真值标签选择文本；每个 Query 都计算
episode 的全部候选类别。也不再用 Support 特征改写 Query 文本。

```text
视觉特征 + 原有两路运动特征 [T=8,N=256,D]
  → 每条轨迹：8 个视觉 token + 1 个类别文本 token
  → 时间 self-attention；只保留 8 个视觉输出；原有 temporal_fc / 残差
  → 每帧：256 个视觉 token + 原有 CLS + 1 个原始类别文本
  → 空间 self-attention；保留视觉及原有 CLS 输出
  → 原有残差、MLP、最终 Norm
  → 用本次空间 attention 的文本→视觉权重，池化最终视觉输出
  → 每个视频、每个类别得到 [8,D]
```

时间轴更新后的文本输出丢弃，空间轴重新拼原始文本。现在每个轴均为单个文本；
不额外添加 MA-FSAR 的全局时间 token、Adapter、门控或交叉注意力。

## 空间如何聚合

设空间 attention 已计算的 logits 为 `L[B*K*T,H,258,258]`。
取最后 1 个文本 query 行，以及 256 个视觉 key 列：

1. 对每个 head、每个文本行，只在有效视觉 key 上做 softmax。
2. 对 head 取平均（文本行只有一行），得到 `w[B,K,T,256]`。
3. 使用最终 Norm 后的视觉 patch `Z`，计算 `frame = Σ_n w_n * Z_n`。

这等价于取 **attention dropout 之前**的文本→视觉概率子矩阵，在视觉列内重新
归一化。实现直接复用已经算好的 QK logits，不新增 QKV、cosine 相似度、门控
或交叉注意力模块。权重和最终视觉值都参与反向传播。CLS/text 不参与池化求和；
无有效视觉 key 时权重及帧特征为零。空间有效性继续使用原有 query mask；
当前 `QUERY_MASK_MODE: all` 保持不变，未改为 visibility mask。

历史 32 份相同文本的 query 行在同一 head 内相同；固定相同的参数和空间输入时，
换成一份不改变这个视觉归一化读出权重，但会改变 self-attention 的视觉更新，
因而最终帧特征及训练轨迹会改变。单文本是新的消融实验，不是数值等价优化。

## 匹配及损失

- 正 Support 按类别平均得到 `[K,8,D]`；未知/负 Support 不参与该类别平均。
- Query 保留 `[Q,K,8,D]`，继续使用原 BiMHM，`logit = 10 * similarity - 2`。
- `QUALITY_ENABLE=False`、`UNIQUENESS_ENABLE=False`：不构建 m/ρ 判断头，
  不计算两个 MIL 损失，不做 mρ transport。最终匹配与基础匹配相同。
- 保留原全局分类头；每个视频的候选类别分支等权平均后分类。
- 当前显式使用 `BASE_ONLY_LOSS_ENABLE=True`、`DUAL_LOGIT_LOSS_ENABLE=False`，
  base/final 系数为 `1/0`；只计算一次 base BCE，不再计算重复的 final BCE。
  m/ρ 两个 MIL 系数、transport 强度及 quality relation 强度均为 0。
  总损失为 `global_BCE + base_BCE`，与原来 base==final 时的 `.6/.4` 数学目标相同。
  日志 `q2s_verified_loss=0` 表示该损失关闭，不表示模型预测已无误差。
- 数据、backbone、mask、运动模块、学习率及 epoch 配置不变。

`QUERY_CLASS_MATCHABILITY.ENABLE` 保留 True 以继续使用已有的未知标签损失屏蔽；
两个判断开关均为 False，不表示 m/ρ 仍然启用。

## 实现及运行边界

- `trokens/models/attention.py`：双轴拼接、复用空间 QK 读出。
- `trokens/models/text_concat_conditioning.py`：候选类别/视频分块、Post 聚合、
  Support 原型和 Query 匹配。默认每块 2 类 × 4 视频，完整块 activation checkpoint。
- `trokens/models/pointformer.py`：新路径分发；旧前置路由和 Support/text 融合关闭。
- `trokens/utils/checkpoint.py`：保存并核对结构签名 `[1,1,1,1]`；拒绝旧路由或
  不同文本数量的 checkpoint 混用。应从头训练，不能直接续训旧实验。
- `tests/test_text_concat_conditioning.py`：独立公式、mask、角色对称性、标签隔离、
  分块/梯度、checkpoint、完整 forward 分发及 CPU DDP 测试。

启动命令（选择空闲 GPU；不加 `--dry-run` 才会正式运行）：

```bash
/home/duanwei/miniconda3/envs/trokens/bin/python /data3/duanwei/trokens-main/tools/run_local.py --cfg /data3/duanwei/trokens-main/configs/trokens/sav.yaml --gpus 0,1 --dry-run
```

输出目录为 `output/sav_text_concat_t1_s1_attention_readout_m_rho_off`。
上面的 GPU 编号只是命令示例，必须确认空闲后才正式运行；本次修改不启动正式训练。
有限步检查可复用 `tools/smoke_label_pre_temporal.py --cfg configs/trokens/sav.yaml`；
默认合成输入，`--real-data` 使用本地数据和缓存 backbone，不下载或保存 checkpoint。
这些检查只验证可执行性及数值链路，不代表验证集准确率或实验收益。

## 保持逻辑的提速：冻结文本缓存

当前启用 `FEW_SHOT.TEXT_CONCAT_ST.CACHE_FROZEN_TEXT=True`：

- 仅缓存冻结、eval 的 DinoTxt text tower 输出（仍取原来的后半特征）。
- 在缓存之后照常执行 `text_to_model_proj`、逐提示词归一化、组内平均及再次归一化；
  不缓存可训练投影结果，不截断其梯度，不改提示词和候选类别顺序。
- 每个 rank 按需编码未见过的提示词组。命中后不再 tokenize 或运行 text tower。
- 编码器权重/缓冲区版本、设备、dtype、autocast 或 tokenizer 改变会失效重算；
  提示词字符串变化产生新的缓存项。可训练 tower 自动绕过缓存。
- 缓存不写入 checkpoint，旧前置路由不启用；首次填充和精度切换仍需编码。
- 最初采用 `CLASS_CHUNK_SIZE=2`、`VIDEO_CHUNK_SIZE=4` 和完整块 activation checkpoint，
  不改变 episode、梯度累积、优化器、损失和原型构造。

减少文本到 1 份使空间序列从 289 缩至 258；QK 矩阵元素数减少约 20.3%，
不等于端到端训练必然提速 20.3%。缓存消除了重复文本编码，但实际收益需实测。
不同编码 batch 大小在混合精度下可能有浮点舍入差异，不承诺逐位相同的训练轨迹。

临时测速代码及其专用参数已按用户要求移除；冻结文本缓存、原有有限步检查工具
和回归测试保留。分块只改变执行并行度，不改变模型目标，但可能改变随机数分配/
浮点归约顺序，且显存上升。
之前物理 GPU 3 的完整块 checkpoint 测量中，5×4 与 5×8 速度接近，前者峰值
allocated 约 12.09 GiB，后者约 15.36 GiB；这是历史完整块重算结果。
当前 `sav.yaml` 及独立输出入口 `configs/trokens/sav_text_concat_s1_fast.yaml` 均采用 5×8、
完整块 checkpoint 关闭、仅 MLP 和最终 Post Norm/聚合 checkpoint 开启；
注意力前向结果保留，不重算整个时空编码块。2×4 完整块重算为此前测速对照。
该执行方式参考 `612e3045b2009f66c97c3cd512f959be106c8617` 的 fast 配置，
没有迁回旧前置路由，也没有共享不同候选文本的完整时间注意力结果。
详见[原 GPU 3 阶段记录](text_concat_speed_gpu3.md)及[当前双卡提速验证](text_concat_speed_2gpu.md)。
旧 early-routing 的提速数字不能套用到当前 concat 结构。
不建议未经测试关闭全部 activation checkpoint、减少轨迹/帧数或删除损失。

启动加速版（先确认 GPU 0、1 空闲；这是正式启动命令，本次不自动执行）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/duanwei/miniconda3/envs/trokens/bin/python /data3/duanwei/trokens-main/tools/run_local.py --cfg /data3/duanwei/trokens-main/configs/trokens/sav_text_concat_s1_fast.yaml --gpus 0,1
```

每卡仍累计 2 个 episode，双卡每次更新共 4 个 episode。帧数、轨迹数、
候选类别、mask、增强、Support 原型、BiMHM、优化器和学习率均不改。
分块改变矩阵执行形状，允许混合精度舍入和随机数分配的差别，不能承诺
训练轨迹逐位相同；固定分块的 checkpoint 开关已经验证输出、梯度和 RNG 一致性。

## 单文本版本本次验证

- GPU 3 环境完整回归：537 passed、13 skipped；包含缓存值/投影梯度、单文本独立公式、
  空 mask、候选类/角色对称性、checkpoint 边界及 CPU DDP 检查。
- 真实 SAV 有限步测试完成原 32 文本、单文本、缓存、5×4/5×8 分块对照；
  成功组均无非有限梯度、无 AMP 跳步。未保存 checkpoint、未启动正式训练。
- 关闭完整块 checkpoint 的探测触发了约 31.52 GiB allocator 安全上限，未设为默认。
  该上限只用于当时的测速进程，现已移除；不代表 40GB 显卡的物理上限。
- 计时条件、峰值显存、被排除的共卡结果见 [GPU 3 报告](text_concat_speed_gpu3.md)。

## 原 32 份版本的历史有限步验证（2026-10-23，非本次单文本实测）

- 完整回归 `pytest -q tests`：534 passed；`git diff --check` 和启动器 dry-run 通过。
- 完整尺寸 `B=35,K=5,T=8,N=256,D=1024`，FP16，梯度累积 2。
- 单卡合成数据：2 次成功优化器更新，峰值 allocated 约 6.80 GiB。
- 双卡合成数据：每卡 2 次成功更新、参数同步一致，峰值 allocated 约 6.92 GiB/卡。
- 单卡真实 SAV + 本地冻结 DinoTxt：1 次成功更新（2 个 microbatch），峰值
  allocated 约 10.04 GiB；真实检查保留原数据加载和文本编码，未下载权重。
- 上述检查无非有限梯度、无 AMP 跳步；主编码器、运动分支、分类头梯度检查通过。
- 五个历史前置 YAML 的有效配置与修改前逐项一致（忽略新增且关闭的 TEXT_CONCAT_ST 配置）。

检查未保存训练 checkpoint，未启动完整 epoch 或正式实验；显存数字仅对应上述有限步。
