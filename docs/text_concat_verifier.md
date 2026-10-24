# 双轴单文本拼接 v2：Raw m + 同空间权重 Post ρ

入口 `configs/trokens/sav.yaml`；当前配置输出 `output/savo`（以启动时加载的配置为准）。
这是新的判断/损失实验，不是与关闭版数值等价的提速。未自动启动正式训练。

## 保持不变

- Support / Query 都按全部候选类别独立编码，参数共享；Query 真值不参与前向。
- 时间每条轨迹 8 个视觉 token + 1 个原始类别文本；空间每帧 256 个视觉
  token + CLS + 1 个原始类别文本。普通 self-attention，不加前置路由或额外交叉注意力。
- 同一次空间 attention 的 text-query / visual-key 权重聚合最终 Norm 后的视觉 patch；
  文本和 CLS 不进入聚合值。主路径继续使用 query mask，不改为 visibility mask。
- 正 Support 平均形成类原型、全局分类头、两路运动、BiMHM 基础匹配不变。
- 默认 5 类 × 8 视频分块，关闭完整块重算、开启 MLP/readout 重算；冻结文本缓存。

## 判断分支

### m：原始视觉证据

复用 `QueryEvidenceVerifier.compute_raw_quality` / `RelationDecisionQuality`。
输入为同轨迹对齐的、位置编码/运动/时空建模之前的 Raw DinoTxt patch、原始类别
文本和逐个正负 Support 证据。输出 `[Q,K,T]`；每个类别组只算一次，不随 Query
视频分块重复计算。原有质量头私有证据权重不替换主空间聚合。

### ρ：Post 正负参考关系

1. 主编码只执行一次。在 Support 分块内部，用空间权重读取每个 Support 的
   Post 视觉帧参考 `[S,K,T,D]`，保留已知负样本，不先合并为类原型。
2. 判断有效 mask = 主 query mask ∩ visibility。仅判断分支在该 mask 上重新
   归一化空间权重；主匹配的权重和帧特征保持原样。
3. 在每个 Query 分块内部，用当前类别 c 的 Post patch 比较同类别分支下
   所有正/负 Support 帧。复用旧 ρ 的 count-normalized log-mean-exp 度量，
   再用当前空间权重汇总正负差，经学习到的正 scale 和 sigmoid 得到 `[Q,K,T]`。
4. 未知 Support 标签不是负例；缺少正或负参考时 ρ=1，辅助监督无效。
   Raw m 同样保留原缺参考中性回退，无有效 Query patch 时 m=0。

旧 ρ 的 Post Support+文本额外读出被当前空间注意力权重替代；没有恢复
`SUPPORT_TEXT_FUSION` / `POT_ROUTE` / `LABEL_PRE_TEMPORAL`。
Post patch 只在当前分块内消费，不返回整批 `[B,K,T,N,D]`；跨 Query 分块
只保存小尺寸 Support 帧参考。Support-first 执行顺序可能改变 DropPath 随机数分配，
不能宣称训练轨迹逐位不变。

## 匹配和损失

复用 `finish_query_evidence_aux` 和 `hypothesis_mass_bimhm_logits`：

```
g = m * rho
g_eff = 1 - strength * (1 - g)
C_verified = min(C, g_eff * C)  # unmatched_cost=0, one-sided=True
```

作用于归一化后的帧对相似度，在两个方向的时间最大值归约之前。使用相对于
基础匹配的修正量，保留原 alpha/bias；不缩放后再做 cosine，也不额外添加 log-rho。
训练前 5% optimizer updates 将 strength 从 0 升到 1，评估使用全强度。

```
L = L_global + 0.6 L_base + 0.4 L_final + 0.05 L_m_MIL + 0.05 L_rho_MIL
```

MIL 是带有效帧/未知标签 mask 的视频级监督，不使用帧真值。两个判断开启后
base 和 final 不再相等。`QUALITY_RELATION_STRENGTH=0` 是旧 semantic_relation
模式的遗留值，relation_decision 不读取它，不代表当前 m 被关闭。
`DETACH_INPUTS=True` 在进入判断头前断开证据输入梯度；判断头参数仍学习，
final 匹配损失仍通过帧相似度训练主编码器。

## 配置、checkpoint 与启动

- 当前 `sav.yaml`：m/ρ 都开，独立输出目录，默认从头训练。
- `sav_text_concat_s1_base.yaml`：切换前 sav 的完整有效配置，仍输出 `output/sav`。
- `sav_text_concat_s1_fast.yaml`、S32 和全部旧 early-route 配置保持切换前的有效值。
- v1 签名 `[1,1,1,1]`；双判断 v2 签名 `[2,1,1,1,1,1]`。不能交叉 resume/eval。
- 可显式设置 `TRAIN.CHECKPOINT_FILE_PATH` 为已训练的同 token 数 v1 权重，并设置
  `TRAIN.CHECKPOINT_EPOCH_RESET=True`，初始化编码器、重新初始化 m/ρ；强制重置
  epoch/optimizer/scaler，不是完整续训。旧 early-route 判断 checkpoint 不自动迁移。
- v2 评估或真正续训必须包含完整、匹配的 encoder 和判断头参数。

先 dry-run（只打印启动命令）：

```bash
/home/duanwei/miniconda3/envs/trokens/bin/python tools/run_local.py --cfg configs/trokens/sav.yaml --gpus 0,1 --dry-run
```

正式启动（先确认 GPU 空闲；本次未自动执行）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/duanwei/miniconda3/envs/trokens/bin/python tools/run_local.py --cfg configs/trokens/sav.yaml --gpus 0,1
```

可复用既有 `tools/smoke_label_pre_temporal.py` 做有限步检查；默认合成输入，
`--real-data` 使用本地冻结骨干和真实 SAV。不下载、不保存 checkpoint、不训练完整 epoch。
正确性测试和有限步更新不证明准确率提升；仍需独立对照实验。

## 原适配实现验证（2026-10-24，恢复前记录）

- 全套 CPU 回归：`515 passed, 78 skipped`（CUDA 专用用例因隐藏 GPU 跳过）。
- 新增/相关 CUDA、配置和损失测试：`56 passed`；包括 FP16/BF16 全强度 mρ
  前后向和选择性 checkpoint 对照。
- 最终补充串行/并行分块梯度对照后，GPU 环境下拼接与判断两个专项文件：
  `57 passed`，覆盖关闭版与开启版。
- 真实 SAV、GPU 0/1、FP16、`B=35,K=5,T=8,N=256,D=1024`、每卡累计 2
  个 episode：两卡各完成 2 次成功更新。无 AMP 跳步、无非有限梯度，所有判断头
  参数均有有限梯度，主编码器/分类头更新，两 rank 参数一致。
- 每卡峰值 allocated 约 31.76 GiB，reserved 约 35.69 GiB。第 2 次更新约
  2.66 秒，仅为有限步可执行性检查，不作为稳定吞吐或准确率结论。
- 测试沿用训练 warmup，两个更新的 transport progress 为 0 / 0.002；全强度
  匹配修正另由单元测试覆盖，不能把短测理解为完整收敛验证。
- 比较全部原有 YAML 合并配置哈希：除当前 sav 外全部与改动前一致；新增
  `sav_text_concat_s1_base.yaml` 与原 sav 完全一致。dry-run、`git diff --check` 通过。
- 没有启动正式训练、保存训练 checkpoint 或终止其他 GPU 任务。

## 部分撤销后的恢复检查（2026-10-24）

- 按原实现补丁恢复被撤销的测试、文档、有限步检查工具和分支整理；保留当前
  `sav.yaml` 中用户设置的 `OUTPUT_DIR: output/savo`。
- 隐藏 GPU 后重新执行全套 CPU 回归：`516 passed, 78 skipped`；
  `git diff --check` 通过。未重复运行上述 GPU 检查，未重启或终止现有训练。
