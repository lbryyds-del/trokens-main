# SAV ρ 代码收敛（2026-11-06）

## 本次范围

按用户要求实际删除 m、配对竞争、Query-only 反馈与不再使用的离线实验脚本，不只是关闭开关。不启动训练、不补多 seed、不新增绑定或否决模块。共享训练设施和其他数据集的有效路径保留。

默认配置已统一为 `configs/trokens/sav.yaml`：双侧类别条件化 + 原 ρ + 轨迹簇状态变化/几何证据，加性重评分。学习率 3e-5、10 轮、3 轮 warmup、2 卡 × 累积 2，新标量学习率倍率 100、零 weight decay，与最新 pilot 一致。HoD/CTD 保持关闭。

正式测试用原始完整测试池，10,000 episodes；训练期验证继续沿用 development.csv。新的输出目录 `output/sav/rho_change`，不覆盖历史运行、不自动续训。仅保留一个历史 pilot 配置用于开发/reserved 协议复现。匹配基线分别为 `sav_rho_only.yaml`、`sav_decoupled_base.yaml`。

## 代码内删除

- 两套 Raw m 网络：SemanticRelationQuality、RelationDecisionQuality，及其输入采样、输出、辅助 MIL 和日志字段。
- m × ρ 三态分解；当前 transport 直接使用 ρ。旧报告需要的 null_mass 固定为零，不再有 m 因子。
- M_ENABLE、QUALITY_*、SEMANTIC_MIL_WEIGHT、M_AUX_WEIGHT、PAIR_COMPETITION 和 RHO_QUERY_FEEDBACK 配置。
- 竞争惩罚与 Query-only 梯度覆盖的运行时入口。ρ 与变化证据仍按当前 DETACH_INPUTS 规则运行。
- 启动器不再默认进入竞争实验，改为 sav.yaml。

ρ-only 和最新完整模型的参数名称、有效算式、架构签名保持不变。签名中保留的固定零槽只是兼容现有 checkpoint，不对应任何 m 模块或开关。废弃结构的 checkpoint 不得静默按新配置加载。

## 已删除文件（28 个）

- `trokens/models/pair_competition.py`
- `configs/trokens/sav_competition_c_s1.yaml`
- `configs/trokens/sav_rho_grad_query_s1.yaml`
- `configs/trokens/sav_text_concat_s1_m_rho_on.yaml`
- `configs/trokens/sav_text_concat_s1_base.yaml`
- `configs/trokens/sav_text_concat_s1_fast.yaml`
- `configs/trokens/sav_text_concat_s32.yaml`
- `configs/trokens/sav_text_concat_shared_query.yaml`
- `configs/trokens/sav_shared_support_query.yaml`
- `configs/trokens/sav_label_pre_temporal.yaml`
- `configs/trokens/sav_label_pre_temporal_cosine.yaml`
- `configs/trokens/sav_label_pre_temporal_fast.yaml`
- `configs/trokens/sav_label_pre_temporal_parallel.yaml`
- `configs/trokens/sav_label_pre_temporal_post_readout.yaml`
- `tests/test_pair_competition.py`
- `tests/test_rho_query_feedback.py`
- `tests/test_bound_change_probe.py`
- `tests/test_local_evidence_audit.py`
- `tools/bound_change_probe.py`
- `tools/audit_bound_change.py`
- `tools/analyze_bound_change.py`
- `tools/probe_cross_class_rho.py`
- `tools/analyze_local_evidence.py`
- `tools/check_local_evidence_inputs.py`
- `tools/local_evidence_export.py`
- `tools/prepare_local_evidence_remote.py`
- `tools/run_local_evidence_audit.py`
- `tools/launch_entity_change_pilot.py`

## 恢复与保留

清理前源码备份：`output/code_cleanup_20261106.VttE8P/source_before_cleanup.tar.gz`，覆盖当时的 trokens/tools/configs/tests/docs，包括尚未提交的实验代码。不要直接覆盖恢复整个工作区；如需复现，解包至独立目录。

既有 output、实验日志、预测、checkpoint、run 内 source 快照和历史研究文档未删除。本次没有 git reset、没有修改提交历史、没有启动或停止训练进程。TinyVIRAT 只删除无效的 m 配置字段，数据、模型开关、训练参数和评估协议不变。

## 验证

- 清理前保存合成模型非零融合系数的前向与反向基准；精简后 strict-load，63 个输出、损失、输入梯度及 35 个参数的梯度逐元素一致（atol=rtol=0）。
- 实际已训练 epoch-3 checkpoint 的架构签名、ρ/变化证据权重名称和形状检查通过。未重新跑数据集测试，也未修改权重文件。
- 完整 CPU 回归 163 项通过；随后补充 checkpoint 签名固定值测试，配置组 18 项全部通过。覆盖空参考/未知标签、Query 标签隔离、辅助梯度、学习率倍率和两进程 Gloo。
- run_local.py --dry-run 验证默认入口为完整 SAV 模型；语法检查与 git diff --check 通过。

上述为代码等价性验证，不是新增准确率实验。
