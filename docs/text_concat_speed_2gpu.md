# 单文本版本：不改变模型逻辑的双卡加速（2026-10-23）

## 当前入口与结论

当前 `configs/trokens/sav.yaml` 默认采用每块 5 类×8 视频，
完整块 activation checkpoint 关闭，仅 MLP 和最终 Post Norm/聚合 checkpoint 开启。
`configs/trokens/sav_text_concat_s1_fast.yaml` 是相同执行参数、独立输出目录的入口。
下面的 2×4 数据为切换前对照。损失为 global+base。已运行进程在下次启动时采用新配置。

真实 SAV、本地冻结 DinoTxt、GPU 0/1 空闲时的短测：

| 执行配置 | 秒/优化器更新 | episode/秒（两卡合计） | 每卡峰值 allocated | 每卡峰值 reserved |
| --- | ---: | ---: | ---: | ---: |
| 2×4，完整块 checkpoint | 2.925 | 1.368 | 10.27 GiB | 13.76 GiB |
| 5×8，完整块 checkpoint | 2.753 | 1.453 | 15.48 GiB | 18.73 GiB |
| 5×8，仅 MLP/聚合 checkpoint | 2.452 | 1.632 | 28.55 GiB | 32.36 GiB |
| 5×8，全部 checkpoint 关闭（未采用） | 2.356 | 1.698 | 35.55 GiB | 37.90 GiB |

耗时下降约 16.2%，吞吐提高约 19.3%。每组 5 次成功更新，舍弃第 1 次预热，
对第 2–5 次取平均；每次使用两 rank 中较慢的墙钟时间。
每卡累计 2 个 episode，故每次更新为两卡合计 4 个 episode，不能将秒/update
误读为秒/episode。两个配置的每卡输入均为 `B=35,K=5,T=8,N=256,D=1024`、FP16，
数据加载均为每 rank 4 workers。无非有限梯度、无 AMP 跳步，分类头和主编码器
确实更新，每次更新后两卡参数逐位相同。

这是有限步吞吐检查，不是完整 epoch 或准确率实验；smoke 的一致性检查也计入
墙钟时间，不保证正式训练的绝对秒数相同。最初两卡被其他任务占用的结果不用于此表。

### 后续瓶颈检查与停止优化

选择性 checkpoint 版的单卡 CUDA 分阶段计时（4 次更新，去掉第 1 次，
按每个 episode 计）约为：反向总计 0.501 秒，冻结 DinoTxt 视觉骨干 0.459 秒，
候选时空编码前向 0.199 秒；分别约占完整前反向的 42%、38.5%、16.7%。
MLP/聚合重算约 0.042 秒，已经包含在反向总计中，不能重复相加。
所以低显存不代表没使用双卡，继续减少 checkpoint 的收益也有限。
全关比选择性 checkpoint 再省约 3.9% 墙钟，却增加约 7 GiB allocated；
按用户要求保留选择性 checkpoint 配置，不再继续扩大优化。

另尝试过候选类初始时间 QKV 共享和空间 SDPA，但未完成正式双卡对照。
这两条试验实现及开关已撤回，不属于当前训练代码。
新增的 `test_text_concat_acceleration.py`、临时 profiling/SDPA 探测脚本与临时
YAML 已清理；原有通用 smoke 工具、模型正确性回归测试和测速日志保留。
清理前代码另存 `/tmp/trokens-speed-cleanup.x8o4U1/`，仅供临时恢复，不被训练导入。

## 与指定提交 612e304 的对应关系

检查了 `612e3045b2009f66c97c3cd512f959be106c8617` 的实际源码：

- `sav_label_pre_temporal_fast.yaml` 已采用 5×8，完整块重算关闭，MLP/readout 重算开启。
- `label_conditioned_routing._encode_classes` 把独立候选类折叠到 batch 轴；并非让不同
  类别互相做 attention。当前 concat 仍按这个原则扩展候选类和视频。
- `tools/train_few_shot.py`、`trokens/models/build.py`、数据 loader 和 optimizer 与
  该提交没有差异：仍为 DDP，梯度累积非更新步用 `no_sync`，没有换并行训练算法。
- 旧路径共享初始视觉时间 QKV，当前路径按候选文本拼接；没有为了提速把不同类别
  的完整时间注意力结果直接共享，也没有恢复旧前置路由、门控或独立交叉注意力。

本次增加 concat 专用的 `MLP_CHECKPOINT` / `READOUT_CHECKPOINT`，与完整块 checkpoint
互斥。关闭重算不改变前向公式或可训练参数。固定分块时，测试覆盖输出、输入/参数梯度、
Dropout/DropPath 的 RNG 一致性；分块大小不同仍可能有混合精度舍入和随机数分配差别。

## 损失与实验边界

当前 active/fast 配置明确为 `L_total = L_global + L_base`：

- `BASE_ONLY_LOSS_ENABLE=True`，`DUAL_LOGIT_LOSS_ENABLE=False`，base/final 权重为 1/0。
- `QUALITY_ENABLE=False`、`UNIQUENESS_ENABLE=False`：不构建 m/ρ 判断头。
- 两个 MIL 权重、`ABSOLUTE_MASS_TRANSPORT_STRENGTH` 和 quality relation 强度均为 0。
- base-only 分支直接计算一次 base BCE，不先计算 final BCE 再乘零，也不调用 MIL。
  保留原温度、非有限 logit 清理和未知标签 mask；`q2s_verified_loss=0` 表示关闭。
- 旧版 base==final 时 `.6*L_base+.4*L_final` 与当前目标相同；测试验证数值及梯度。
  这次简化并不构成“收敛或准确率一定改善”的证据。

时空文本数量均为 1、256 条轨迹、8 帧、两路运动、空间 text→patch 权重聚合、
Support/Query 对称条件化、Support 正样本平均、BiMHM、数据增强、学习率和梯度累积不变。
五个历史 early-route 配置及原 S32 配置已显式固定原有损失，合并结果与修改前一致
（忽略新增且关闭的开关），避免继承当前 ablation 而改变历史实验。

## 日志与运行

目录：`output/text_concat_speed_2gpu_20261023/`。

- `idle_2gpu_c2v4_full_w4.log`：空闲双卡、原分块对照。
- `idle_2gpu_c5v8_full_w4.log`：空闲双卡、大分块完整重算。
- `idle_2gpu_c5v8_selective_w4.log`：空闲双卡、选择性重算加速版。
- `idle_2gpu_c5v8_nocheckpoint_w4.log`：空闲双卡、全部重算关闭，未设为默认。
- `profile_selective_c5v8_gpu0.log`：单卡分阶段计时，不与双卡墙钟直接混算。
- 其他 `baseline` / `parallel` 开头的早期双卡日志来自共卡环境，不用于提速结论。
- 单卡 `*_real_gpu1.log` 与早期 smoke 使用 0 workers，仅作分解检查，不与上述
  4-worker 双卡墙钟混算。smoke 现遵循配置的 worker 数，不再强制置零。

正式启动命令（本次未执行；先确认 GPU 0/1 仍空闲）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/duanwei/miniconda3/envs/trokens/bin/python /data3/duanwei/trokens-main/tools/run_local.py --cfg /data3/duanwei/trokens-main/configs/trokens/sav_text_concat_s1_fast.yaml --gpus 0,1
```

输出为 `output/sav_text_concat_t1_s1_fast_c5v8_selective_base`，默认从头训练，不自动续训。
本次调整的是 activation checkpoint（反向重算），没有关闭训练权重的定期保存。
