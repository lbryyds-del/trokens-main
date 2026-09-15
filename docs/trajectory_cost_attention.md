# 轨迹类别响应注意力接入说明

本次修改基于 `2514d8cac02adde580a3d2ec981575035aec1421`，直接移除原 TCN，
不串联、不保留类型选择。时间分支仍使用 `temporal_similarity_refiner` 属性名。

## 计算与开关

输入为类别余弦响应 `[B,K,T,N]`、同序视觉特征 `[B,T,N,D]` 和时间有效性
`[B,T,N]`。类别响应经 `1→16→16` 嵌入得到 E，视觉经归一化、`D→16` 和
LayerNorm 得到 G；Q/K 使用 `[E;G]`，V 只使用 E。单头注意力只在同视频、
同类别、同轨迹内访问自身和前后两帧，不跨轨迹或视频传播。

输出为 `delta_logits = 0.5 * tanh(Wout(context - E))`，Wout 无 bias 且零初始化。
没有有效外部邻帧时修正严格为零；全无效 attention 行为零。原空间掩码不变，
时间有效性沿用原有空间掩码与 `pred_visibility` 的交集。

接入为 `refined_similarity = similarity.float() + tau * delta_logits`，随后仍由原
prototype builder 除以 tau、沿 N 做空间 softmax、聚合当前帧视觉值。
Support、Query 和启用验证时的类参考原型共享同一个模块。
原空间 softmax、Support-Text 融合、BiMHM、质量/独特性判断和所有损失不变。

SAV 配置的 `FEW_SHOT.POT_ROUTE.TEMPORAL_REFINEMENT.ENABLE=True`，因此直接使用新模块；
关闭这一开关仍是严格的无时间修正路径。删除了旧 `TYPE/HIDDEN_DIM/KERNEL_SIZE/
LONG_DILATION/USE_VISIBILITY` 字段；旧命令或 YAML 中的这些字段也应移除。

## 相对独立原型的工程调整

- E 和 Q/K 增加无参数 LayerNorm，平衡 E/G 尺度并稳定注意力 logits。
- 新增分支内同时 detach similarity 与视觉输入；原相似度、原空间路由和原型
  聚合仍向 Pointformer 正常回传。该分支的所有投影参数仍参加训练。
- Linear 投影遵循调用端 AMP，注意力 logits、masked softmax 和残差读出用 FP32。
  模块参数保持 FP32，不应调用 `.half()`。
- Q/K 的拼接投影等价拆为 cost 与 guidance 两次投影的和，避免按 K 复制视觉引导。
  注意力用局部 offset 切片计算；只有 `return_aux=True` 才重建完整 `[B,K,N,T,T]`
  矩阵用于诊断。它不改变注意力公式。
- 参数量仍为 17,765（D=1024），不是与原 TCN 等参数的对照。

## 训练设置与旧权重

学习率沿用 `BASE_LR=1e-4`，cosine、5 epoch warmup、结束 LR `1e-6`、20 epoch 不变。
新模块没有 LR 倍率；其二维权重 WD=0.01，一维参数（含 LayerNorm 和 relative bias）
WD=0。其余模型的优化器分组不变。

episode 配置仍为 `TRAIN_EPISODES=2000`、`TRAIN_OG_EPISODES=False`、seed=1，
原 `set_epoch()` 与随机 label-combo 采样完全不变。
SAV 原有质量/独特性开关状态保持不变：`EVIDENCE_VERIFICATION_ENABLE=False`、
`ABSOLUTE_MASS_ENABLE=False`、MIL weight=0；Query 路由和 dual-logit loss 仍开启。

旧 TCN 权重不能直接当作新模块权重载入。现有仓库 checkpoint loader 只加载
同名同形状参数：旧 TCN 的 5 个时间权重均不匹配新模块，因而不会加载；
共有主干可加载，新分支应新初始化。直接对旧 TCN state dict 调用
`load_state_dict(strict=False)` 会因同名 out_proj 的 Conv1D/Linear 形状不同报错。
不要把旧 optimizer state 当作新结构的正式续训状态。本次未增加自动迁移逻辑。
当前 SAV 仍为 `AUTO_RESUME=False`、`CHECKPOINT_FILE_PATH=""`、`NEW_TRAIN=True`。

## 本次验证（2026-09-14）

环境为本仓库 Python 3.10、PyTorch `2.0.1+cu117`，CUDA 为 A100 PCIe 40GB。
运行当前 `tests`：`129 passed in 7.90s`；`git diff --check` 与修改文件的
py_compile 均通过。
专项覆盖零初始化等价、批量/单视频、类别/轨迹隔离及置换、局部性、掩码、
短轨迹/单帧、常数响应、温度换算、Query 标签隔离、detach、优化器注册、
旧权重形状、CUDA AMP 和局部切片/完整数学公式等价。

真实 SAV train episode（5 Support + 30 Query；类别 `[4,7,0,13,8]`）使用本地
冻结 DinoTxt 塔提取真实视频/文本特征；dense 特征为 `[35,8,16,16,1024]`。
配对测速缓存这些冻结特征，只测可训练 Pointformer、运动支路、路由及原损失的
前后向；同一 episode、相同 RNG，交替 ON/OFF，预热后各取 5 次。
为覆盖已激活分支的开销，仅在诊断进程里临时设置非零输出权重。

| 路径 | 前后向中位耗时 | 峰值 allocated 显存 |
| --- | ---: | ---: |
| 时间 OFF | 223.512 ms | 15,921.66 MiB |
| 新 attention ON | 240.217 ms | 16,312.49 MiB |

可训练路径耗时增加 7.474%，显存增加 390.83 MiB。冻结特征提取此次为
2,021.353 ms（分块提取，非稳态测速）；不能据此推断完整 epoch、DDP 或数据加载耗时。

真实特征上零初始化时开关两端的原型和 logits 逐位一致；第一步输出层有梯度、
Q 梯度为零，输出层更新后第二步 Q 梯度非零；新参数梯度均有限。
两步优化器更新只在独立诊断进程内存执行，没有保存 checkpoint、启动正式 epoch
或修改正式训练状态。

`return_aux=True` 可检查 attention 与 delta_logits，并配合原 prototype builder 对照
空间权重。此次临时激活但尚未训练的 Query 诊断：帧内残差 std `0.00024274`、
权重 TV `0.00006450`、Top-1 patch switch `0.3333%`、自参考比例 `41.64%`。
这些数值只证明计算与诊断路径可用，不是训练收益证据。

正式增益仍需同协议重训和配对 ON/OFF、轨迹对应扰动对照，不能从 smoke 的
BCE 下降或注意力/热图平滑直接推断 Base/Novel/HM 改善。
