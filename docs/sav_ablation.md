# SAV 消融开关与损失联动

统一入口：`configs/trokens/sav.yaml`，只改下面四项，不再手工同步旧 shared、
判断头、transport、base/final 或 MIL 开关。Query/Support 都关时自动回到原始
Trokens 默认支路和匹配 BCE；任意一侧开启则保留现有解耦匹配，不另设匹配模式。
当前四项全部关闭；修改配置不会自动启动或重启训练。

```yaml
FEW_SHOT:
  DECOUPLING:
    QUERY_ENABLE: False
    SUPPORT_ENABLE: False
    M_ENABLE: False
    RHO_ENABLE: False  # rho，即讨论中的 p
```

## 编码与匹配

| Query | Support | 主编码 / 匹配 | m、rho |
| --- | --- | --- | --- |
| 关 | 关 | 原始 `pt_forward` + `process_patch_tokens` 负距离匹配，1:K | 必须关 |
| 关 | 开 | Support 文本条件化；Query 通用，1:K | 必须关 |
| 开 | 关 | Support 通用；K 路 Query 对 K 个 Support 类原型，K:K | 可分别开关 |
| 开 | 开 | 两侧文本条件化；K 路 Query 对对应 Support 类原型，K:K | 可分别开关 |

K 由 episode 候选类数决定，5-way 时 K:K 即 5:5。含义是第 c 路 Query 匹配
第 c 类 Support 原型，不是视频数量配对，也不是不同候选类之间的 K×K 交叉匹配。
Query 真值不参与编码或聚合。

- 解耦开启：时间沿轨迹拼文本、空间每帧拼文本；普通 self-attention。
  最终用同次空间注意力的文本→视觉权重聚合视觉 patch，文本和 CLS 不作为匹配聚合值。
- 单侧解耦关闭：该侧每视频只普通编码一次，不复制类别轴、不拼文本，每帧视觉 patch 求均值。
  Support 类别轴由已知正标签在编码之后形成；多标签视频可参与多个类原型。
- 任意一侧解耦开启时维持当前归一化和 `10*S-2` 双向帧相似度。
  两侧都关闭时不再进入文本模块的 shared 分支，直接走下面的原始路径。
- 分类头、seen 类监督及权重不变：关闭侧用视觉 CLS，开启侧用候选文本分支 CLS 均值。
  没有恢复此前额外无文本 Query CLS 分支，因此分类输入仍随该侧编码方式变化。
- 若 Support 关闭而 rho 开启，rho 的 Support Post 参考来自同一次普通编码，
  对有效可见视觉 patch 均值读出；只扩展小型帧参考，不复制或重编码 Support patch。
  m 仍保留独立的 Raw 证据读取，不把其内部文本读出称为 Support 主编码解耦。

## 两侧都关闭：原始 Trokens 默认支路

恢复原始模型入口、聚合和匹配 BCE，不回退学习率、每轮 episode 重采样、梯度累积、
运动模块、数据划分或指标口径。m/rho 必须同时关闭；非法组合报错，不静默关闭。

1. 解析器令 `TEXT_CONCAT_ST.ENABLE=False`、`POT_ROUTE.QUERY_PARTIAL_ENABLE=False`、
   `QUERY_CLASS_MATCHABILITY.ENABLE=False`，base-only/dual-logit 专用损失开关都关。
   不创建文本模块、文本编码器、tokenizer 或判断头，不走解耦分块/选择性 checkpoint。
   前向直接复用原本的 `Pointformer.forward -> pt_forward`。
2. 输出每个视频的视觉 patch，不提前做帧均值。复用原来的
   `support_query_split_multilabel`：先把同类正 Support 视频的 patch 求均值。
3. 训练、验证、测试都复用既有 `process_patch_tokens`：Support 原型和 Query
   各自经过 ReLU，再做空间 patch 均值，余弦分母保留 `norm(q)*norm(s)+0.01`。
4. `d=1-cos`，分数为两个方向最近帧距离的负和：
   `z = -(sum_t min_s d[t,s] + sum_s min_t d[t,s])`。
   不乘 `LOGIT_SCALE`、不加 `LOGIT_BIAS`；其内部有效值设为 1/0。
5. 总损失仍为 `L_global + BCEWithLogits(z / TEMPRATURE, y)`，不是删除匹配监督，
   也不是仅把 `10*S-2` 中的 -2 改成 0。全局分类仍用同一视觉 CLS、只监督 seen 类。
   final/m/rho 损失和 transport 都为 0；日志仍记到 `q2s_base_loss`，verified 记 0。

这是**原始支路和原始损失**，不是历史整套训练协议复现，也不是最新
`79.22 / 75.26 / 77.19` 的无解耦 `10*S-2` 对照。现在四开关对比同时涉及
编码/聚合/匹配差异，不能将相对这个 baseline 的全部差值归因于解耦。

不再为比较保留文本加载的 RNG 消耗；初始化遵循原始无文本模型。因此不要声称
与先前文本初始化版本的所有运动模块初始权重一致。原始整批默认分支不使用
`TEXT_CONCAT_ST` 的分块/checkpoint 配置，显存/速度需另行实测。
先前在文本 dispatcher 内返回 patch 的临时 `legacy_distance` 实现及 v7 签名已移除。
无文本默认模型与 v5 校准匹配模型互相拒绝续训/评估，需要独立训练。

## 判断、匹配修正和损失必须同步

下表使用 `sav.yaml` 默认权重，`L_global` 是 seen 类全局分类 BCE，
`L_base/L_final` 是 episode 少样本匹配 BCE，`L_m/L_rho` 是对应判断头的 MIL 辅助损失。

| m | rho | 生效的匹配证据 | 总损失 |
| --- | --- | --- | --- |
| 关 | 关 | 不修正，final = base | `L_global + L_base` |
| 开 | 关 | m，rho = 1 | `L_global + 0.6 L_base + 0.4 L_final + 0.05 L_m` |
| 关 | 开 | rho，m = 1 | `L_global + 0.6 L_base + 0.4 L_final + 0.05 L_rho` |
| 开 | 开 | m × rho | `L_global + 0.6 L_base + 0.4 L_final + 0.05 L_m + 0.05 L_rho` |

关闭的头不实例化、不参与 optimizer、不输出辅助 logits、对应辅助权重归零。
两个头都关闭时，不重复计算 final BCE；transport 强度为 0，base/final 权重为 1/0。
开启判断时保留原 5% transport warmup，不改其 Raw/Post 输入、正负参考、
one-sided 匹配机制或未知标签屏蔽。它不是额外增加一份独立的“m 匹配损失”：
m/rho 共同决定 final 匹配分数，final BCE 监督这个分数，各头另有各自的 MIL。

`FEW_SHOT.MATCHING` 中 `.6/.4/.05/.05` 是**开启时**的权重设定。
即便配置保留这些数，关闭的头/损失也不会执行。实际生效值在启动日志的
`QUERY_CLASS_MATCHABILITY` 中展开，启动器同时打印简短说明。
`CLASS_LOSS_LAMBDA` 乘 `L_global`；`Q2S_LOSS_LAMBDA` 乘整个匹配与 MIL 之和，当前均为 1。

Query 关而 m/rho 开属于非法组合，配置解析直接报错，不偷偷改变用户开关。
开启判断却把其辅助损失、final 权重或 transport 设为零也会报错，避免名义开启。
四开关在 YAML 与命令行合并后统一解析，训练、评估和直接模型构造共用同一解析器。
旧实验配置没有四开关时，仍按其原有显式值运行；不能混用新开关与旧前置路由。

## 使用

编辑 `sav.yaml` 后先检查（不启动训练）：

```bash
/home/duanwei/miniconda3/envs/trokens/bin/python tools/run_local.py \
  --cfg configs/trokens/sav.yaml --gpus 0,1 --dry-run
```

确认卡可用后手动启动，建议每组消融用单独输出目录：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/duanwei/miniconda3/envs/trokens/bin/python tools/run_local.py \
  --cfg configs/trokens/sav.yaml --gpus 0,1 --output-dir output/sav_q0_s0_trokens_default
```

仍为从头训练（`AUTO_RESUME=False`、`NEW_TRAIN=True`、空 checkpoint 路径）。
改动开关后不可把另一组合的 checkpoint 当成同结构续训/评估；架构签名会校验。
既有 v1/v2/v3/v5/v6 签名保留；原始默认模型没有 text-concat 签名。
此前四开关全关的已归档配置也会按新规则解析为原始支路，不能直接搭配其 v5 权重。
若要复测历史无解耦 `10*S-2` 控制组，用 `sav_shared_support_query.yaml` 的显式旧配置
和该 run 的准确 checkpoint；不要把新 `sav.yaml` 当成其历史配置。

## 清理范围

- 从默认 SAV 移除无关的旧前置路由、Support/text 融合、旧语义残差配置、
  重复 shared/质量/独特性/损失开关；保留实际使用的训练、数据、运动与执行参数。
- 删除无调用方的 SubBatchNorm 模块及统计入口、Permute/round_width 死代码。
  主线 episode 输出工具不再依赖旧前置解耦模块；历史前置实现改为按需导入。
- 历史前置配置、其他数据集和对应仍被引用的实现保留，不能把“当前 SAV 关闭”
  当成“整个仓库无用”。不删除轨迹、数据、训练日志或模型检查点。
- 合并保留一份轻量开关/损失矩阵回归，移除历史实验调试测试和两个临时测速工具；
  正式评估入口 `tools/test_few_shot.py` 保留。
- 删除前的工作区备份（含未提交修改）：
  `/tmp/trokens-ablation-cleanup-20261027-F1EWCt/worktree-before.tar.gz`。
  可从中按需取回文件；不要直接整包覆盖后续新修改。

## 验证记录

- 原始默认支路切换后：隐藏 GPU、限制 CPU 线程，`pytest -q tests` 为
  `141 passed, 5 skipped`；启动器 `--dry-run` 和 `git diff --check` 通过。覆盖自动分支、原始 train/test
  打分一致、多标签 Support 先平均再 ReLU、`+0.01` 余弦分母、scale/bias 无效、
  BCE/梯度、忽略文本 checkpoint 配置、旧 v5 权重隔离及双进程 CPU DDP。
  使用假视觉骨干的真实构造器测试确认不加载文本资源，且同 seed 下与显式关闭路由的
  默认模型全部参数和 CPU RNG 状态一致；这不等于与历史文本初始化版本一致。
  未启动正式训练或 GPU 性能/精度测试；显存和最终指标须在正式实验中确认。
- 移除旧测试前：`604 passed, 82 skipped, 1 deselected`。排除的旧用例要求
  “Support 关闭必须 Query 关闭”，该限制已被本次独立开关替代；配置测试另外 27 项通过。
- 清理后 `pytest -q tests`：`140 passed, 5 skipped`，隐藏 GPU，仅保留 7 个测试文件。
  覆盖全部 16 种开关组合（10 合法、6 报错）、真实 5-way 形状、精确损失加权、
  空 mask/未知标签、有限梯度、全局 CLS、checkpoint 隔离和双进程 CPU DDP 更新。
- 与备份实现逐张量对比：6 种已有组合 × 3 组执行配置，在 eval 模式下
  627 个输出张量及梯度逐位一致；新增 Query 开/Support 关另做前后向与 DDP 测试。
- 比较全部 19 个 YAML：18 个历史配置合并值不变；默认 SAV 除新增公开开关外，
  仅两个无效遗留值恢复默认（已关闭 POT_ROUTE 的温度，以及 relation_decision 不读取的
  QUALITY_RELATION_STRENGTH），模型/数据/采样/优化参数不变。
- 训练/评估模块导入、合法与非法组合启动器检查、`git diff --check` 通过。
  没有启动正式训练、没有执行 CUDA 精度或性能测试；这些检查不代表准确率收益。
