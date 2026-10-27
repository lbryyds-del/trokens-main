# Support 文本拼接 + 通用共享 Query（m/ρ 全关闭）

本页为 v3 对照：`configs/trokens/sav_text_concat_shared_query.yaml`。
当前默认 `sav.yaml` 已改为[双侧关闭文本解耦](shared_support_query.md)，不再与本页配置相同。
输出：`output/sav_text_concat_shared_query_m_rho_off`。
原双侧拼接 m/ρ 开启版保留为 `sav_text_concat_s1_m_rho_on.yaml`；
原关闭版、fast、S32 和旧前置路由配置改为继承该保留版本，有效配置不变。

## 本组实验的边界

- **Support 不变**：每个候选类分别做时间/空间单文本拼接，使用原空间
  text-query / visual-key 注意力权重聚合最终视觉 patch。正 Support 形成类原型。
- **Query 关闭解耦**：只对 `[Q,T,N,D]` 做一次通用 temporal-then-spatial
  self-attention；不拼类别文本、不做类别条件化、不扩展成 `[Q,K,T,N,D]`。
  与 Support 共享原时空模块参数，不新建第二套 Transformer。
- **Query 聚合**：对最终 Norm 后的每帧有效视觉 patch 取均值；不使用文本权重，
  CLS 不参与帧匹配聚合。当前 `QUERY_MASK_MODE=all` 即对 256 个 patch 求均值；
  空 mask 的帧返回零向量。得到 `[Q,T,D]`，没有 K 维。
- **匹配**：一份 Query 特征对所有 K 个 Support 类原型，输出 `[Q,K]`。
  不是 K 路 Query 各自只与同类 Support 做对角匹配。类别轴只出现在匹配结果中。
- **m/ρ 全关闭**：不构建两个判断头，不运行 Raw 证据判断、Post 正负参考判断、
  `m*rho` 匹配修正或两个 MIL 辅助损失。
- 保留 appearance + HOD + CrossMotion、位置编码、全局分类监督、采样与优化设置、
  5×8 分块、MLP/readout checkpoint、冻结文本缓存。Query 只按视频分块，不按类别循环。

Support 保留文本条件，所以这是混合结构消融，不是完全无文本 baseline。
关闭 m/ρ 不等于文本从整个模型消失。与双侧拼接相比改变了 Query 编码/聚合；
不承诺训练轨迹、准确率或实际吞吐与原版相同。

## 分数和损失

对每个 Query q、Support 类 c，先算帧间余弦矩阵 `C[q,c,t,s]`：

```text
S[q,c] = 0.5 * (mean_t max_s C[q,c,t,s] + mean_s max_t C[q,c,t,s])
z_final = z_base = 10 * S - 2
L = L_global + L_base
```

保留当前分数尺度；没有额外改回历史负双向距离、添加 ReLU 或改变余弦分母。
不乘 m/ρ、不扣证据惩罚、不重复计算 final BCE。

合并配置中的关键值：

```yaml
TEXT_CONCAT_ST.SHARED_QUERY_ENABLE: True
TEXT_CONCAT_ST.SHARED_SUPPORT_ENABLE: False
QUERY_CLASS_MATCHABILITY.QUALITY_ENABLE: False
QUERY_CLASS_MATCHABILITY.UNIQUENESS_ENABLE: False
QUERY_CLASS_MATCHABILITY.ABSOLUTE_MASS_TRANSPORT_STRENGTH: 0.0
QUERY_CLASS_MATCHABILITY.BASE_ONLY_LOSS_ENABLE: True
QUERY_CLASS_MATCHABILITY.DUAL_LOGIT_LOSS_ENABLE: False
QUERY_CLASS_MATCHABILITY.BASE_LOGIT_LOSS_WEIGHT: 1.0
QUERY_CLASS_MATCHABILITY.VERIFIED_LOGIT_LOSS_WEIGHT: 0.0
LEARNED_VERIFIER.SEMANTIC_MIL_WEIGHT: 0.0
LEARNED_VERIFIER.RELATION_MIL_WEIGHT: 0.0
```

父级 `QUERY_CLASS_MATCHABILITY.ENABLE=True` 只保留训练时的未知标签 mask；
两个子判断开关均为 False，实际 `query_evidence_verifier=None`。
历史命名 `POT_ROUTE.QUERY_PARTIAL_ENABLE=True` 仍供统一评分/监督接口使用，
不代表 Query 继续按类拆分；该实验实际 Query 原型是三维 `[Q,T,D]`。

全局分类沿用同一个 head：Support 对候选类编码的全局向量先取平均，Query 使用
通用编码的全局向量，两者按原视频顺序拼回 batch，每个视频只算一次分类损失。

## 启动与 checkpoint

检查命令（不启动训练）：

```bash
/home/duanwei/miniconda3/envs/trokens/bin/python tools/run_local.py --cfg configs/trokens/sav_text_concat_shared_query.yaml --gpus 0,1 --dry-run
```

确认所选 GPU 可用后，从头训练：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/duanwei/miniconda3/envs/trokens/bin/python tools/run_local.py --cfg configs/trokens/sav_text_concat_shared_query.yaml --gpus 0,1
```

这只是准备好的启动命令，本次修改不会自动启动/停止正式训练。
配置继承 `AUTO_RESUME=False`、空 `CHECKPOINT_FILE_PATH`。
新架构签名是 `[3,1,1,1]`，防止把双侧拼接 v1/v2 的 checkpoint 当作同一结构续训或评估。
不能直接复用 m/ρ 开启版或双侧拼接关闭版 checkpoint；本组对照默认从头训练。

## v3 当时的验证记录

- 切换默认入口后，隐藏 GPU 的全套 CPU 回归：`540 passed, 80 skipped`；CUDA 专用用例跳过。
- 新分支覆盖：与普通 `pt_forward` 对照、Query 编码次数 Q 而非 Q×K、候选文本/
  类别数量不影响 Query、Support 原型/权重/全局输出不变、逐对匹配公式、
  无 m/ρ/MIL 调用、空 mask、前后向、运动保留、分块与 checkpoint 等价、
  checkpoint 架构隔离，以及双进程 CPU DDP 更新/参数同步。
- 默认入口切换前后，全部已有 YAML 的合并配置哈希仅 `sav.yaml` 改变；
  保留的 `sav_text_concat_s1_m_rho_on.yaml` 与切换前 sav 完全一致，
  新 sav 与独立共享 Query 配置完全一致。旧对照不再继承可编辑的默认 sav。
- 启动器 dry-run 和 `git diff --check` 通过。没有启动正式训练；这些测试不证明
  GPU 峰值显存、训练提速幅度或准确率收益。
