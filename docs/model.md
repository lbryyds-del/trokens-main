# 当前 SAV 模型

默认入口：`configs/trokens/sav.yaml`。本页描述 2026-11-06 收敛后的实现，其他数据集仍按各自配置运行。

## 模型路径

1. 冻结 DinoTxt，读取既有轨迹缓存，构造 Query/Support 的同候选类别条件化表示。
2. 原 ρ：同条件正负 Support 相对证据，以单向 transport 修正外观匹配。
3. 轨迹变化：轨迹簇状态变化与几何运动分别生成正负相对证据。
4. 分数端双向相加：`z = z_rho + lambda_state * L_state + lambda_geo * L_geo`。

m 已从网络、损失、配置及诊断输出中删除；竞争配对、Query-only 梯度反馈也已删除。没有增加对应绑定、可靠度门控或单向变化否决。HoD/CTD 特征融合保持关闭；轨迹几何证据不是 HoD/CTD 分支。

Query 真值只用于监督和评估，不参与推理证据的构造。未知 Support 标签不作负标签。缺少正/负有效参考时，该证据不参与融合或辅助监督。

## 实现入口

| 文件 | 职责 |
| --- | --- |
| `trokens/models/text_concat_conditioning.py` | 文本条件编码、同条件参考、匹配及证据接入 |
| `trokens/models/query_evidence_verifier.py` | 原 ρ 正负相对核验与 transport |
| `trokens/models/entity_change_verifier.py` | 轨迹簇变化证据、几何证据、加性融合 |
| `trokens/datasets/entity_tracks.py` | 轨迹簇与几何元数据，按不重复轨迹计数 |
| `tools/few_shot_multilabel.py` | base/final BCE、ρ MIL 和变化通道辅助 BCE |
| `trokens/models/optimizer.py` | 新标量独立参数组及学习率倍率 |

## 配置与训练

5-way 1-shot；每类 6 个训练 Query；每轮 2,000 episodes；8 帧、256 个采样点。基础学习率 3e-5，10 轮、3 轮 warmup、cosine，2 卡 × 累积 2。变化分支输入断梯度，融合系数零初始化；六个新增标量使用 100 倍学习率、零 weight decay。融合改变 final BCE，因此主路径梯度仍会改变。

训练期验证沿用 pilot 的 development.csv；训练数据池没有改成该子集。正式测试 `TEST.DATA_CSV: ""`，即数据集原始测试池，10,000 episodes；预设末轮为 epoch 10。历史 pilot 的 reserved 测试入口只保留在独立配置中。

维护中的 SAV 配置：

- `sav.yaml`：完整模型，输出 `output/sav/rho_change`。
- `sav_rho_only.yaml`：同协议 ρ-only 基线。
- `sav_decoupled_base.yaml`：同协议双侧解耦、不使用 ρ 或变化证据。
- `sav_entity_change_s1.yaml`：原 pilot 协议，含 development/reserved 划分，仅用于复现。

保持最新完整模型和原 ρ-only 的参数名、架构签名不变。旧 m/竞争/反馈模型不再受当前代码支持，需使用清理前备份或对应 run 的 source 快照。不能把新旧结构的权重静默混载。

## 运行

```bash
python tools/run_local.py --cfg configs/trokens/sav.yaml --gpus 0,1
```

当前修改不自动启动训练或测试；多 seed 暂缓。旧结果和 checkpoint 未删除。详见 [清理记录](rho_cleanup_20261106.md)。
