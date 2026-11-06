# SAV 角色与证据开关

当前默认配置是 `configs/trokens/sav.yaml`，含 Query/Support 双侧条件化、ρ 和两路轨迹变化证据。m 已移除，不再支持 M_ENABLE 或旧质量头配置。

## 维护入口

| 配置 | Query / Support 条件化 | ρ | 变化证据 |
| --- | --- | --- | --- |
| sav.yaml | 开 / 开 | 开 | 状态 + 几何 |
| sav_rho_only.yaml | 开 / 开 | 开 | 关 |
| sav_decoupled_base.yaml | 开 / 开 | 关 | 关 |

这些配置采用相同训练参数与完整测试池，使用不同输出目录；本次只维护配置，没有运行消融或多 seed。

公开角色开关只有 `FEW_SHOT.DECOUPLING.QUERY_ENABLE`、`SUPPORT_ENABLE`、`RHO_ENABLE`。ρ 要求 Query 条件化开启；变化证据要求双侧条件化和 ρ 开启。关闭 ρ 时应同时关闭 ENTITY_CHANGE；直接选基线配置即可。

## 自动联动

- ρ 开启：匹配损失 `0.6 L_base + 0.4 L_final + 0.05 L_rho`。
- 两路变化开启：各加入权重 0.05 的有效证据 BCE；最终分数为双向加性重评分。
- ρ 关闭、至少一侧条件化开启：只用 `L_base`，无重复 final BCE。
- 两侧条件化都关闭：保留原始 Pointformer 负双向距离匹配路径。
- 全局分类损失独立保留。

配置检查在模型加载前执行，未知/废弃开关不会被静默解释为现有方案。默认 HoD/CTD 关闭。原 ρ 同条件正负对比、one-sided transport 和变化分支输入断梯度保持不变。

清理前的旧配置和实现见 [恢复记录](rho_cleanup_20261106.md)。
