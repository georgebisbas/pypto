# 性能案例

下面每个案例都遵循相同的模式：**基线 → 调查 → 变更 → 效果 → 验证**。

> **状态：** 性能案例已规划但尚未编写。代表性工作负载的真实测量数据尚
> 未积累。用户手册计划 ([USER_MANUAL_PLAN_EN §8.1 item 5](https://github.com/hw-native-sys/pypto/issues/2120))
> 目标是在第一版中提供方法论和相对趋势，待数据积累后回填测量值。
>
> 单节点性能技术（[01-single-node](01-single-node.md)）和分布式性能指南
> （[02-distributed](02-distributed.md)）现已可用。
>
> 计划案例：
>
> - **单节点：** Tile 维度粒度、自动 Tiling 诊断、假任务依赖、Ring sizing
>   竞技场重建。
> - **分布式：** Mesh-to-ring 集合通信转换、驻留分片 vs 每分发 H2D。

## 相关链接

- [00-methodology](00-methodology.md) — 测量循环和工具
- [01-single-node](01-single-node.md) — 单节点性能技术
- [02-distributed](02-distributed.md) — 分布式性能和总线带宽
