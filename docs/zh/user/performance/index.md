# 性能

PyPTO 的性能工作遵循 **测量 → 定位 → 优化 → 验证** 循环，分为两个轨道：

| 轨道 | 范围 | 页面 |
| ---- | ---- | ---- |
| 共享方法论 | 工具和测量循环（两个轨道通用） | [00-methodology](00-methodology.md) |
| 单节点 | Kernel、tile、流水线、matmul、内存、调度 | [01-single-node](01-single-node.md) |
| 分布式 | 集合通信成本、ring vs mesh、跨 rank 偏差、总线带宽 | [02-distributed](02-distributed.md) |
| 案例 | 端到端工作示例 | [03-cases](03-cases.md) |

> **前置知识：** 分布式轨道需要 [分布式编程](../distributed/00-model.md)；
> kernel 编写基础见[入门指南](../00-getting_started.md)。

## 工具矩阵

| 工具 | 观测对象 | 入口点 |
| ---- | -------- | ------ |
| 编译时性能提示 | 代码模式 | `report/perf_hints.log` |
| 基准测试 span tree | 端到端分段 | `pypto.runtime.benchmark` → `stats.print_mean_tree(spread=...)` |
| In-core msprof | 每个 kernel 的周期数 | op-simulator + Insight traces |
| 内存映射 | 片上缓冲区 | `pypto.tools.memory_map` → HTML |
| Scope 统计 | 运行时水位 | `RunConfig(enable_scope_stats=True)` |
| L2 swimlane / PMU / dep gen | 任务调度 | `RunConfig(enable_l2_swimlane / enable_pmu / enable_dep_gen)` |

## 相关链接

- [分布式](../distributed/index.md) — 编写分布式程序
- [入门指南](../00-getting_started.md) — `ir.compile()` 和 `RunConfig`
- [Simpler 运行时](https://hw-native-sys.github.io/simpler/) — 调度器内部机制
