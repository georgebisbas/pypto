# 单节点性能

下面每项技术都说明**适用场景、成本、启用方式和验证方式**。

## 分区与并行

### `pl.split(SplitMode)`

在 `CORE_GROUP` 区域内对跨核数据传输做几何切分——并非独立调用，而是
传入 `pl.at(..., optimizations=[...])`。

- **适用场景：** `CORE_GROUP` 区域的数据需要在其核心间切分
- **成本：** 需要 split 兼容操作
- **启用方式：** `pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.UP_DOWN)])`——可选模式为 `NONE`、`UP_DOWN`（高度对半分）、`LEFT_RIGHT`（宽度对半分）
- **验证方式：** 与非 split 基线 benchmark 对比

### `pl.split_aiv`

将计算分配到 AIV 核心。

- **适用场景：** 向量密集型工作负载
- **成本：** 仅 AIV
- **验证方式：** 内存映射中确认 AIV 核心被使用

### `pl.spmd(N)`

SPMD 并行——启动 N 个相同 kernel 副本。

- **适用场景：** 天然可并行的工作负载
- **成本：** N 倍内存占用
- **启用方式：** `with pl.spmd(n):` 或 `for i in pl.spmd(n):`
- **验证方式：** Benchmark 显示近线性加速

### `pl.cluster()`

协同调度 AIC + AIV 共享物理集群资源。

- **适用场景：** 受益于并发 AIC + AIV 执行
- **成本：** 需要集群兼容的 kernel 对
- **验证方式：** L2 swimlane 显示并发 AIC/AIV 执行

## 流水线与展开

### `pl.pipeline`

跨迭代重叠计算的软件流水线。

- **适用场景：** 迭代独立的循环体
- **成本：** 增加寄存器和 buffer 压力
- **启用方式：** 用 `pl.pipeline` 包裹循环体
- **验证方式：** Benchmark 显示减少的每迭代延迟

### `pl.unroll`

完全展开编译时已知的循环。

- **适用场景：** 编译时已知的小循环次数
- **成本：** 更大的二进制文件
- **验证方式：** 检查编译产物中的展开代码

## Matmul 路径

### AutoTileMatmulL0

自动 tiling pass 选择 L0 matmul tile 大小。

- **启用方式：** 默认开启
- **成本：** 仅有编译时分析开销
- **验证方式：** IR dump 显示 tiled matmul 维度

### `enable_pypto_l0c_double_buffer`

为 L0C 输出 buffer 启用双缓冲。

- **适用场景：** Matmul 密集型工作负载
- **成本：** 2x L0C buffer 分配
- **启用方式：** `ir.compile(..., enable_pypto_l0c_double_buffer=True)`
- **验证方式：** Benchmark 显示 sched 时间减少

## 内存

### `target_memory`

为 tile 选择片上内存空间（`MemorySpace.DDR` 为片外；`.Vec`/`.Mat`/
`.Left`/`.Right`/`.Acc` 为片上 buffer）。并非每个 op 都接受所有空间：
`pl.load`（DDR 到片上）只接受 `.Vec` 或 `.Mat`——其余值会抛出
`ValueError`。要把数据放到 `.Left`/`.Right`，需先 load 到 `.Vec`/`.Mat`，
再用 `pl.move` 搬运。`.Acc` 是 matmul 累加的输出空间，不是
`load`/`move` 的目标。

- **适用场景：** 数据放置优化
- **成本：** 片上空间更快但远小于 DDR
- **启用方式：** `pl.load(a, [0, 0], [rows, cols], target_memory=pl.MemorySpace.Vec)`
- **验证方式：** 内存映射确认分配到指定空间

### MemoryReuse vs `memory_planner=PTOAS`

内存规划策略。

- **MemoryReuse：** 默认——不重叠的生命周期重用 buffer
- **PTOAS：** 更积极的规划，可能导致编译时间增加
- **验证方式：** 内存映射显示分配和重用

## 调度

### `predicate=`

在分发点动态跳过任务。

- **适用场景：** 条件执行
- **成本：** 可忽略
- **验证方式：** 依赖图显示跳过的边

### `no_dep`

对调用点放弃自动依赖推断。

- **适用场景：** 表面重叠但实际独立的操作
- **成本：** 错误使用导致竞态条件
- **验证方式：** 依赖图确认操作间无边

### `manual_scope`

关闭自动依赖跟踪。

- **启用方式：** `with pl.scope(mode=pl.ScopeMode.MANUAL):`
- **成本：** 必须显式声明所有边
- **验证方式：** 依赖图精确匹配声明的边

## 数据驻留

### `DeviceTensor`

在分发之间保持 tensor 驻留在设备上。

- **适用场景：** 权重、查找表等可重用数据
- **成本：** 减少设备内存
- **启用方式：** `rt.alloc_tensor(shape, dtype, init=host_data)`
- **验证方式：** 第二次分发起不存在 H2D/D2H span

## 相关链接

- [00-methodology](00-methodology.md) — 测量循环和工具
- [02-distributed](02-distributed.md) — 分布式性能技术
- [03-cases](03-cases.md) — 端到端工作示例
