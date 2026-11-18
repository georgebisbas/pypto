# 特性矩阵

什么在哪里被支持，以及那些你会在 dump 里读到、却很少手写的类型。

## Backend

PyPTO 面向两代 Ascend。`backend_type` 在编译期选择其一，它同时改变所跑的 pass 与所发出的 ISA。

| 特性 | `Ascend910`（A2/A3） | `Ascend950`（A5） |
| ---- | -------------------- | ----------------- |
| 平台字符串 | `a2a3`、`a2a3sim` | `a5`、`a5sim` |
| cube/vector 混合作用域 | 支持 | 支持 |
| 跨核环（`pl.cross_core_slot`） | 支持 | 支持 |
| GM pipe buffer 注入 | 支持（`InjectGMPipeBuffer` 受 backend 门控） | 无 |
| 多跳 `tile.cast` 展开 | 不需要 | `INT32 -> FP16` 经 FP32 展开（[精度](../precision/00-workflow.md)） |
| MX matmul 系列 | 无 | 支持 |

逐算子的支持情况见 [PTOAS 算子状态](../../dev/ptoas-op-status.md)，那张表是生成的而非手工维护；上表覆盖的是用户会撞上的特性级差异。

## 内存规划器

| 规划器 | 谁来分配 | 备注 |
| ------ | -------- | ---- |
| `PYPTO`（默认） | `MemoryReuse` + `AllocateMemoryAddr` 烘焙地址 | [内存图](../tools/02-memory-map.md)能画出结果 |
| `DSA_RP` | PyPTO 自带的容量受限规划器 | |
| `PTOAS` | ptoas `PlanMemory` 负责复用与寻址 | PyPTO 的分配 pass 被跳过，因此 pass dump 里没有偏移 |

## 校验级别

| 级别 | 会跑什么 |
| ---- | -------- |
| `NONE` | 不做 IR 校验 |
| `BASIC`（默认） | 每个 pass 之后的结构检查 |
| `ROUNDTRIP` | 额外重新解析每份 dump —— 明显更慢，用于追查畸形 IR |

## 你会读到但很少手写的类型

它们出现在 pass dump 与打印器发出的 IR 里。它们属于公开面，是因为打印器要经它们往返，而不是因为 kernel 作者会手写它们。

| 类型 | 是什么 | 取值 |
| ---- | ------ | ---- |
| `Ptr` | 分配身份令牌的 DSL 包装类型 | 由分配操作产生 |
| `MemRefType` | `pl.MemRef` 绑定的类型 | — |
| `TileView` | tile 的有效形状与 stride 视图 | 由 `pl.TileView(...)` 构造 |
| `TileLayout` | tile 布局 | `row_major`、`col_major`、`none_box` |
| `CompactMode` | 非满 tile 的压缩方式 | `normal`、`null` |
| `PipeType` | 指令走哪条硬件流水 | `MTE1`、`MTE2`、`MTE3`、`M`、`V`、`S` |

`PipeType` 正是 [L0 指令级 trace](../performance/04-incore.md) 汇报所用的词汇，所以即便你从不写它，也值得认得。

## 异步预取句柄

GM→L2 预取面暴露三种句柄类型。它们由预取 API 产出、由其完成调用消费；kernel 持有它们，而不构造它们。

| 句柄 | 是什么 |
| ---- | ------ |
| `PrefetchAsyncContext` | 一个 GM→L2 预取上下文 |
| `AsyncEvent` | 一个在飞的预取完成事件 |
| `AsyncSession` | `AsyncEvent` 所属的会话 |

## 分布式

每个 `pld.tensor.*` 集合通信运行在哪条通路上。"InCore composite" 与 "HOST builtin"
对每个集合通信都存在；"CHIP/L2 builtin" 是第三条通路，只有两个 all-to-all-v 算子才有。
dtype 与 reduce 算子的细节以脚注列在表格下方，没有塞进单元格里。

| 集合通信 | InCore composite | HOST builtin | CHIP/L2 builtin | Reduce 算子 | Dtype |
| -------- | ---------------- | ------------ | --------------- | ----------- | ----- |
| `allreduce` | ✅ | ✅ | — | Sum/Max/Min/Prod [1] | FP16/FP32 [1] |
| `allgather` | ✅ | ✅ | — | — | 任意/FP32 [2] |
| `reduce_scatter` | ✅ | ✅ | — | Sum/Max/Min/Prod [3] | 任意/FP32 [2] |
| `broadcast` | ✅ | ✅ | — | — | 任意/FP32 [2] |
| `all_to_all` | ✅ | ✅ | — | — | 任意/FP32 [2] |
| `all_to_all_v` | ✅ [4] | ✅ | ✅ | — | FP32、INT8 |
| `indexed_all_to_all_v` | — | — | ✅ [5] | — | FP32、INT8 |
| `barrier` | ✅ | ✅ | — | — | 仅 INT32 信号 |

1. `allreduce`：两条通路的 `mesh` 模式与 InCore 的 `ring` 模式都支持全部四种
   `ReduceOp`，dtype 为 FP16/FP32；HOST builtin 的 `ring` 模式仅支持 `Sum` + FP32。
2. "任意/FP32" 表示：InCore 上任意 GM dtype，HOST builtin 上仅 FP32。
3. `reduce_scatter`：InCore 上支持全部四种 `ReduceOp`；HOST builtin 上仅 `Sum`。
4. `all_to_all_v` 的 InCore composite 通路本身无限制（它基于 `pld.tile.put`
   构建，而后者没有自己的 dtype 限制）——FP32/INT8 的限制只适用于 HOST/CHIP
   builtin 通路。
5. `indexed_all_to_all_v` 的 CHIP/L2 通路仅支持 A5 URMA。

依据 [Distributed Operators](../../dev/distributed_ops.md) 与
[分布式章节](../distributed/01-collectives.md)的 "InCore 手写 vs Host 级别内置集合通信"
一节核实；把这张表当作摘要指引，而不是这两页的替代品。确切的 API 签名见
[算子目录](../ops/01-catalog.md)。

## 参见

- [编译](../execution/00-compile.md) —— `backend_type`、`memory_planner` 与 `verification_level` 在哪里设置。
- [算子](../ops/index.md) —— 算子面，以及该用哪个命名空间。
- [FAQ](01-faq.md) —— 已知限制与迁移。
