# 集合通信

本页介绍五种内置集合通信及算法选择。所有集合通信在各 rank 间**同步执行**——
每个 rank 必须以相同形状的 signal tensor 调用同一集合通信，否则程序会挂起
或静默数据损坏。

## AllReduce

每个 rank 提交其本地数据；每个 rank 接收求和结果。

```python
# Host 编排器——最简形式（编译器合成 signal）。
data = pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)  # mesh 模式，就地

# InCore kernel——显式 signal。
data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode="mesh")
data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode="ring")
```

### Mesh 模式

- 每步 O(N) 远程流量——每个 rank 读取所有对端
- 每次调用一个全局屏障（AtomicAdd/Ge 在 `[NR, 1]` signal 上）
- 支持 `pl.dynamic("NR")`
- 最适合小消息和低延迟

### Ring 模式

- 2(P-1) 步：reduce-scatter + allgather
- 每步 O(N/P) 远程流量——每个 rank 读取一个邻居
- Signal 形状：`[2 × (NR − 1), NR]`
- 要求编译时已知 NR——使用工厂函数模式
- 最适合大消息（>16 KiB）和高带宽

| 方面 | Mesh | Ring |
| ---- | ---- | ---- |
| 每步远程流量 | O(N) | O(N/P) |
| 屏障轮次 | 1 | 2(P-1) |
| Signal 形状 | `[NR, 1]` | `[2 × (NR − 1), NR]` |
| 最适合 | 小消息，低延迟 | 大消息，高带宽 |

**经验法则：** 默认使用 `mode="mesh"`。当负载超过约 16 KiB 且 mesh 带宽达到平台期时
切换到 `mode="ring"`。

Host 编排器形式（省略 `signal`）是语法糖——编译器合成 `[world_size(), 1]` 的
signal（仅限 mesh）。

### 变更

`target: InOut` — 数据既被读取（作为规约输入）又被写入（作为规约结果）。所有 rank
必须传入形状相同的 `target` tensor。

### 支持的 ReduceOp

全部四种——`Sum`、`Max`、`Min`、`Prod`——InCore 组合调用和 Host 内置路径均
支持。`target` 的 dtype 必须是 `FP16` 或 `FP32`；这是编译期硬性检查，而非
仅存储位宽的限制。除了本页开头要求的形状相同的 signal tensor 外，所有
rank 还必须使用相同的 `ReduceOp` 和 `mode`。

## Barrier

跨 rank 屏障——阻塞直到所有 rank 到达。

```python
# signal: pld.DistributedTensor[[NR, 1], pl.INT32]，新分配的。
signal = pld.tensor.barrier(signal)
```

在 signal 上使用 `Set(1)` + `Ge(1)`。单次使用；下一次 barrier 前需分配新
buffer。

## Broadcast

将 root rank 的数据广播到所有 rank。

```python
if my_rank == ROOT_RANK:
    data = pl.store(local, [0, 0], data)
data = pld.tensor.broadcast(data, signal, root=ROOT_RANK)
```

## AllGather

推式 allgather。

```python
stage_buf = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
stage = pld.window(stage_buf, [1, SIZE], dtype=pl.FP32)
stage = pl.store(local_input, [0, 0], stage)

data_buf = pld.alloc_window_buffer(NR * SIZE * pl.FP32.get_byte())
data = pld.window(data_buf, [NR, SIZE], dtype=pl.FP32)
sig_buf = pld.alloc_window_buffer(NR * pl.INT32.get_byte())
sig = pld.window(sig_buf, [NR], dtype=pl.INT32)

data = pld.tensor.allgather(stage, data, sig)
```

`local_data` 和 `target` **必须是不同的** window buffer。

## ReduceScatter

```python
sig_buf = pld.alloc_window_buffer(NR * pl.INT32.get_byte())
sig = pld.window(sig_buf, [NR], dtype=pl.INT32)

for j in pl.range(nranks):
    data = pl.store(chunk_j, [j, 0], data)
data = pld.tensor.reduce_scatter(data, sig, op=pld.ReduceOp.Sum)
```

## AllToAll

个性化 all-to-all 交换。

```python
stage_buf = pld.alloc_window_buffer(NR * SIZE * pl.FP32.get_byte())
stage = pld.window(stage_buf, [NR, SIZE], dtype=pl.FP32)
for dest in pl.range(nranks):
    stage = pl.store(chunk_for_dest, [dest, 0], stage)

data_buf = pld.alloc_window_buffer(NR * SIZE * pl.FP32.get_byte())
data = pld.window(data_buf, [NR, SIZE], dtype=pl.FP32)
sig_buf = pld.alloc_window_buffer(NR * pl.INT32.get_byte())
sig = pld.window(sig_buf, [NR], dtype=pl.INT32)

data = pld.tensor.all_to_all(stage, data, sig)
```

`input` 和 `target` 必须是**不同的** window buffer。

## InCore 手写 vs Host 级别内置集合通信

PyPTO 有三种方式运行集合通信——根据代码运行的位置以及是否需要
`mode="ring"` 来选择：

| 方面 | InCore 手写 | InCore 组合调用 | Host 级别内置 |
| ---- | ----------- | --------------- | ------------- |
| **位置** | `@pl.function(type=InCore)` | `@pl.function(type=InCore)` | `@pl.function(level=HOST, role=Orchestrator)` |
| **实现** | 手写 `notify`/`wait` + `remote_load` 循环 | 直接调用 `pld.tensor.allreduce(data, sig, ...)` | 直接调用 `pld.tensor.allreduce(data, [sig,] ...)` |
| **Lowering** | 自行实现原语 | `LowerCompositeOps` | `LowerHostTensorCollectives` |
| **支持的模式** | 取决于自己的实现 | `mesh` 和 `ring` | 仅 `mesh` |
| **Signal 形状** | 取决于自己的分配 | mesh 为 `[nranks, 1]`（rank 数量可为动态）；ring 为 `[2×(NR−1), NR]`（`NR` 必须是编译期常量） | 一维 `[world_size]` 或二维 `[world_size, 1]`——编译器合成的 signal 为二维 |
| **适用场景** | 学习、自定义协议 | 需要 `ring` 模式，或已身处 InCore kernel 内部 | 日常的 host 编排集合通信 |

日常 host 编排代码优先使用 Host 级别内置——它们自动处理 signal 分配、
屏障编排和分块。当需要 `mode="ring"` 时改用 InCore 组合调用，因为 Host
内置路径只 lowering `mesh`。

## 相关链接

- [00-model](00-model.md) — 快速开始和模型词汇
- [02-primitives](02-primitives.md) — 集合通信的底层基础
- [04-debugging](04-debugging.md) — 常见故障模式
