# 分布式性能

分布式（L3）程序在单节点性能的基础上，还要处理跨 rank 的问题——总线带宽、
集合通信选择、启动偏差。

## L3 分布式基准测试

分布式程序使用相同的 `benchmark()` API，但在计时和准备方面有重要差异。

### 准备

```python
import torch
from pypto.runtime import benchmark

host_x = torch.zeros((4, 1, 256), dtype=torch.float32).share_memory_()
host_out = torch.zeros_like(host_x).share_memory_()

stats = benchmark(compiled, (host_x, host_out), rounds=100, warmup=3)
```

### L3 指标

| 指标 | `per_round("...")` 键 | 描述 |
| ---- | --------------------- | ---- |
| device | `"device"` | 每轮跨 rank 最大设备墙钟（µs）。 |
| host | `"host"` | 每轮跨 rank 最大 host 墙钟（µs）。 |
| union | `"union"` | 跨 rank host 时间线并集（µs）。捕获重叠和启动偏差。 |

```python
ranks = stats.per_rank("device")  # {pid: [round0_us, ...]}
device = stats.per_round("device")
union  = stats.per_round("union")
```

## 理解总线带宽

总线带宽（`busbw`）是评估集合通信性能的标准指标，源自 nccl-tests 基准测试套件。

### 公式

```text
algbw = data_size / time
busbw = algbw x correction_factor
```

### 修正因子

| 操作 | 修正因子 | 说明 |
| ---- | -------- | ---- |
| AllReduce | `2(n-1)/n` | 双向流量（reduce + broadcast），大 n 时趋近 2。 |
| AllGather | `(n-1)/n` | 每个 rank 接收 n-1 个 chunk。 |
| ReduceScatter | `(n-1)/n` | 每个 rank 发送 n-1 个 chunk。 |
| All-to-All | `(n-1)/n` | 个性化交换。 |
| Broadcast | `1` | Root 发送给 n-1 个 rank。 |

### 计算示例

用于说明公式的假设数值（依据 v6 §8.1 第 5 项，真实测量数据会在积累后回填）——
8 个 rank，1 GB AllReduce 在 100 ms 内完成：

```text
algbw = 1 GB / 0.100 s = 10 GB/s
busbw = 10 x 2(8-1)/8 = 17.5 GB/s
```

## 集合通信算法选择

### Mesh vs Ring 权衡

| 方面 | Mesh | Ring |
| ---- | ---- | ---- |
| 每步远程流量 | O(N) | O(N/P) |
| 屏障轮次 | 1 | 2(P-1) |
| NR 支持 | `pl.dynamic("NR")` | 编译时静态 |
| 最适合 | 小消息 | 大消息（>16 KiB） |

### 通信与计算重叠

PyPTO 的信号模型通过流水线化的 notify/wait 模式支持通信与计算重叠。

### 跨 Rank 启动偏差

`union` 指标捕获跨 rank 启动偏差。高 `union` 相对 `device` 表示 rank 同步不佳。

## 驻留分片

```python
host_weights = torch.randn(4, 1024, 4096).share_memory_()
with compiled.prepare() as rt:
    stacked = rt.alloc_stacked_tensor(host_weights)
    rt(x, stacked, out)
```

## Ring 大小与预热

通过 `RunConfig` 调整 ring-task window 和 heap 大小，需要同时传给
`prepare()` 和每次派发调用：

```python
from pypto.runtime import RunConfig

compiled = ir.compile(MyRingProgram, platform="a2a3", distributed_config=dc)
ring_config = RunConfig(ring_task_window=256, ring_heap=1024)
worker = compiled.prepare(config=ring_config)
worker(host_x, host_out, config=ring_config)  # 每次派发都传相同的 config
```

`prepare(config=...)` 仅用该配置预热运行时 arena 缓存，使**第一次**派发
跳过约 800ms 的冷启动构建——该配置不会存储在 worker 上。每次派发都必须
携带自己的 `config=`（使用相同的 `ring_task_window` / `ring_heap`），否则
arena 会重建（缓存是单槽位的，交替使用不同大小会导致每次切换都重建）。

`ring_task_window` 必须是 `>= 4` 的 2 的幂（或长度为 4 的 list/tuple，分别
设置 ring 0..3）；`ring_heap`（以字节为单位）必须是 `>= 1024` 的 2 的幂。

## 共享 Worker

```python
with compiled_a.prepare(extra_compiled=[compiled_b]) as rt:
    rt.run(compiled_a, host_x, host_out)
    rt.run(compiled_b, host_x, host_out)
```

准备多个程序会让 worker 进入多程序模式，此时 `rt(*args)` 快捷方式含义
不明确会抛出 `TypeError`——包括主程序在内的每个程序都必须显式通过
`rt.run(...)` 派发。

## 重要注意事项

> **`*sim` 平台设备墙钟为 0。**
> **L3 需要共享内存 IO tensor。**
> **DFX 标志通过 L3 管道传递。**

## 相关链接

- [00-methodology](00-methodology.md) — 测量循环和工具
- [01-single-node](01-single-node.md) — 单节点性能技术
- [03-cases](03-cases.md) — 端到端工作示例
