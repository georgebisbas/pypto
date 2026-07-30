# 性能方法论

单节点和分布式两条轨道共享下面的测量循环——先选择怀疑瓶颈所在层级对应的
工具，再确认改动确实改变了那个关键数字。

## 决策树

```text
性能低于预期
├─ 1. 编译器是否已提示？ → report/perf_hints.log
├─ 2. 哪个端到端分段？   → benchmark span tree
├─ 3. 单个 kernel 内部？  → in-core msprof op-simulator
├─ 4. 资源饱和或浪费？   → memory map HTML, scope stats
└─ 5. 调度被序列化？     → dependency graph, L2 swimlane
```

## 快速开始

`pypto-lib` golden 框架提供一个环境变量驱动的快速开始方式
（`PYPTO_BENCH=1 python my_kernel.py`）——该框架的环境变量和默认值
见 `pypto-lib` 自己的文档，本仓库中未定义它们。本仓库自身的基准测试
契约是下面的程序化 API。

## 程序化 Benchmark API

```python
from pypto.runtime import benchmark

compiled = ir.compile(MyProgram)
stats = benchmark(compiled, args=(x, out), rounds=100, warmup=3)

print(f"median: {stats.device_us_median:.1f} us")
```

### API 签名

```python
benchmark(
    compiled,              # CompiledProgram 或 DistributedCompiledProgram
    args,
    *,
    rounds: int = 100,
    warmup: int = 3,
    platform: str | None = None,   # 仅 L2
    device_id: int | None = None,  # 仅 L2
    config: RunConfig | None = None,
    persistent: bool = False,               # 跨分发保留 CommDomain（L3）
    reset_persistent_windows: bool | None = None,  # 复用前是否清零保留的 window
) -> BenchmarkStats
```

### BenchmarkStats 字段

| 字段 | 类型 | 描述 |
| ---- | ---- | ---- |
| `device_wall_us` | `list[float]` | 每轮 NPU 端设备墙钟（µs）。 |
| `host_wall_us` | `list[float]` | 每轮 host 端墙钟（µs）。 |
| `rounds` | `int` | 测量轮次（不含预热）。 |
| `warmup` | `int` | 舍弃的前置轮次。 |
| `all_zero_device` | `bool` | 所有采样为 0 时为 True。 |

### 聚合值

| 属性 | 描述 |
| ---- | ---- |
| `stats.device_us_median` | 中位数（µs）。 |
| `stats.device_us_min` | 最小值（µs）。 |
| `stats.device_us_max` | 最大值（µs）。 |
| `stats.device_us_mean` | 算术平均（µs）。 |
| `stats.device_us_stdev` | 标准差（µs）。 |

### Span Tree 渲染

```python
stats.print_tree()
stats.print_mean_tree()
```

### 完整示例

```python
import torch
import pypto.language as pl
from pypto import ir
from pypto.runtime import benchmark

ROWS = COLS = 128

@pl.program
class MatAdd:
    @pl.function(type=pl.FunctionType.InCore)
    def add_kernel(self, a, b, c):
        ta = pl.load(a, [0, 0], [ROWS, COLS])
        tb = pl.load(b, [0, 0], [ROWS, COLS])
        return pl.store(pl.add(ta, tb), [0, 0], c)

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch(self, a, b, c):
        return self.add_kernel(a, b, c)

compiled = ir.compile(MatAdd, platform="a2a3sim")
a = torch.full((ROWS, COLS), 2.0)
b = torch.full((ROWS, COLS), 3.0)
c = torch.zeros((ROWS, COLS))

stats = benchmark(compiled, args=(a, b, c), rounds=20, warmup=5, platform="a2a3sim")
print(f"median: {stats.device_us_median:.1f} us")
```

## 重要注意事项

> **`SIMPLER_HOST_STRACE` 必须编译到运行时中。**
> **`*sim` 平台设备墙钟为 0。**
> **L3 需要共享内存 IO tensor。**

## 相关链接

- [01-single-node](01-single-node.md) — 单节点性能技术
- [02-distributed](02-distributed.md) — 分布式性能和总线带宽
- [03-cases](03-cases.md) — 端到端工作示例
