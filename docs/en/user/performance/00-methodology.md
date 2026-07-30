# Performance Methodology

The single-node and distributed tracks share the measurement loop below —
pick the tool for the layer you suspect, then confirm the fix moved the
number that mattered.

## Decision Tree

```text
Performance below expectation
├─ 1. Did the compiler already hint? → report/perf_hints.log (PH001 TileInnermostDimGranularity, ...)
├─ 2. Which end-to-end segment?     → benchmark span tree (print_mean_tree), host vs device domain
├─ 3. Inside a single kernel?       → in-core msprof op-simulator, instruction / cycle level
├─ 4. Resources saturated or wasted? → memory map HTML, scope stats (heap / task_window / tensormap)
└─ 5. Is scheduling serialized?     → dependency graph (enable_dep_gen), L2 swimlane
```

## Quick Start

The `pypto-lib` golden harness offers an environment-variable-driven quick
start (`PYPTO_BENCH=1 python my_kernel.py`) — see `pypto-lib`'s own
documentation for that harness's env vars and defaults; they are not defined
in this repository. This repo's own benchmarking contract is the
programmatic API below.

## Programmatic Benchmark API

```python
from pypto.runtime import benchmark

compiled = ir.compile(MyProgram)
stats = benchmark(compiled, args=(x, out), rounds=100, warmup=3)

print(f"median: {stats.device_us_median:.1f} us")
print(f"min:    {stats.device_us_min:.1f} us")
print(f"max:    {stats.device_us_max:.1f} us")
print(f"mean:   {stats.device_us_mean:.1f} us")
print(f"stdev:  {stats.device_us_stdev:.1f} us")
```

### API Signature

```python
benchmark(
    compiled,              # CompiledProgram (L2) or DistributedCompiledProgram (L3)
    args,                  # tuple of dispatch arguments
    *,                     # all remaining args are keyword-only
    rounds: int = 100,
    warmup: int = 3,
    platform: str | None = None,   # target platform (L2 only)
    device_id: int | None = None,  # NPU device index (L2 only)
    config: RunConfig | None = None,
    persistent: bool = False,               # retain CommDomains across dispatches (L3)
    reset_persistent_windows: bool | None = None,  # zero retained windows before reuse
) -> BenchmarkStats
```

### BenchmarkStats Fields

| Field | Type | Description |
| ----- | ---- | ----------- |
| `device_wall_us` | `list[float]` | Per-round on-NPU device wall (us). L2: one per launch. L3: per-round max across ranks. |
| `host_wall_us` | `list[float]` | Per-round host wall (us). Includes arg coercion and H2D overhead. |
| `rounds` | `int` | Number of measured launches (warmup excluded). |
| `warmup` | `int` | Number of leading launches discarded. |
| `all_zero_device` | `bool` | `True` when every `device_wall_us` is `0` — typical on `*sim` platforms. |

### Aggregates

| Property | Description |
| -------- | ----------- |
| `stats.device_us_median` | Median `device_wall_us` (us). |
| `stats.device_us_min` | Minimum `device_wall_us` (us). |
| `stats.device_us_max` | Maximum `device_wall_us` (us). |
| `stats.device_us_mean` | Arithmetic mean (us). |
| `stats.device_us_stdev` | Standard deviation (us). |

### Span Tree Rendering

```python
stats.print_tree()          # render one per-launch span tree to stdout
stats.print_mean_tree()     # render mean-duration tree annotated with +/-stdev
```

Expected output for a single-launch L2 kernel:

```text
launch[0] (pid=1 inv=1 hid=0):
simpler_run                                   10875.1us
|- bind                                          29.6us
|- runner_run                                 10786.3us
|  `- device_wall [dev]                        9981.6us
|     |- orch [dev]                               4.0us
|     |- sched [dev]                            563.4us
|     `- post_orch [dev]                          0.5us
`- validate                                       5.5us
```

The **primary metric** is `device_wall [dev]` — the total on-NPU time.

### Complete Example

```python
import torch
import pypto.language as pl
from pypto import ir
from pypto.runtime import benchmark

ROWS = COLS = 128

@pl.program
class MatAdd:
    @pl.function(type=pl.FunctionType.InCore)
    def add_kernel(
        self,
        a: pl.Tensor[[ROWS, COLS], pl.FP32],
        b: pl.Tensor[[ROWS, COLS], pl.FP32],
        c: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
        ta = pl.load(a, [0, 0], [ROWS, COLS])
        tb = pl.load(b, [0, 0], [ROWS, COLS])
        return pl.store(pl.add(ta, tb), [0, 0], c)

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch(
        self,
        a: pl.Tensor[[ROWS, COLS], pl.FP32],
        b: pl.Tensor[[ROWS, COLS], pl.FP32],
        c: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
        return self.add_kernel(a, b, c)

compiled = ir.compile(MatAdd, platform="a2a3sim")
a = torch.full((ROWS, COLS), 2.0)
b = torch.full((ROWS, COLS), 3.0)
c = torch.zeros((ROWS, COLS))

stats = benchmark(compiled, args=(a, b, c), rounds=20, warmup=5, platform="a2a3sim")

print(f"median: {stats.device_us_median:.1f} us")
if stats.all_zero_device:
    print("[sim] device_wall_us is all-zero — expected on simulator builds")
```

### Interpreting Variance

- **Prefer median over mean** — median is robust to cold-start outliers
- **Increase `warmup`** to absorb one-time setup (5-10 typical on hardware)
- **Check `all_zero_device`** before interpreting results on simulator builds

## Important Caveats

> **`SIMPLER_HOST_STRACE` must be compiled into the runtime.** If the runtime
> was built without this compile-time macro, `benchmark()` raises `RuntimeError`.
>
> **`*sim` platforms report 0 for device wall.** Check `stats.all_zero_device`
> to detect this — if `True`, all `device_wall_us` samples are 0.
>
> **L3 requires shared-memory IO tensors.** Every host `torch.Tensor` passed
> to `benchmark()` for a distributed program must call `.share_memory_()` before
> the call.

## See Also

- [01-single-node](01-single-node.md) — Single-node performance techniques
- [02-distributed](02-distributed.md) — Distributed performance and bus bandwidth
- [03-cases](03-cases.md) — End-to-end worked examples
