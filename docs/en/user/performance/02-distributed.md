# Distributed Performance

Distributed (L3) programs add cross-rank concerns — bus bandwidth, collective
choice, start skew — on top of everything in single-node performance.

## L3 Distributed Benchmarking

Distributed programs (`DistributedCompiledProgram`) use the same `benchmark()`
API but have important differences in timing and preparation.

### Preparation

```python
import torch
from pypto.runtime import benchmark

# Shared-memory host tensors — MUST call .share_memory_() before benchmark().
host_x = torch.zeros((4, 1, 256), dtype=torch.float32).share_memory_()
host_out = torch.zeros_like(host_x).share_memory_()

stats = benchmark(compiled, (host_x, host_out), rounds=100, warmup=3)
```

### L3 Metrics

| Metric | `per_round("...")` key | Description |
| ------ | ---------------------- | ----------- |
| device | `"device"` | Per-round max across ranks of each rank's summed dispatch device walls (us). |
| host | `"host"` | Per-round max across ranks of host wall (us). |
| effective | `"effective"` | Per-round max effective window (orch/sched union, us). |
| union | `"union"` | Cross-rank host-timeline union: `max(host-end) - min(host-start)` across all ranks' dispatches (us). Captures overlap and start skew. L3 only. |

```python
# Per-rank breakdown (L3 only).
ranks = stats.per_rank("device")  # {pid: [round0_us, round1_us, ...]}

# Per-round aggregation.
device = stats.per_round("device")    # list[float], length = rounds
union  = stats.per_round("union")     # list[float], L3 only
```

### DFX Flag Availability

`RunConfig(enable_l2_swimlane=True)` enables per-task timing inside the worker
and propagates through L3 orchestration — swimlane traces appear in span-tree
output even for distributed jobs.

## Understanding Bus Bandwidth

Bus bandwidth (`busbw`) is the standard metric for evaluating collective
communication performance, adopted from the nccl-tests benchmark suite.
It corrects for the fact that algorithms like AllReduce transfer different
amounts of data across the interconnect than the algorithmic data size.

### Formulas

```text
algbw = data_size / time              (algorithmic bandwidth)
busbw = algbw x correction_factor     (interconnect-corrected bandwidth)
```

### Correction Factors

| Operation | Correction Factor | Notes |
| --------- | ----------------- | ----- |
| AllReduce | `2(n-1)/n` | Two-way traffic (reduce + broadcast) scaled by rank count. Approaches 2 for large n. |
| AllGather | `(n-1)/n` | Each rank receives n-1 chunks. |
| ReduceScatter | `(n-1)/n` | Each rank sends n-1 chunks. |
| All-to-All | `(n-1)/n` | Personalized exchange — same factor as AllGather/ReduceScatter. |
| Broadcast | `1` | Root sends to n-1 ranks, but the link carries data once. |
| Reduce | `1` | n-1 ranks send to root. |
| Point-to-Point | `1` | Single-link transfer. |

### Worked Example

Hypothetical numbers to illustrate the formula (per v6 §8.1 item 5, real
measured figures are backfilled once data accumulates) — 8 ranks, 1 GB
AllReduce completing in 100 ms:

```text
algbw = 1 GB / 0.100 s = 10 GB/s
busbw = 10 x 2(8-1)/8 = 10 x 14/8 = 17.5 GB/s
```

If your interconnect has a theoretical ceiling of 25 GB/s, then 17.5 GB/s is
**70% utilization** — good for an initial implementation. 90%+ means you're
near the hardware bandwidth ceiling.

## Collective Algorithm Choice

### Mesh vs Ring Trade-off

| Aspect | Mesh | Ring |
| ------ | ---- | ---- |
| Remote traffic per step | O(N) | O(N/P) |
| Barrier rounds | 1 | 2(P-1) |
| Signal shape | `[NR, 1]` | `[2(NR-1), NR]` |
| NR support | `pl.dynamic("NR")` | Compile-time static only |
| Best for | Small messages | Large messages (>16 KiB) |

**Rule of thumb:** Use mesh for small messages and low latency; switch to
ring when bandwidth plateaus.

### Overlapping Communication with Compute

PyPTO's signal model allows overlapping communication and computation
through pipelined notify/wait patterns:

- Use non-blocking `notify` early in a loop iteration
- Schedule compute work between `notify` and `wait`
- The `wait` blocks only when the result is needed

### Cross-Rank Start Skew

The `union` metric captures cross-rank start skew — the difference between
the earliest rank's dispatch start and the latest rank's dispatch end.
High `union` relative to `device` indicates poor rank synchronisation.

## Resident Shards

Use `alloc_stacked_tensor` to keep model weights resident on each device:

```python
host_weights = torch.randn(4, 1024, 4096).share_memory_()
with compiled.prepare() as rt:
    stacked = rt.alloc_stacked_tensor(host_weights)
    rt(x, stacked, out)
    # Weights stay resident — no H2D on subsequent dispatches.
```

## Ring Sizing and Prewarm

Tune ring-task window and heap sizes via `RunConfig`, passed to both
`prepare()` and every dispatch call:

```python
from pypto.runtime import RunConfig

compiled = ir.compile(MyRingProgram, platform="a2a3", distributed_config=dc)
ring_config = RunConfig(ring_task_window=256, ring_heap=1024)
worker = compiled.prepare(config=ring_config)
worker(host_x, host_out, config=ring_config)  # same config on every dispatch
```

`prepare(config=...)` only prewarms the runtime-arena cache with that sizing
so the **first** dispatch skips the ~800ms cold build — the config is not
stored on the worker. Every dispatch must pass its own `config=` with the
same `ring_task_window` / `ring_heap`, or the arena rebuilds (the cache is
single-slot, so alternating sizings rebuilds on every switch).

`ring_task_window` must be a power of 2 `>= 4` (or a 4-element list/tuple
sizing rings 0..3 independently); `ring_heap` (in bytes) must be a power of
2 `>= 1024`.

## Sharing One Worker Across Programs

A single `DistributedWorker` can dispatch multiple compiled programs,
reusing its chip processes and comm setup:

```python
with compiled_a.prepare(extra_compiled=[compiled_b]) as rt:
    rt.run(compiled_a, host_x, host_out)      # dispatch ProgramA
    rt.run(compiled_b, host_x, host_out)      # dispatch ProgramB
```

Preparing more than one program puts the worker in multi-program mode, where
the `rt(*args)` shortcut is ambiguous and raises `TypeError` — dispatch every
program explicitly through `rt.run(...)`, including the primary one.

## Important Caveats

> **`*sim` platforms report 0 for device wall.** Simulator builds emit host
> `[STRACE]` markers but not device-domain spans. Check
> `stats.all_zero_device` to detect this.
>
> **L3 requires shared-memory IO tensors.** Every host `torch.Tensor` passed
> to `benchmark()` for a distributed program must call `.share_memory_()` before
> the call. Forgetting this causes a `TypeError` at dispatch time.
>
> **DFX flags are plumbed through L3.** `RunConfig(enable_l2_swimlane=True)`
> enables per-task timing and swimlane traces propagate through L3 orchestration.

## See Also

- [00-methodology](00-methodology.md) — Measurement loop and tools
- [01-single-node](01-single-node.md) — Single-node performance techniques
- [03-cases](03-cases.md) — End-to-end worked examples
