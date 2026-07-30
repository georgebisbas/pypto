# Performance

PyPTO's performance work follows a **measure → locate → optimize → verify**
loop, split into two tracks:

| Track | Scope | Page |
| ----- | ----- | ---- |
| Shared methodology | Tools and measurement loop (both tracks) | [00-methodology](00-methodology.md) |
| Single-node | Kernel, tile, pipelining, matmul, memory, scheduling | [01-single-node](01-single-node.md) |
| Distributed | Collective cost, ring vs mesh, cross-rank skew, bus bandwidth | [02-distributed](02-distributed.md) |
| Cases | End-to-end worked examples | [03-cases](03-cases.md) |

> **Prerequisites:** [Distributed programming](../distributed/00-model.md) for
> the distributed track; [Getting Started](../00-getting_started.md) for a
> kernel-authoring baseline.

## Tool Matrix

| Tool | Observes | Entry point |
| ---- | -------- | ----------- |
| Compile-time perf hints | Code patterns | `report/perf_hints.log` |
| Benchmark span tree | End-to-end segmentation | `pypto.runtime.benchmark` → `stats.print_mean_tree(spread=...)` |
| In-core msprof | Per-kernel cycles | op-simulator + Insight traces |
| Memory map | On-chip buffers | `pypto.tools.memory_map` → HTML |
| Scope stats | Runtime watermarks | `RunConfig(enable_scope_stats=True)` |
| L2 swimlane / PMU / dep gen | Task scheduling | `RunConfig(enable_l2_swimlane / enable_pmu / enable_dep_gen)` |

## See Also

- [Distributed](../distributed/index.md) — Writing distributed programs
- [Getting Started](../00-getting_started.md) — `ir.compile()` and `RunConfig`
- [Simpler Runtime](https://hw-native-sys.github.io/simpler/) — Scheduler internals
