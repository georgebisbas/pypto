# Feature Matrix

What is supported where, and the types you will meet in dumps but rarely write.

## Backends

PyPTO targets two Ascend generations. `backend_type` selects which at compile time, and it
changes both the passes that run and the ISA that is emitted.

| Feature | `Ascend910` (A2/A3) | `Ascend950` (A5) |
| ------- | ------------------- | ---------------- |
| Platform strings | `a2a3`, `a2a3sim` | `a5`, `a5sim` |
| Mixed cube/vector scopes | Yes | Yes |
| Cross-core ring (`pl.cross_core_slot`) | Yes | Yes |
| GM pipe buffer injection | Yes (`InjectGMPipeBuffer` is backend-gated) | No |
| Multi-hop `tile.cast` expansion | Not needed | `INT32 -> FP16` expands via FP32 ([Precision](../precision/00-workflow.md)) |
| MX matmul family | No | Yes |

Per-operator support is tracked in [PTOAS Operator Status](../../dev/ptoas-op-status.md),
which is generated rather than hand-maintained; the table above covers the feature-level
differences a user hits.

## Memory planners

| Planner | Who allocates | Notes |
| ------- | ------------- | ----- |
| `PYPTO` (default) | `MemoryReuse` + `AllocateMemoryAddr` bake addresses | The [memory map](../tools/02-memory-map.md) can draw the result |
| `DSA_RP` | PyPTO's in-tree capacity-constrained planner | |
| `PTOAS` | ptoas `PlanMemory` owns reuse and addressing | PyPTO's allocation passes are skipped, so pass dumps carry no offsets |

## Verification levels

| Level | Runs |
| ----- | ---- |
| `NONE` | No IR verification |
| `BASIC` (default) | Structural checks after each pass |
| `ROUNDTRIP` | Also reparses each dump — markedly slower, for chasing malformed IR |

## Types you will read but rarely write

These appear in pass dumps and in IR the printer emits. They are part of the public surface
because the printer round-trips through them, not because a kernel author writes them by
hand.

| Type | Meaning | Values |
| ---- | ------- | ------ |
| `Ptr` | DSL wrapper for allocation identity tokens | Produced by allocation operations |
| `MemRefType` | The type of a `pl.MemRef` binding | — |
| `TileView` | A tile's valid shape and stride view | Built by `pl.TileView(...)` |
| `TileLayout` | Tile layout | `row_major`, `col_major`, `none_box` |
| `CompactMode` | Partial-tile compaction | `normal`, `null` |
| `PipeType` | Which hardware pipe an instruction uses | `MTE1`, `MTE2`, `MTE3`, `M`, `V`, `S` |

`PipeType` is the vocabulary the [L0 instruction trace](../performance/04-incore.md) reports
in, so it is worth recognising even though nothing you write names it.

## Asynchronous prefetch handles

The GM→L2 prefetch surface exposes three handle types. They are produced by the prefetch
API and consumed by its completion calls; a kernel holds them, it does not construct them.

| Handle | Meaning |
| ------ | ------- |
| `PrefetchAsyncContext` | A GM→L2 prefetch context |
| `AsyncEvent` | An in-flight prefetch completion event |
| `AsyncSession` | The session an `AsyncEvent` belongs to |

## Distributed

Which rail each `pld.tensor.*` collective runs on. "InCore composite" and "HOST builtin"
exist for every collective below except `indexed_all_to_all_v`; "CHIP/L2 builtin" is a
third rail that only the two all-to-all-v operators have. Dtype and reduce-op caveats are
footnoted rather than crammed into cells — see the notes below the table.

| Collective | InCore composite | HOST builtin | CHIP/L2 builtin | Reduce ops | Dtypes |
| ---------- | ---------------- | ------------ | --------------- | ---------- | ------ |
| `allreduce` | ✅ | ✅ | — | Sum/Max/Min/Prod [1] | FP16/FP32 [1] |
| `allgather` | ✅ | ✅ | — | — | any/FP32 [2] |
| `reduce_scatter` | ✅ | ✅ | — | Sum/Max/Min/Prod [3] | any/FP32 [2] |
| `broadcast` | ✅ | ✅ | — | — | any/FP32 [2] |
| `all_to_all` | ✅ | ✅ | — | — | any/FP32 [2] |
| `all_to_all_v` | ✅ [4] | ✅ | ✅ | — | FP32, INT8 |
| `indexed_all_to_all_v` | — | — | ✅ [5] | — | FP32, INT8 |
| `barrier` | ✅ | ✅ | — | — | INT32 signal only |

1. `allreduce`: both rails' `mesh` mode and InCore's `ring` mode support all four
   `ReduceOp`s with FP16/FP32; the HOST builtin's `ring` mode is `Sum`+FP32 only.
2. "any/FP32" means: any GM dtype on InCore, FP32 only on the HOST builtin.
3. `reduce_scatter`: all four `ReduceOp`s on InCore; `Sum` only on the HOST builtin.
4. `all_to_all_v`'s InCore composite rail has no FP32/INT8 allowlist, but input and
   target dtypes must match (it's built from `pld.tile.put`) — the FP32/INT8 limit
   applies to the HOST/CHIP builtin rails only.
5. `indexed_all_to_all_v`'s CHIP/L2 rail is A5 URMA only.

Verified against [Distributed Operators](../../dev/distributed_ops.md) and
[InCore vs Host-Level Collectives](../distributed/01-collectives.md#incore-vs-host-level-collectives);
treat this table as a summary pointer, not a replacement for either page. See the
[catalog](../ops/01-catalog.md) for the exact API signatures these cells summarize.

## See Also

- [Compiling](../execution/00-compile.md) — where `backend_type`, `memory_planner` and `verification_level` are set.
- [Operations](../ops/index.md) — the operator surface, and which namespace to use.
- [FAQ](01-faq.md) — known limitations and migrations.
