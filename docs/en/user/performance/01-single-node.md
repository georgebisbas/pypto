# Single-Node Performance

Every technique below states **when it applies, what it costs, how to
enable it, and how to verify it took effect**.

## Partitioning and Parallelism

### `pl.split(SplitMode)`

Split cross-core data transfer geometrically within a `CORE_GROUP` region —
not a standalone call; it's passed into `pl.at(..., optimizations=[...])`.

- **When:** A `CORE_GROUP` region's data needs partitioning across its cores
- **Cost:** Requires split-compatible operations
- **How:** `pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.UP_DOWN)])` — modes are `NONE`, `UP_DOWN` (height halved), `LEFT_RIGHT` (width halved)
- **Verify:** Compare against non-split baseline via benchmark

### `pl.split_aiv`

Split computation across AIV (vector) cores specifically.

- **When:** Vector-heavy workloads benefit from AIV parallelism
- **Cost:** AIV-only — incompatible with AIC units
- **Verify:** Check that AIV cores are utilized in the memory map

### `pl.spmd(N)`

Single-program-multiple-data parallelism — launch N copies of the same kernel
with different data shards.

- **When:** Embarrassingly parallel workloads
- **Cost:** N x memory footprint; workspace serialization for >1
- **How:** `with pl.spmd(n):` or `for i in pl.spmd(n):`
- **Verify:** Check benchmark scaling — near-linear speedup expected

### `pl.cluster()`

Group co-scheduled AIC (Cube) and AIV (Vector) kernels sharing physical
cluster resources.

- **When:** Workloads benefit from concurrent AIC + AIV execution
- **Cost:** Requires cluster-compatible kernel pairs
- **Verify:** L2 swimlane shows concurrent AIC/AIV execution

### `pl.at(level=)`

Mark a region of code for execution at a specific point in the hierarchy
(`pl.Level.AIV`, `.AIC`, `.CORE_GROUP`, `.CHIP_DIE`, `.CHIP`, `.HOST`, ...) —
this is where the region *executes*, not a memory tier.

- **When:** Coordinating co-scheduled AIC/AIV work, or applying `pl.split`
- **Cost:** The chosen level determines which `optimizations=` entries apply
- **Verify:** L2 swimlane shows the region executing at the specified level

## Pipelining and Unrolling

### `pl.pipeline`

Software pipeline a loop body to overlap compute across iterations.

- **When:** Loop bodies with independent iterations
- **Cost:** Increased register and buffer pressure
- **How:** Wrap the loop body with `pl.pipeline`
- **Verify:** Benchmark shows reduced per-iteration latency

### `pl.unroll`

Fully unroll a compile-time-known loop.

- **When:** Small loop trip counts known at compile time
- **Cost:** Larger binary — code size grows with unroll factor
- **Verify:** Inspect unrolled code in the compiled artifact

### Cross-Core Software Pipelining

Use `pl.cross_core_slot(slot_num=)` to pipeline data across AICore
computing units.

- **When:** Ring-buffer or producer-consumer patterns across cores
- **Cost:** Ring depth determines buffer size
- **How:** Set `slot_num` to the ring depth
- **Verify:** Swimlane shows pipelined execution with no idle gaps

## Matmul Path

### AutoTileMatmulL0

The auto-tiling pass selects L0 matmul tile sizes.

- **When:** All matmul operations (enabled by default)
- **Cost:** Compile-time analysis — no runtime overhead
- **Verify:** IR dump shows tiled matmul dimensions

### `enable_pypto_l0c_double_buffer`

Enable double-buffering for L0C output buffer.

- **When:** Matmul-bound workloads where output write-back is a bottleneck
- **Cost:** 2x L0C buffer allocation
- **How:** `ir.compile(..., enable_pypto_l0c_double_buffer=True)`
- **Verify:** Benchmark shows reduced sched time in span tree

### `a_trans`/`b_trans`

Transpose matmul operands in-place.

- **When:** Memory layout doesn't match the matmul instruction's expected order
- **Cost:** Potential additional transpose overhead
- **Verify:** Benchmark both transposed and non-transposed

### Split-K + Atomic Add

Split the K-dimension of a matmul across multiple units with atomic
accumulation.

- **When:** Very large K dimensions exceed single-unit capacity
- **Cost:** Atomic add introduces non-determinism (order of accumulation varies)
- **Verify:** Accuracy check against golden; benchmark for throughput

## Memory

### `target_memory`

Select the on-chip memory space for a tile (`MemorySpace.DDR` for off-chip,
or `.Vec` / `.Mat` / `.Left` / `.Right` / `.Acc` for on-chip buffers). Not
every op accepts every space: `pl.load` (DDR to on-chip) only accepts `.Vec`
or `.Mat` — it raises `ValueError` for the others. To land data in `.Left` /
`.Right`, load into `.Vec` / `.Mat` first, then use `pl.move` to relocate it.
`.Acc` is a matmul-accumulate output space, not a `load`/`move` destination.

- **When:** Data placement optimization
- **Cost:** On-chip spaces are faster but much smaller than DDR
- **How:** `pl.load(a, [0, 0], [rows, cols], target_memory=pl.MemorySpace.Vec)`
- **Verify:** Memory map confirms allocation at the specified space

### MemoryReuse vs `memory_planner=PTOAS`

Memory planning strategies for allocating buffers.

- **When:** MemoryReuse (default) — reuse buffers when live ranges don't overlap
- **When:** `memory_planner=PTOAS` — delegate to the PTOAS memory planner
- **Cost:** PTOAS planner is more aggressive but may increase compile time
- **Verify:** Memory map shows buffer allocation and reuse

### Persistent L3

Keep L3 buffers resident across dispatches.

- **When:** Multi-dispatch workloads with large working sets
- **Cost:** Reduced L3 availability for other uses
- **Verify:** Scope stats show persistent allocation

## Scheduling

### `predicate=`

Skip tasks dynamically at the dispatch point.

- **When:** Conditional execution paths
- **Cost:** Negligible — predicate check at dispatch
- **Verify:** Dependency graph shows skipped edges

### `no_dep`

Drop automatic dependency inference for a call site.

- **When:** Two ops appear to overlap but are actually independent
- **Cost:** Incorrect use causes race conditions
- **Verify:** Dependency graph confirms no edge between the ops

### `allow_early_resolve`

Allow the scheduler to resolve a task early.

- **When:** Output is produced before the task's full completion
- **Cost:** Weaker ordering guarantees
- **Verify:** Swimlane trace confirms early resolution

### `manual_scope`

Turn off automatic dependency tracking for a region.

- **When:** Manual control over every dependency edge
- **How:** `with pl.scope(mode=pl.ScopeMode.MANUAL):` or `with pl.manual_scope():`
- **Cost:** All edges must be declared explicitly via `deps=`
- **Verify:** Dependency graph matches the declared edges exactly

### `task_dummy`

Insert a no-op task as a fan-in point.

- **When:** Multiple producers feed a single consumer with no data dependency
- **How:** `barrier = pl.system.task_dummy(deps=[task_a, task_b])` — `deps` is required and keyword-only
- **Verify:** Dependency graph shows the dummy task as a convergence node

### Ring Sizing

Tune ring-task window and heap sizes for the L2 scheduler.

- **When:** Task-heavy workloads benefit from larger ring buffers
- **How:** `RunConfig(ring_task_window=N, ring_heap=M)`
- **Verify:** Scope stats show ring buffer utilization

## Data Residency

### `DeviceTensor`

Keep tensors resident on the device across dispatches.

- **When:** Weights, lookup tables, or other reusable data
- **Cost:** Reduced device memory for other allocations
- **How:** `rt.alloc_tensor(shape, dtype, init=host_data)`
- **Verify:** H2D/D2H spans absent from second dispatch onward

## See Also

- [00-methodology](00-methodology.md) — Measurement loop and tools
- [02-distributed](02-distributed.md) — Distributed performance techniques
- [03-cases](03-cases.md) — End-to-end worked examples
