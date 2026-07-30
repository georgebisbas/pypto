# Collectives

This page covers the five built-in collectives and when to use each algorithm.
All collectives are **synchronous** across ranks — every rank must call the same
collective with identically shaped signal tensors, or the program hangs or
silently corrupts data.

## AllReduce

Every rank contributes its local data; every rank receives the summed
result.

```python
# Host orchestrator — simplest form (compiler synthesizes signal).
data = pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)  # mesh mode, in-place

# InCore kernel — explicit signal.
data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode="mesh")
data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode="ring")
```

### Mesh Mode

- O(N) remote traffic per step — every rank reads every peer
- One global barrier per call (AtomicAdd/Ge on `[NR, 1]` signal)
- Works with `pl.dynamic("NR")`
- Best for small messages and low latency

### Ring Mode

- 2(P-1) steps: reduce-scatter + allgather
- O(N/P) remote traffic per step — each rank reads one neighbour
- Signal shape: `[2 × (NR − 1), NR]`
- Requires compile-time-known NR — use a factory function pattern
- Best for large messages (>16 KiB) and high bandwidth

| Aspect | Mesh | Ring |
| ------ | ---- | ---- |
| Remote traffic per step | O(N) — every rank reads every peer | O(N/P) — each rank reads one neighbour |
| Barrier rounds | 1 (global AtomicAdd/Ge) | 2(P-1) — reduce-scatter + allgather phases |
| Signal shape | `[NR, 1]` | `[2 × (NR − 1), NR]` |
| Best for | Small messages, low latency | Large messages, high bandwidth |

**Rule of thumb:** Use the default `mode="mesh"`. Switch to `mode="ring"` when
your payload exceeds ~16 KiB and you see mesh bandwidth plateau.

The host orchestrator form (`signal` omitted) is syntactic sugar — the compiler
synthesizes a signal of `[world_size(), 1]` (mesh only).

### Mutation

`target: InOut` — data is both read (as the reduction input) and written (as
the reduced result). All ranks must pass identically shaped `target` tensors.

### Supported ReduceOp

All four — `Sum`, `Max`, `Min`, `Prod` — on both the InCore composite and the
HOST builtin path. `target`'s dtype must be `FP16` or `FP32`; this is a hard
compile-time check, not just a storage-width constraint. Every rank must
agree on the same `ReduceOp` and `mode`, on top of the identically-shaped
signal tensors required of every collective.

## Barrier

Cross-rank barrier — blocks until all ranks arrive.

```python
# signal: pld.DistributedTensor[[NR, 1], pl.INT32], freshly allocated.
signal = pld.tensor.barrier(signal)
```

Uses `Set(1)` + `Ge(1)` on the signal. Single-shot; allocate a fresh buffer
before the next barrier.

## Broadcast

Broadcast root rank's data to all ranks.

```python
# Root stages data before the call.
if my_rank == ROOT_RANK:
    data = pl.store(local, [0, 0], data)
data = pld.tensor.broadcast(data, signal, root=ROOT_RANK)
# Every rank now holds root's data in data[0, 0:SIZE].
```

Root must stage data before the call; non-root slots are ignored on input.
After the call, every rank holds root's data.

## AllGather

Push-based all-gather — every rank pushes its local chunk, every rank receives
the full gathered matrix.

```python
# Stage buffer: this rank's [1, SIZE] chunk (push source).
stage_buf = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
stage = pld.window(stage_buf, [1, SIZE], dtype=pl.FP32)
stage = pl.store(local_input, [0, 0], stage)

# Result buffer: gathered [NR, SIZE] (push target).
data_buf = pld.alloc_window_buffer(NR * SIZE * pl.FP32.get_byte())
data = pld.window(data_buf, [NR, SIZE], dtype=pl.FP32)
sig_buf = pld.alloc_window_buffer(NR * pl.INT32.get_byte())
sig = pld.window(sig_buf, [NR], dtype=pl.INT32)

data = pld.tensor.allgather(stage, data, sig)
# data[src, :] now holds rank src's chunk for every src.
```

`local_data` and `target` **must be different** window buffers. The stage buffer
is the per-rank push source; the target buffer receives the gathered `[NR, SIZE]`
result.

## ReduceScatter

Reduce-scatter: every rank stages all NR chunks, receives its own reduced chunk.

```python
# Signal for the barrier (1-D for host builtins).
sig_buf = pld.alloc_window_buffer(NR * pl.INT32.get_byte())
sig = pld.window(sig_buf, [NR], dtype=pl.INT32)

# Stage all NR chunks into data[NR, SIZE].
for j in pl.range(nranks):
    data = pl.store(chunk_j, [j, 0], data)
data = pld.tensor.reduce_scatter(data, sig, op=pld.ReduceOp.Sum)
# data[my_rank, 0:SIZE] holds this rank's reduced chunk.
```

## AllToAll

Personalized all-to-all exchange — every rank sends a distinct chunk to every
peer and receives a distinct chunk from every peer.

```python
# Stage buffer: push source, [NR, SIZE] with per-destination chunks.
stage_buf = pld.alloc_window_buffer(NR * SIZE * pl.FP32.get_byte())
stage = pld.window(stage_buf, [NR, SIZE], dtype=pl.FP32)
for dest in pl.range(nranks):
    stage = pl.store(chunk_for_dest, [dest, 0], stage)

# Result buffer: push target, [NR, SIZE].
data_buf = pld.alloc_window_buffer(NR * SIZE * pl.FP32.get_byte())
data = pld.window(data_buf, [NR, SIZE], dtype=pl.FP32)
sig_buf = pld.alloc_window_buffer(NR * pl.INT32.get_byte())
sig = pld.window(sig_buf, [NR], dtype=pl.INT32)

data = pld.tensor.all_to_all(stage, data, sig)
# data[src, :] holds the chunk received from rank src.
```

`input` and `target` must be **separate** window buffers.

## InCore vs Host-Level Collectives

PyPTO has three ways to run a collective — pick based on where your code
runs and whether you need `mode="ring"`:

| Aspect | InCore Hand-Rolled | InCore Composite | HOST Builtin |
| ------ | ------------------ | ---------------- | ------------ |
| **Where** | `@pl.function(type=InCore)` | `@pl.function(type=InCore)` | `@pl.function(level=HOST, role=Orchestrator)` |
| **How** | Manual `notify`/`wait` + `remote_load` loops | `pld.tensor.allreduce(data, sig, ...)` called directly | `pld.tensor.allreduce(data, [sig,] ...)` called directly |
| **Lowering** | You write the primitives | `LowerCompositeOps` | `LowerHostTensorCollectives` |
| **Modes** | Whatever you implement | `mesh` and `ring` | `mesh` only |
| **Signal shape** | Whatever you allocate | `[nranks, 1]` for mesh (rank count may be dynamic); `[2×(NR−1), NR]` for ring (`NR` must be a compile-time constant) | Rank-1 `[world_size]` or rank-2 `[world_size, 1]` — the compiler-synthesized signal is rank-2 |
| **When** | Learning, custom protocols | Need `ring` mode, or already inside an InCore kernel | Day-to-day host-orchestrated collectives |

Prefer HOST builtins for day-to-day host-orchestrated code — they handle
signal allocation, barrier orchestration, and chunking automatically. Reach
for the InCore composite specifically when you need `mode="ring"`, since the
HOST builtin path only lowers `mesh`.

## See Also

- [00-model](00-model.md) — Quickstart and model vocabulary
- [02-primitives](02-primitives.md) — The substrate beneath the collectives
- [04-debugging](04-debugging.md) — Common failure patterns
