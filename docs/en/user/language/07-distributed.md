# Distributed

What `pld` adds on top of `pl` as a language surface: its own types, its own
namespace split, and the rule for choosing it. This page does not duplicate
the seven `pl` pages — `pld` composes `pl`; the types and control flow you
already know apply unchanged. It also does not re-teach the model or the
collectives — see [Distributed Programming](../distributed/index.md) for that.

> **Prerequisites:** [Types](00-types.md), [Functions and Programs](01-functions.md).

## When to use `pld`

`pld` is chosen by **what the data does, not where it lives**: if an operation
crosses ranks, it's `pld`; otherwise it's `pl`. The namespace split mirrors
`pl`'s — `pld.tile.*` inside an InCore function, `pld.tensor.*` in
orchestration on whole tensors, plus a third family, `pld.system.*`, for
rank/signal primitives that have no `pl` equivalent. See
[Choosing a Namespace](../ops/00-dispatch.md) for the full rule and the exact
list of names with a 2-segment short form.

## `DistributedTensor`

`pld.DistributedTensor` is the one new type most `pld` code touches. It
subclasses `pl.Tensor` — same DSL surface, same `memref`/layout/`tensor_view`
subscript slots, and `pl.load`/`pl.store` and friends operate on it
transparently (your own rank's slice). What's different is what it's backed
by: a per-rank slice of a symmetric HCCL **window buffer**, carved by a
`CommDomainScopeStmt`. At the IR level, `ir.DistributedTensorType` is a
precise-`ObjectKind` subclass of `TensorType`, so every cross-rank op
verifier can reject a plain `Tensor` argument outright — passing one by
accident is a compile-time error, not a runtime surprise.

A `DistributedTensor` is produced by `pld.tensor.window(...)`, which views an
allocation `pld.tensor.alloc_window_buffer(...)` made. See
[Window Buffer](../distributed/08-window_buffer.md) for the allocate → view
walkthrough and what "symmetric" actually guarantees (same size and layout
per rank, not a shared address).

## `CommCtx` and rank/world queries

`pld.CommCtx` is a communication-context handle, following the same
two-role pattern as `pl.Ptr`: used as a type annotation
(`ctx: pld.CommCtx`) or as the return value of `pld.get_comm_ctx(...)`
wrapping the underlying IR call. Four functions make up the query surface:

| Call | Takes | Scope |
| ---- | ----- | ----- |
| `pld.world_size()` | nothing | Host-only |
| `pld.get_comm_ctx(dist_tensor)` | a window-bound `DistributedTensor` | Host and InCore |
| `pld.rank(ctx)` | a `CommCtx` | Host and InCore |
| `pld.nranks(ctx)` | a `CommCtx` | Host and InCore |

Note that `get_comm_ctx` takes a `DistributedTensor`, not nothing — it looks
up the context through the window-bound tensor it's handed, it doesn't
conjure one from thin air. `rank`/`nranks` then take that `CommCtx`, not a
tensor.

## The three namespaces and the short form

`pld.tensor.*`, `pld.tile.*` and `pld.system.*` are the three namespaces
(mirroring the `pl.tensor.*`/`pl.tile.*`/`pl.system.*` split). A handful of
names also have a 2-segment short form that dispatches by name, the same way
`pl.add` dispatches to `pl.tile.add` or `pl.tensor.add`: `world_size`,
`alloc_window_buffer`, `window`, `get_comm_ctx`, `rank`, `nranks`,
`remote_load`, and `remote_store` — exactly those eight, no more. Every other
`pld` operation (the collectives, `put`/`get`, the signal primitives) needs
its full 3-segment spelling.

## Enums

`pld` carries five ABI enums: `NotifyOp`, `WaitCmp`, `ReduceOp`, `AtomicType`,
and `CommEngine`. They're an **append-only ABI** — new variants may only be
added at the end, never inserted or renumbered, because their underlying
`int` values are serialised into compiled IR. See the
[Types and Enums table](../distributed/02-primitives.md#types-and-enums) in
the primitives page for each enum's values and what they mean; this page
just names them as part of the type surface, rather than repeating that
table.

## Where to go next

- [Distributed Programming](../distributed/index.md) — the model, the
  collectives, and the 16-step tutorial ladder.
- [Distributed Operators](../../dev/distributed_ops.md) — op semantics, the
  ABI enums' exact numbering, and the barrier-signal protocol.
- [Catalog](../ops/01-catalog.md) — every `pld` operator with its signature
  page.

## See Also

- [Choosing a Namespace](../ops/00-dispatch.md) — the `pl` vs `pld` rule in full.
- [Programming Model](../03-programming-model.md) — `Level.HOST` +
  `Role.Orchestrator`, which marks a distributed program's host orchestrator.
