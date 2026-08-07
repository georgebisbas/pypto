# Barrier: Signals Only

Build a reusable N-rank barrier from `notify`/`wait` — no data moves — then
reveal the builtin `pld.tensor.barrier` that provides the same
synchronization.

> **Prerequisites:** [08-window_buffer](08-window_buffer.md). Two devices.

**Suggested reading order:** 01 → 02 → 03 → **04** → 05 → 06 — this page is step 04.

## The idea

The window buffer's **signal tail** exists for one job: cross-rank
synchronization. Two primitives drive it. `pld.system.notify(...)` increments
a signal cell on a peer; `pld.system.wait(...)` blocks until a signal cell
reaches a threshold. Together they form the handshake every collective
lowers to.

A **barrier** is the rendezvous: every rank waits until *all* ranks arrive. No
data is exchanged — the barrier is pure synchronization. This step writes the
barrier by hand from `notify`/`wait` and makes the arrival pattern *visible*:
each rank owns one row of every peer's signal window, and after the barrier,
rank `r`'s own row reads `[1, …, 0, …, 1]` — a `1` in every column except its
own, because a rank never notifies itself. Surfacing that row is the proof
that every peer arrived.

**Why `AtomicAdd` + `Ge` keeps the barrier reusable.** A one-shot barrier could
use `Set`/`Eq`, but a reusable one must be safe to call many times on the same
window without resetting it. `AtomicAdd` only ever *grows* a counter and `Ge`
passes when the counter reaches the threshold — so consecutive calls compose
naturally, and nothing needs zeroing between calls.

**Cost card:** one communication round, `P-1` notifications + `P-1` waits per
rank, zero data bytes. This is the cheapest rendezvous in the language —
a benchmark floor for every collective.

## Run it

```bash
# Hand-rolled barrier (default):
python examples/distributed/04_barrier.py -p a2a3sim -d 0,1

# The reveal -- pld.tensor.barrier orders a remote_load:
python examples/distributed/04_barrier.py -p a2a3sim -d 0,1 --use-builtin
```

Expected output:

```text
OK
```

## Walkthrough

The hand-rolled kernel:

```python
@pl.jit.incore
def barrier_handrolled(
    y: pl.Out[pl.Tensor[[N_RANKS, 1], pl.INT32]],
    signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
):
    ctx = pld.get_comm_ctx(signal)
    my_rank = pld.rank(ctx)

    for peer in pl.range(N_RANKS):
        if peer != my_rank:
            pld.system.notify(
                signal, peer=peer, offsets=[my_rank, 0],
                value=1, op=pld.NotifyOp.AtomicAdd,
            )
    for src in pl.range(N_RANKS):
        if src != my_rank:
            pld.system.wait(
                signal, offsets=[src, 0],
                expected=1, cmp=pld.WaitCmp.Ge,
            )

    for i in pl.range(N_RANKS):
        val = pl.read(signal, [i, 0])
        pl.write(y, [i, 0], val)
    return y
```

- **The context.** `pld.get_comm_ctx(signal)` resolves the comm context the
  window belongs to; `pld.rank(ctx)` (and `pld.nranks(ctx)`) come from it. The
  InCore kernel does not get rank as a scalar argument — it derives it from the
  window.
- **The notify phase.** Every rank notifies every *other* rank, writing `1`
  with `AtomicAdd` into row `my_rank` of the peer's signal window. Rank `r`
  never notifies itself — that is why its own row ends with a `0` in column
  `r`.
- **The wait phase.** Every rank waits on every other rank's row in its *own*
  window, `expected=1, cmp=Ge`. Because peers use `AtomicAdd`, the cell only
  needs to reach `1`.
- **The observable.** Reading the signal row cell-by-cell with `pl.read`/
  `pl.write` surfaces the arrival pattern. (A *tile* load of the `[2,1] INT32`
  window would be rejected: its 8-byte column is below the 32-byte alignment
  ptoas requires for a col-major tile — see Edge cases.)

The builtin reveal:

```python
@pl.jit.incore
def barrier_builtin(x, y, data, signal):
    ...
    local = pl.load(x, [0, 0], [1, SIZE])
    data = pl.store(local, [0, 0], data)
    signal = pld.tensor.barrier(signal)
    peer = (my_rank + 1) % nranks
    recv = pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[1, SIZE])
    y = pl.store(recv, [0, 0], y)
    return y
```

`pld.tensor.barrier(signal)` is exactly the synchronization the hand-rolled
loop performs — but it synchronizes *without leaving a tally* in the signal
window, so the reveal proves the barrier a different way: with data. Every
rank stages its slice, barriers, then remote-loads the next rank's slice. A
missing barrier would let the load race the peer's store; the golden
`y[r] = x[(r+1) % N]` holds only because the barrier ordered them. The same
`x`/`signal`/`data` host shape as before, one call instead of a loop.

## Edge cases

> **Fatal pitfall — `Set`/`Eq` barrier that works once.** `NotifyOp.Set` +
> `WaitCmp.Eq` produces a barrier that passes the first call and then either
> blocks or misfires on the second, because nothing resets the counter. **Fix:**
> use `AtomicAdd` + `Ge` so counters are monotonic and calls compose.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| Barrier hangs on the second call | `Set`/`Eq` — counter not monotonic | Use `AtomicAdd` + `Ge` |
| Own signal row shows `0` where peers arrived | Forgot rank `r` skips itself in the notify loop | Skip `peer == my_rank` |
| `pto.alloc_tile` … `32-byte aligned` | Tile-loading a narrow `INT32` window (e.g. `[2,1]` = 8 B column) | Read/write cells as scalars with `pl.read`/`pl.write`, or widen the window |
| Builtin reveal output is all zeros | Reading the signal tally after `pld.tensor.barrier` | The builtin synchronizes but does not leave a tally — prove ordering with data instead |
| One rank waits forever | Notify/wait target rows mismatched | Notify writes row `my_rank` of peer; wait reads row `src` of self |

## See also

- [05-ladder](05-ladder.md) — the ladder map (this step = row 04)
- [02-primitives](../distributed/02-primitives.md) §Notify & Wait + §Choosing
  NotifyOp and WaitCmp — the full signal API
- [01-collectives](../distributed/01-collectives.md) §Barrier — where the
  barrier sits in the collective zoo
- Next step: [10-remote_load_store](10-remote_load_store.md) — move a slice
