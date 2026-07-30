# Performance Cases

Each case below follows the same pattern: **baseline → investigation →
change → effect → verification**.

> **Status:** Performance cases are planned but not yet written. Real measured
> data across representative workloads has not yet been accumulated. The user
> manual plan ([USER_MANUAL_PLAN_EN §8.1 item 5](https://github.com/hw-native-sys/pypto/issues/2120))
> targets this page for methodology and relative trends in the first version,
> with measured values backfilled once data accumulates.
>
> The single-node performance techniques in [01-single-node](01-single-node.md)
> and distributed performance guidance in [02-distributed](02-distributed.md)
> are available today.
>
> Planned cases:
>
> - **Single-node:** Tile dimension granularity, auto-tiling diagnostics, false
>   task dependencies, ring-sizing arena rebuilds.
> - **Distributed:** Mesh-to-ring collectives transition, resident shards vs
>   H2D per-dispatch.

## See Also

- [00-methodology](00-methodology.md) — Measurement loop and tools
- [01-single-node](01-single-node.md) — Single-node performance techniques
- [02-distributed](02-distributed.md) — Distributed performance and bus bandwidth
