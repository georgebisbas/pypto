# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Strong-scaling benchmark: L3 GEMM (M-shard) and hierarchical split-K GEMM (K-shard).

Strong scaling:
  - Fixed global ``(M, K, N)``.
  - ``gemm-only``: Phase 2 row-shard (from #1563).
  - ``hier-split-k``: K-shard + intra-chip CORE_GROUP split-K + inter-chip allreduce (P=2).

``speedup(P) = T(1) / T(P)``, ``efficiency(P) = speedup / P``.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

from tests.st.distributed.l3_gemm import build_l3_gemm_program
from tests.st.distributed.l3_hier_split_k_gemm import (
    DEFAULT_K,
    DEFAULT_SPLIT,
    build_l3_hier_split_k_allreduce_gemm_program,
)


def _parse_device_ids(raw: str) -> list[int]:
    raw = raw.strip()
    if re.fullmatch(r"\d+-\d+", raw):
        lo, hi = (int(x) for x in raw.split("-", 1))
        if lo > hi:
            raise ValueError(f"invalid device range: {raw}")
        return list(range(lo, hi + 1))
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_ranks(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _trimmed_mean_us(samples_s: list[float], trim_frac: float = 0.1) -> float:
    if not samples_s:
        return 0.0
    sorted_s = sorted(samples_s)
    n = len(sorted_s)
    k = int(n * trim_frac)
    core = sorted_s[k : n - k] if n > 2 * k else sorted_s
    return statistics.mean(core) * 1e6


def _bench_gemm_only(
    *,
    platform: str,
    device_ids: list[int],
    p: int,
    m: int,
    k: int,
    n: int,
    warmup: int,
    rounds: int,
) -> float:
    m_r = m // p
    if m_r * p != m:
        raise ValueError(f"M={m} must be divisible by P={p}")

    program = build_l3_gemm_program(nranks=p, m0=m_r, k=k, n=n)
    compiled = ir.compile(
        program,
        platform=platform,
        distributed_config=DistributedConfig(
            device_ids=device_ids[:p],
            num_sub_workers=0,
        ),
    )

    torch.manual_seed(0)
    a = torch.randn(p, m_r, k, dtype=torch.float32)
    b = torch.randn(k, n, dtype=torch.float32)
    c = torch.zeros(p, m_r, n, dtype=torch.float32)

    for _ in range(warmup):
        compiled(a, b, c)

    samples: list[float] = []
    for _ in range(rounds):
        c.zero_()
        t0 = time.perf_counter()
        compiled(a, b, c)
        samples.append(time.perf_counter() - t0)
    return _trimmed_mean_us(samples)


def _bench_hier_split_k(
    *,
    platform: str,
    device_ids: list[int],
    p: int,
    m: int,
    k: int,
    n: int,
    split: int,
    warmup: int,
    rounds: int,
) -> float:
    if p != 2:
        raise ValueError(f"hier-split-k bench requires P=2 (got P={p})")
    if k % p != 0:
        raise ValueError(f"K={k} must be divisible by P={p}")

    k_s = k // p
    program = build_l3_hier_split_k_allreduce_gemm_program(
        nranks=p,
        m0=m,
        k=k,
        n=n,
        split=split,
    )
    compiled = ir.compile(
        program,
        platform=platform,
        distributed_config=DistributedConfig(
            device_ids=device_ids[:p],
            num_sub_workers=0,
        ),
    )

    torch.manual_seed(0)
    a = torch.randn(p, m, k_s, dtype=torch.float32)
    b = torch.randn(p, k_s, n, dtype=torch.float32)
    partials = torch.zeros(p, m, n, dtype=torch.float32)
    outputs = torch.zeros(p, m, n, dtype=torch.float32)

    for _ in range(warmup):
        compiled(a, b, partials, outputs)

    samples: list[float] = []
    for _ in range(rounds):
        partials.zero_()
        outputs.zero_()
        t0 = time.perf_counter()
        compiled(a, b, partials, outputs)
        samples.append(time.perf_counter() - t0)
    return _trimmed_mean_us(samples)


def _add_metrics(rows: list[dict], *, mode: str) -> None:
    baseline = next((r for r in rows if r["mode"] == mode and r["P"] == 1), None)
    if baseline is None:
        return
    t1 = baseline["T_med_us"]
    for r in rows:
        if r["mode"] != mode:
            continue
        r["T1_us"] = t1
        r["speedup"] = t1 / r["T_med_us"] if r["T_med_us"] > 0 else 0.0
        r["efficiency"] = r["speedup"] / r["P"] if r["P"] > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="L3 strong-scaling benchmark (gemm-only / hier-split-k)")
    parser.add_argument("-p", "--platform", default="a2a3sim", help="PyPTO platform id")
    parser.add_argument("-d", "--devices", default="0-1", help="Device ids, e.g. 0-7 or 0,1,2")
    parser.add_argument("--ranks", default="1,2", help="Comma-separated P values to sweep")
    parser.add_argument("--m", type=int, default=64, help="Global M (rows)")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Global K")
    parser.add_argument("--n", type=int, default=64, help="Global N")
    parser.add_argument("--split", type=int, default=DEFAULT_SPLIT, help="CORE_GROUP split factor per chip")
    parser.add_argument(
        "--mode",
        choices=("gemm-only", "hier-split-k", "all"),
        default="all",
        help="Benchmark program variant",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("-o", "--output", default="", help="JSON output path (optional)")
    args = parser.parse_args()

    device_ids = _parse_device_ids(args.devices)
    rank_list = _parse_ranks(args.ranks)
    modes: list[str] = ["gemm-only", "hier-split-k"] if args.mode == "all" else [args.mode]

    rows: list[dict] = []
    for mode in modes:
        for p in rank_list:
            if p > len(device_ids):
                print(f"skip P={p} mode={mode}: need {p} devices, have {len(device_ids)}", file=sys.stderr)
                continue
            if mode == "hier-split-k" and p != 2:
                print(f"skip P={p} mode={mode}: hier-split-k requires P=2", file=sys.stderr)
                continue
            try:
                if mode == "gemm-only":
                    t_us = _bench_gemm_only(
                        platform=args.platform,
                        device_ids=device_ids,
                        p=p,
                        m=args.m,
                        k=args.k,
                        n=args.n,
                        warmup=args.warmup,
                        rounds=args.rounds,
                    )
                    m_r = args.m // p
                else:
                    t_us = _bench_hier_split_k(
                        platform=args.platform,
                        device_ids=device_ids,
                        p=p,
                        m=args.m,
                        k=args.k,
                        n=args.n,
                        split=args.split,
                        warmup=args.warmup,
                        rounds=args.rounds,
                    )
                    m_r = args.m
            except Exception as exc:  # noqa: BLE001
                print(f"fail P={p} mode={mode}: {exc}", file=sys.stderr)
                continue

            rows.append(
                {
                    "mode": mode,
                    "P": p,
                    "M": args.m,
                    "K": args.k,
                    "N": args.n,
                    "M_r": m_r,
                    "split": args.split if mode == "hier-split-k" else None,
                    "T_med_us": t_us,
                    "platform": args.platform,
                    "devices": device_ids[:p],
                }
            )

    for mode in modes:
        _add_metrics(rows, mode=mode)

    header = f"{'mode':<16} {'P':>3} {'M_r':>6} {'K':>6} {'N':>6} {'T_med_us':>12} {'speedup':>8} {'eff':>8}"
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda x: (x["mode"], x["P"])):
        speedup = r.get("speedup", float("nan"))
        eff = r.get("efficiency", float("nan"))
        print(
            f"{r['mode']:<16} {r['P']:>3} {r['M_r']:>6} {r['K']:>6} {r['N']:>6} "
            f"{r['T_med_us']:>12.1f} {speedup:>8.3f} {eff:>8.3f}"
        )

    out_path = args.output
    if not out_path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = f"tmp/bench_l3_strong_scaling_{ts}.json"
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": args.platform,
        "global_MKN": [args.m, args.k, args.n],
        "warmup": args.warmup,
        "rounds": args.rounds,
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
