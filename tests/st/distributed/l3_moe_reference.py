# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Host-side golden reference for L3 MoE dispatch_ffn_combine (BF16 SwiGLU FFN).

Mirrors the protocol in ``runtime/examples/workers/l3/ep_dispatch_combine/main.py``
and the dataflow in vLLM ``dispatch_ffn_combine``, without device quant routing.
"""

from __future__ import annotations

import torch

# ST topology (2-rank EP).
MOE_NRANKS = 2
MOE_M = 8
MOE_TOPK = 2
MOE_L = 2  # local experts per rank
MOE_H = 64  # hidden / FFN K
MOE_N = 64  # SwiGLU intermediate (gate/up width)
MOE_RECV_MAX = MOE_M * MOE_TOPK
MOE_E_GLOBAL = MOE_NRANKS * MOE_L


def swiglu_ffn_row(x_row: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """Single-token SwiGLU FFN: x [H], w1 [H, 2N], w2 [N, H] -> [H] (BF16 in/out)."""
    x = x_row.to(torch.float32)
    w1f = w1.to(torch.float32)
    w2f = w2.to(torch.float32)
    gate_up = x @ w1f
    gate, up = gate_up[: MOE_N], gate_up[MOE_N:]
    hidden = torch.nn.functional.silu(gate) * up
    out = hidden @ w2f
    return out.to(torch.bfloat16)


def compute_dispatch_golden(
    x: list[torch.Tensor],
    expert_idx: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Replay token dispatch (BF16 payloads only).

    Returns per-rank ``recv_x[L, RECV_MAX, H]``, ``recv_count[L]``, and
    ``expanded_row_idx[nranks, M, TOPK]`` (global row index = t * TOPK + k).
    """
    recv_x = [torch.zeros(MOE_L, MOE_RECV_MAX, MOE_H, dtype=torch.bfloat16) for _ in range(MOE_NRANKS)]
    recv_count = [torch.zeros(MOE_L, dtype=torch.int32) for _ in range(MOE_NRANKS)]
    expanded_row_idx = torch.zeros(MOE_NRANKS, MOE_M, MOE_TOPK, dtype=torch.int32)

    send_counts = torch.zeros(MOE_NRANKS, MOE_NRANKS, MOE_L, dtype=torch.int32)
    for src in range(MOE_NRANKS):
        for t in range(MOE_M):
            for k in range(MOE_TOPK):
                eid = int(expert_idx[src, t, k].item())
                dst = eid // MOE_L
                loc_e = eid % MOE_L
                send_counts[src, dst, loc_e] += 1

    for dst in range(MOE_NRANKS):
        slot_offset = torch.zeros(MOE_NRANKS, MOE_L, dtype=torch.int32)
        running = torch.zeros(MOE_L, dtype=torch.int32)
        for src in range(MOE_NRANKS):
            slot_offset[src] = running.clone()
            running = running + send_counts[src, dst]

        cursor = torch.zeros(MOE_L, dtype=torch.int32)
        for src in range(MOE_NRANKS):
            for t in range(MOE_M):
                for k in range(MOE_TOPK):
                    eid = int(expert_idx[src, t, k].item())
                    if eid // MOE_L != dst:
                        continue
                    loc_e = eid % MOE_L
                    slot = int(slot_offset[src, loc_e].item() + cursor[loc_e].item())
                    cursor[loc_e] += 1
                    recv_x[dst][loc_e, slot, :] = x[src][t, :]
                    expanded_row_idx[src, t, k] = t * MOE_TOPK + k

        for e in range(MOE_L):
            recv_count[dst][e] = int(running[e].item())

    return recv_x, recv_count, expanded_row_idx


def compute_ffn_golden(
    recv_x: list[torch.Tensor],
    recv_count: list[torch.Tensor],
    w1: list[torch.Tensor],
    w2: list[torch.Tensor],
    *,
    max_rows_per_expert: int | None = None,
) -> list[torch.Tensor]:
    """Per-rank expert FFN on dispatched rows -> recv_y[L, RECV_MAX, H]."""
    recv_y = []
    for r in range(MOE_NRANKS):
        y = torch.zeros(MOE_L, MOE_RECV_MAX, MOE_H, dtype=torch.bfloat16)
        for e in range(MOE_L):
            n = int(recv_count[r][e].item())
            if max_rows_per_expert is not None:
                n = min(n, max_rows_per_expert)
            for s in range(n):
                y[e, s, :] = swiglu_ffn_row(recv_x[r][e, s, :], w1[r][e], w2[r][e])
        recv_y.append(y)
    return recv_y


def compute_combine_unpermute_golden(
    recv_y: list[torch.Tensor],
    recv_count: list[torch.Tensor],
    expert_idx: torch.Tensor,
    probs: torch.Tensor,
    expanded_row_idx: torch.Tensor,
) -> list[torch.Tensor]:
    """Weighted topK reduction back to token layout [M, H] per rank."""
    out = []
    for src in range(MOE_NRANKS):
        acc = torch.zeros(MOE_M, MOE_H, dtype=torch.float32)
        send_counts = torch.zeros(MOE_NRANKS, MOE_NRANKS, MOE_L, dtype=torch.int32)
        for s in range(MOE_NRANKS):
            for t in range(MOE_M):
                for k in range(MOE_TOPK):
                    eid = int(expert_idx[s, t, k].item())
                    send_counts[s, eid // MOE_L, eid % MOE_L] += 1

        slot_offset = torch.zeros(MOE_NRANKS, MOE_L, dtype=torch.int32)
        running = torch.zeros(MOE_L, dtype=torch.int32)
        for s in range(MOE_NRANKS):
            slot_offset[s] = running.clone()
            running = running + send_counts[s, :]

        cursor = torch.zeros(MOE_L, dtype=torch.int32)
        for t in range(MOE_M):
            for k in range(MOE_TOPK):
                eid = int(expert_idx[src, t, k].item())
                dst = eid // MOE_L
                loc_e = eid % MOE_L
                slot = int(slot_offset[src, loc_e].item() + cursor[loc_e].item())
                cursor[loc_e] += 1
                row_idx = int(expanded_row_idx[src, t, k].item())
                y_row = recv_y[dst][loc_e, slot, :].to(torch.float32)
                acc[t, :] += probs[src, t, k].item() * y_row
        out.append(acc.to(torch.bfloat16))
    return out


def golden_dispatch_ffn_combine(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    expert_idx: torch.Tensor,
    probs: torch.Tensor,
    *,
    max_ffn_rows_per_expert: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full pipeline golden.

    Args:
        x: [nranks, M, H] BF16
        w1: [nranks, L, H, 2*N] BF16
        w2: [nranks, L, N, H] BF16
        expert_idx: [nranks, M, topK] int32 global expert id
        probs: [nranks, M, topK] float32

    Returns:
        out: [nranks, M, H] BF16
        expert_token_nums: [nranks, L] int32
    """
    xs = [x[r] for r in range(MOE_NRANKS)]
    w1s = [w1[r] for r in range(MOE_NRANKS)]
    w2s = [w2[r] for r in range(MOE_NRANKS)]
    recv_x, recv_count, expanded_row_idx = compute_dispatch_golden(xs, expert_idx)
    recv_y = compute_ffn_golden(
        recv_x, recv_count, w1s, w2s, max_rows_per_expert=max_ffn_rows_per_expert
    )
    outs = compute_combine_unpermute_golden(recv_y, recv_count, expert_idx, probs, expanded_row_idx)
    expert_nums = torch.stack([c for c in recv_count])
    return torch.stack(outs), expert_nums
