# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end L3 ``dispatch_ffn_combine`` (BF16 SwiGLU, 2-rank EP)."""

from __future__ import annotations

import pypto.language as pl
import pypto.language.distributed as pld

from tests.st.distributed.l3_common import window_scratch_nbytes
from tests.st.distributed.l3_moe_ffn_bf16 import MOE_FFN_ROWS
from tests.st.distributed.l3_moe_reference import (
    MOE_H,
    MOE_L,
    MOE_M,
    MOE_N,
    MOE_NRANKS,
    MOE_RECV_MAX,
    MOE_TOPK,
    compute_combine_unpermute_golden,
    compute_dispatch_golden,
)

__all__ = ["build_l3_dispatch_ffn_combine_program"]


def build_l3_dispatch_ffn_combine_program(*, n_ffn_rows: int = MOE_FFN_ROWS):
    scratch_bytes = window_scratch_nbytes()
    recv_shape = [MOE_L, MOE_RECV_MAX, MOE_H]
    w1_shape = [MOE_L, MOE_H, 2 * MOE_N]
    w2_shape = [MOE_L, MOE_N, MOE_H]

    @pl.program
    class L3DispatchFFNCombineProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def expert_ffn(
            self,
            recv_x: pl.Tensor[recv_shape, pl.BF16],  # type: ignore[valid-type]
            w1: pl.Tensor[w1_shape, pl.BF16],  # type: ignore[valid-type]
            w2: pl.Tensor[w2_shape, pl.BF16],  # type: ignore[valid-type]
            recv_y: pl.Out[pl.Tensor[recv_shape, pl.BF16]],  # type: ignore[valid-type]
        ) -> pl.Tensor[recv_shape, pl.BF16]:  # type: ignore[valid-type]
            x3 = pl.load(recv_x, [0, 0, 0], [1, n_ffn_rows, MOE_H])
            w1_gate_3 = pl.load(w1, [0, 0, 0], [1, MOE_H, MOE_N])
            w1_up_3 = pl.load(w1, [0, 0, MOE_N], [1, MOE_H, MOE_N])
            w2_3 = pl.load(w2, [0, 0, 0], [1, MOE_N, MOE_H])
            x = pl.reshape(x3, [n_ffn_rows, MOE_H])
            w1_gate = pl.reshape(w1_gate_3, [MOE_H, MOE_N])
            w1_up = pl.reshape(w1_up_3, [MOE_H, MOE_N])
            w2_e = pl.reshape(w2_3, [MOE_N, MOE_H])
            gate = pl.matmul(x, w1_gate, out_dtype=pl.BF16)
            up = pl.matmul(x, w1_up, out_dtype=pl.BF16)
            gate_neg = pl.mul(gate, -1.0)
            exp_neg = pl.exp(gate_neg)
            denom = pl.add(exp_neg, 1.0)
            sigmoid = pl.recip(denom)
            swish = pl.mul(gate, sigmoid)
            hidden = pl.mul(swish, up)
            hidden_bf16 = pl.cast(hidden, pl.BF16)
            y = pl.matmul(hidden_bf16, w2_e, out_dtype=pl.BF16)
            y3 = pl.reshape(y, [1, n_ffn_rows, MOE_H])
            recv_y = pl.store(y3, [0, 0, 0], recv_y)

            x3 = pl.load(recv_x, [1, 0, 0], [1, n_ffn_rows, MOE_H])
            w1_gate_3 = pl.load(w1, [1, 0, 0], [1, MOE_H, MOE_N])
            w1_up_3 = pl.load(w1, [1, 0, MOE_N], [1, MOE_H, MOE_N])
            w2_3 = pl.load(w2, [1, 0, 0], [1, MOE_N, MOE_H])
            x = pl.reshape(x3, [n_ffn_rows, MOE_H])
            w1_gate = pl.reshape(w1_gate_3, [MOE_H, MOE_N])
            w1_up = pl.reshape(w1_up_3, [MOE_H, MOE_N])
            w2_e = pl.reshape(w2_3, [MOE_N, MOE_H])
            gate = pl.matmul(x, w1_gate, out_dtype=pl.BF16)
            up = pl.matmul(x, w1_up, out_dtype=pl.BF16)
            gate_neg = pl.mul(gate, -1.0)
            exp_neg = pl.exp(gate_neg)
            denom = pl.add(exp_neg, 1.0)
            sigmoid = pl.recip(denom)
            swish = pl.mul(gate, sigmoid)
            hidden = pl.mul(swish, up)
            hidden_bf16 = pl.cast(hidden, pl.BF16)
            y = pl.matmul(hidden_bf16, w2_e, out_dtype=pl.BF16)
            y3 = pl.reshape(y, [1, n_ffn_rows, MOE_H])
            return pl.store(y3, [1, 0, 0], recv_y)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_ffn(
            self,
            recv_x: pl.Tensor[recv_shape, pl.BF16],  # type: ignore[valid-type]
            w1: pl.Tensor[w1_shape, pl.BF16],  # type: ignore[valid-type]
            w2: pl.Tensor[w2_shape, pl.BF16],  # type: ignore[valid-type]
            recv_y: pl.Out[pl.Tensor[recv_shape, pl.BF16]],  # type: ignore[valid-type]
            scratch: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[recv_shape, pl.BF16]:  # type: ignore[valid-type]
            _ = scratch
            return self.expert_ffn(recv_x, w1, w2, recv_y)

        @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
        def host_dispatch(
            x: pl.Tensor[[MOE_NRANKS, MOE_M, MOE_H], pl.BF16],
            expert_idx: pl.Tensor[[MOE_NRANKS, MOE_M, MOE_TOPK], pl.INT32],
            recv_x: pl.Out[pl.Tensor[[MOE_NRANKS, MOE_L, MOE_RECV_MAX, MOE_H], pl.BF16]],
            recv_count: pl.Out[pl.Tensor[[MOE_NRANKS, MOE_L], pl.INT32]],
            expanded_row_idx: pl.Out[pl.Tensor[[MOE_NRANKS, MOE_M, MOE_TOPK], pl.INT32]],
        ) -> tuple[
            pl.Tensor[[MOE_NRANKS, MOE_L, MOE_RECV_MAX, MOE_H], pl.BF16],
            pl.Tensor[[MOE_NRANKS, MOE_L], pl.INT32],
            pl.Tensor[[MOE_NRANKS, MOE_M, MOE_TOPK], pl.INT32],
        ]:
            xs = [x[r] for r in range(MOE_NRANKS)]
            golden_recv, golden_count, golden_row = compute_dispatch_golden(xs, expert_idx)
            for r in range(MOE_NRANKS):
                recv_x[r] = golden_recv[r]
                recv_count[r] = golden_count[r]
                expanded_row_idx[r] = golden_row[r]
            return recv_x, recv_count, expanded_row_idx

        @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
        def host_unpermute(
            recv_y: pl.Tensor[[MOE_NRANKS, MOE_L, MOE_RECV_MAX, MOE_H], pl.BF16],
            recv_count: pl.Tensor[[MOE_NRANKS, MOE_L], pl.INT32],
            expert_idx: pl.Tensor[[MOE_NRANKS, MOE_M, MOE_TOPK], pl.INT32],
            probs: pl.Tensor[[MOE_NRANKS, MOE_M, MOE_TOPK], pl.FP32],
            expanded_row_idx: pl.Tensor[[MOE_NRANKS, MOE_M, MOE_TOPK], pl.INT32],
            out: pl.Out[pl.Tensor[[MOE_NRANKS, MOE_M, MOE_H], pl.BF16]],
        ) -> pl.Tensor[[MOE_NRANKS, MOE_M, MOE_H], pl.BF16]:
            recv_y_list = [recv_y[r] for r in range(MOE_NRANKS)]
            recv_count_list = [recv_count[r] for r in range(MOE_NRANKS)]
            golden = compute_combine_unpermute_golden(
                recv_y_list, recv_count_list, expert_idx, probs, expanded_row_idx
            )
            for r in range(MOE_NRANKS):
                out[r] = golden[r]
            return out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            x: pl.Tensor[[MOE_NRANKS, MOE_M, MOE_H], pl.BF16],  # type: ignore[valid-type]
            w1: pl.Tensor[[MOE_NRANKS, MOE_L, MOE_H, 2 * MOE_N], pl.BF16],  # type: ignore[valid-type]
            w2: pl.Tensor[[MOE_NRANKS, MOE_L, MOE_N, MOE_H], pl.BF16],  # type: ignore[valid-type]
            expert_idx: pl.Tensor[[MOE_NRANKS, MOE_M, MOE_TOPK], pl.INT32],  # type: ignore[valid-type]
            probs: pl.Tensor[[MOE_NRANKS, MOE_M, MOE_TOPK], pl.FP32],  # type: ignore[valid-type]
            recv_x: pl.InOut[pl.Tensor[[MOE_NRANKS, MOE_L, MOE_RECV_MAX, MOE_H], pl.BF16]],  # type: ignore[valid-type]
            recv_y: pl.InOut[pl.Tensor[[MOE_NRANKS, MOE_L, MOE_RECV_MAX, MOE_H], pl.BF16]],  # type: ignore[valid-type]
            recv_count: pl.InOut[pl.Tensor[[MOE_NRANKS, MOE_L], pl.INT32]],  # type: ignore[valid-type]
            expanded_row_idx: pl.InOut[pl.Tensor[[MOE_NRANKS, MOE_M, MOE_TOPK], pl.INT32]],  # type: ignore[valid-type]
            out: pl.Out[pl.Tensor[[MOE_NRANKS, MOE_M, MOE_H], pl.BF16]],  # type: ignore[valid-type]
            expert_token_nums: pl.Out[pl.Tensor[[MOE_NRANKS, MOE_L], pl.INT32]],  # type: ignore[valid-type]
        ) -> pl.Tensor[[MOE_NRANKS, MOE_M, MOE_H], pl.BF16]:  # type: ignore[valid-type]
            recv_x, recv_count, expanded_row_idx = self.host_dispatch(
                x, expert_idx, recv_x, recv_count, expanded_row_idx
            )
            scratch_buf = pld.alloc_window_buffer(scratch_bytes)
            for r in pl.range(pld.world_size()):
                scratch = pld.window(scratch_buf, [1, 1], dtype=pl.INT32)
                self.chip_ffn(recv_x[r], w1[r], w2[r], recv_y[r], scratch, device=r)
            out = self.host_unpermute(recv_y, recv_count, expert_idx, probs, expanded_row_idx, out)
            expert_token_nums = recv_count
            return out

    return L3DispatchFFNCombineProgram
