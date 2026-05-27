# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Local MoE expert SwiGLU FFN (BF16) on dispatched token rows — single-rank CHIP program."""

from __future__ import annotations

import pypto.language as pl

from tests.st.distributed.l3_moe_reference import MOE_H, MOE_L, MOE_N, MOE_RECV_MAX

__all__ = ["MOE_FFN_ROWS", "build_l3_moe_ffn_bf16_program"]

# ST uses a fixed number of rows per expert (injective routing fixture).
MOE_FFN_ROWS = 4


def build_l3_moe_ffn_bf16_program(
    *,
    experts_per_rank: int = MOE_L,
    n_rows: int = MOE_FFN_ROWS,
):
    """FFN on recv_x with fixed row count per expert (ST fixture)."""
    if experts_per_rank != MOE_L or n_rows != MOE_FFN_ROWS:
        raise ValueError("only default MOE_L / MOE_FFN_ROWS are supported in ST")

    recv_shape = [experts_per_rank, MOE_RECV_MAX, MOE_H]
    w1_shape = [experts_per_rank, MOE_H, 2 * MOE_N]
    w2_shape = [experts_per_rank, MOE_N, MOE_H]

    @pl.program
    class L3MoeFfnBf16Program:
        @pl.function(type=pl.FunctionType.InCore)
        def expert_ffn(
            self,
            recv_x: pl.Tensor[recv_shape, pl.BF16],  # type: ignore[valid-type]
            w1: pl.Tensor[w1_shape, pl.BF16],  # type: ignore[valid-type]
            w2: pl.Tensor[w2_shape, pl.BF16],  # type: ignore[valid-type]
            recv_y: pl.Out[pl.Tensor[recv_shape, pl.BF16]],  # type: ignore[valid-type]
        ) -> pl.Tensor[recv_shape, pl.BF16]:  # type: ignore[valid-type]
            x3 = pl.load(recv_x, [0, 0, 0], [1, n_rows, MOE_H])
            w1_gate_3 = pl.load(w1, [0, 0, 0], [1, MOE_H, MOE_N])
            w1_up_3 = pl.load(w1, [0, 0, MOE_N], [1, MOE_H, MOE_N])
            w2_3 = pl.load(w2, [0, 0, 0], [1, MOE_N, MOE_H])
            x = pl.reshape(x3, [n_rows, MOE_H])
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
            y3 = pl.reshape(y, [1, n_rows, MOE_H])
            recv_y = pl.store(y3, [0, 0, 0], recv_y)

            x3 = pl.load(recv_x, [1, 0, 0], [1, n_rows, MOE_H])
            w1_gate_3 = pl.load(w1, [1, 0, 0], [1, MOE_H, MOE_N])
            w1_up_3 = pl.load(w1, [1, 0, MOE_N], [1, MOE_H, MOE_N])
            w2_3 = pl.load(w2, [1, 0, 0], [1, MOE_N, MOE_H])
            x = pl.reshape(x3, [n_rows, MOE_H])
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
            y3 = pl.reshape(y, [1, n_rows, MOE_H])
            return pl.store(y3, [1, 0, 0], recv_y)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            recv_x: pl.Tensor[recv_shape, pl.BF16],  # type: ignore[valid-type]
            w1: pl.Tensor[w1_shape, pl.BF16],  # type: ignore[valid-type]
            w2: pl.Tensor[w2_shape, pl.BF16],  # type: ignore[valid-type]
            recv_y: pl.Out[pl.Tensor[recv_shape, pl.BF16]],  # type: ignore[valid-type]
        ) -> pl.Tensor[recv_shape, pl.BF16]:  # type: ignore[valid-type]
            return self.expert_ffn(recv_x, w1, w2, recv_y)

    return L3MoeFfnBf16Program
