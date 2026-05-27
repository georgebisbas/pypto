# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""MoE unpermute: weighted sum of expert outputs back to [M, H] (HOST SubWorker)."""

from __future__ import annotations

import pypto.language as pl
import pypto.language.distributed as pld

from tests.st.distributed.l3_common import window_scratch_nbytes
from tests.st.distributed.l3_moe_reference import (
    MOE_H,
    MOE_L,
    MOE_M,
    MOE_NRANKS,
    MOE_RECV_MAX,
    MOE_TOPK,
    compute_combine_unpermute_golden,
)

__all__ = ["build_l3_moe_unpermute_program"]


def build_l3_moe_unpermute_program():
    scratch_bytes = window_scratch_nbytes()
    recv_y_shape = [MOE_NRANKS, MOE_L, MOE_RECV_MAX, MOE_H]
    recv_count_shape = [MOE_NRANKS, MOE_L]
    expert_idx_shape = [MOE_NRANKS, MOE_M, MOE_TOPK]
    probs_shape = [MOE_NRANKS, MOE_M, MOE_TOPK]
    out_shape = [MOE_NRANKS, MOE_M, MOE_H]

    @pl.program
    class L3MoeUnpermuteProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def chip_stage(
            self,
            out: pl.InOut[pl.Tensor[out_shape, pl.BF16]],  # type: ignore[valid-type]
            _scratch: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[out_shape, pl.BF16]:  # type: ignore[valid-type]
            _ = _scratch
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            out: pl.InOut[pl.Tensor[out_shape, pl.BF16]],  # type: ignore[valid-type]
            scratch: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[out_shape, pl.BF16]:  # type: ignore[valid-type]
            return self.chip_stage(out, scratch)

        @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
        def host_unpermute(
            recv_y: pl.Tensor[recv_y_shape, pl.BF16],
            recv_count: pl.Tensor[recv_count_shape, pl.INT32],
            expert_idx: pl.Tensor[expert_idx_shape, pl.INT32],
            probs: pl.Tensor[probs_shape, pl.FP32],
            expanded_row_idx: pl.Tensor[[MOE_NRANKS, MOE_M, MOE_TOPK], pl.INT32],
            out: pl.Out[pl.Tensor[out_shape, pl.BF16]],
        ) -> pl.Tensor[out_shape, pl.BF16]:
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
            recv_y: pl.Tensor[recv_y_shape, pl.BF16],  # type: ignore[valid-type]
            recv_count: pl.Tensor[recv_count_shape, pl.INT32],  # type: ignore[valid-type]
            expert_idx: pl.Tensor[expert_idx_shape, pl.INT32],  # type: ignore[valid-type]
            probs: pl.Tensor[probs_shape, pl.FP32],  # type: ignore[valid-type]
            expanded_row_idx: pl.Tensor[[MOE_NRANKS, MOE_M, MOE_TOPK], pl.INT32],  # type: ignore[valid-type]
            out: pl.Out[pl.Tensor[out_shape, pl.BF16]],  # type: ignore[valid-type]
        ) -> pl.Tensor[out_shape, pl.BF16]:  # type: ignore[valid-type]
            out = self.host_unpermute(
                recv_y, recv_count, expert_idx, probs, expanded_row_idx, out
            )
            scratch_buf = pld.alloc_window_buffer(scratch_bytes)
            for r in pl.range(pld.world_size()):
                scratch = pld.window(scratch_buf, [1, 1], dtype=pl.INT32)
                out = self.chip_orch(out, scratch, device=r)
            return out

    return L3MoeUnpermuteProgram
