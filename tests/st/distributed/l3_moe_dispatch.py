# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 MoE token dispatch stage (host routing + device staging).

Phase-1 functional path: a HOST SubWorker replays the dispatch golden from
``l3_moe_reference`` (expert_idx-driven placement). A minimal CHIP noop verifies
the distributed launch path. Device put-based irregular all-to-all is covered
separately by ``test_l3_put`` and planned for a follow-on ``put_row`` kernel.
"""

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
    compute_dispatch_golden,
)

__all__ = ["build_l3_moe_dispatch_program"]


def build_l3_moe_dispatch_program():
    """Dispatch program: SubWorker fills recv_x; chip is a no-op staging kernel."""

    scratch_bytes = window_scratch_nbytes()
    x_shape = [MOE_NRANKS, MOE_M, MOE_H]
    recv_shape = [MOE_NRANKS, MOE_L, MOE_RECV_MAX, MOE_H]
    expert_idx_shape = [MOE_NRANKS, MOE_M, 2]

    @pl.program
    class L3MoeDispatchProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def chip_stage(
            self,
            recv_x: pl.InOut[pl.Tensor[recv_shape, pl.BF16]],  # type: ignore[valid-type]
            _scratch: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[recv_shape, pl.BF16]:  # type: ignore[valid-type]
            _ = _scratch
            return recv_x

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            recv_x: pl.InOut[pl.Tensor[recv_shape, pl.BF16]],  # type: ignore[valid-type]
            scratch: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
        ) -> pl.Tensor[recv_shape, pl.BF16]:  # type: ignore[valid-type]
            return self.chip_stage(recv_x, scratch)

        @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
        def host_dispatch(
            x: pl.Tensor[x_shape, pl.BF16],
            expert_idx: pl.Tensor[expert_idx_shape, pl.INT32],
            recv_x: pl.Out[pl.Tensor[recv_shape, pl.BF16]],
        ) -> pl.Tensor[recv_shape, pl.BF16]:
            xs = [x[r] for r in range(MOE_NRANKS)]
            golden_recv, _, _ = compute_dispatch_golden(xs, expert_idx)
            for r in range(MOE_NRANKS):
                recv_x[r] = golden_recv[r]
            return recv_x

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            x: pl.Tensor[x_shape, pl.BF16],  # type: ignore[valid-type]
            expert_idx: pl.Tensor[expert_idx_shape, pl.INT32],  # type: ignore[valid-type]
            recv_x: pl.Out[pl.Tensor[recv_shape, pl.BF16]],  # type: ignore[valid-type]
        ) -> pl.Tensor[recv_shape, pl.BF16]:  # type: ignore[valid-type]
            recv_x = self.host_dispatch(x, expert_idx, recv_x)
            scratch_buf = pld.alloc_window_buffer(scratch_bytes)
            for r in pl.range(pld.world_size()):
                scratch = pld.window(scratch_buf, [1, 1], dtype=pl.INT32)
                recv_x = self.chip_orch(recv_x, scratch, device=r)
            return recv_x

    return L3MoeDispatchProgram
