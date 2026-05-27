# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""MoE combine stage — identity pass-through for Phase-1 (combine fused into unpermute golden)."""

from __future__ import annotations

import pypto.language as pl

from tests.st.distributed.l3_moe_reference import MOE_H, MOE_L, MOE_NRANKS, MOE_RECV_MAX

__all__ = ["build_l3_moe_combine_program"]


def build_l3_moe_combine_program():
    """Combine is modeled inside ``l3_moe_unpermute``; this stage copies recv_y -> recv_y."""

    shape = [MOE_NRANKS, MOE_L, MOE_RECV_MAX, MOE_H]

    @pl.program
    class L3MoeCombineProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def combine(
            self,
            recv_y: pl.Tensor[shape, pl.BF16],  # type: ignore[valid-type]
            out: pl.Out[pl.Tensor[shape, pl.BF16]],  # type: ignore[valid-type]
        ) -> pl.Tensor[shape, pl.BF16]:  # type: ignore[valid-type]
            tile = pl.load(recv_y, [0, 0, 0, 0], [MOE_NRANKS, MOE_L, MOE_RECV_MAX, MOE_H])
            return pl.store(tile, [0, 0, 0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            recv_y: pl.Tensor[shape, pl.BF16],  # type: ignore[valid-type]
            out: pl.Out[pl.Tensor[shape, pl.BF16]],  # type: ignore[valid-type]
        ) -> pl.Tensor[shape, pl.BF16]:  # type: ignore[valid-type]
            return self.combine(recv_y, out)

    return L3MoeCombineProgram
