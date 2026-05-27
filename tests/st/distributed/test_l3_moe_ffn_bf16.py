# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""ST: local MoE SwiGLU FFN (single rank, BF16)."""

import sys

import pytest
import torch
from pypto import ir
from pypto.runtime.runner import RunConfig

from tests.st.distributed.l3_moe_ffn_bf16 import MOE_FFN_ROWS, build_l3_moe_ffn_bf16_program
from tests.st.distributed.l3_moe_reference import MOE_H, MOE_L, MOE_N, MOE_RECV_MAX, swiglu_ffn_row


class TestL3MoeFfnBf16:
    def test_moe_ffn_bf16_single_rank(self, test_config):
        program = build_l3_moe_ffn_bf16_program(n_rows=MOE_FFN_ROWS)
        compiled = ir.compile(program, platform=test_config.platform)

        recv_x = torch.randn(MOE_L, MOE_RECV_MAX, MOE_H, dtype=torch.bfloat16)
        w1 = torch.randn(MOE_L, MOE_H, 2 * MOE_N, dtype=torch.bfloat16)
        w2 = torch.randn(MOE_L, MOE_N, MOE_H, dtype=torch.bfloat16)
        recv_y = torch.zeros_like(recv_x)

        compiled(recv_x, w1, w2, recv_y)

        expected = torch.zeros_like(recv_x)
        for e in range(MOE_L):
            for s in range(MOE_FFN_ROWS):
                expected[e, s, :] = swiglu_ffn_row(recv_x[e, s, :], w1[e], w2[e])

        assert torch.allclose(
            recv_y.to(torch.float32), expected.to(torch.float32), rtol=0.1, atol=0.1
        ), f"max diff {(recv_y - expected).abs().max().item()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
