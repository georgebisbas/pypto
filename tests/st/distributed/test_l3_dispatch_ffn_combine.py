# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""ST: end-to-end L3 dispatch_ffn_combine (BF16, 2-rank EP)."""

import sys

import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

from tests.st.distributed.l3_dispatch_ffn_combine import build_l3_dispatch_ffn_combine_program
from tests.st.distributed.l3_moe_ffn_bf16 import MOE_FFN_ROWS
from tests.st.distributed.l3_moe_reference import (
    MOE_H,
    MOE_L,
    MOE_M,
    MOE_N,
    MOE_NRANKS,
    MOE_RECV_MAX,
    MOE_TOPK,
    golden_dispatch_ffn_combine,
)


def _injective_expert_idx() -> torch.Tensor:
    """Four tokens per local expert on each rank (matches fixed-row CHIP FFN)."""
    idx = torch.zeros(MOE_NRANKS, MOE_M, MOE_TOPK, dtype=torch.int32)
    for r in range(MOE_NRANKS):
        for t in range(MOE_FFN_ROWS):
            idx[r, t, 0] = 2 * r
            idx[r, t, 1] = 2 * r + 1
        for t in range(MOE_FFN_ROWS, MOE_M):
            idx[r, t, 0] = 2 * ((r + 1) % MOE_NRANKS)
            idx[r, t, 1] = 2 * ((r + 1) % MOE_NRANKS) + 1
    return idx


class TestL3DispatchFFNCombine:
    def test_dispatch_ffn_combine_e2e(self, test_config, device_ids):
        if len(device_ids) < 2:
            pytest.skip(f"needs 2 devices, got {device_ids}")

        program = build_l3_dispatch_ffn_combine_program()
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:2],
                num_sub_workers=1,
            ),
        )

        torch.manual_seed(3)
        x = torch.randn(MOE_NRANKS, MOE_M, MOE_H, dtype=torch.bfloat16)
        w1 = torch.randn(MOE_NRANKS, MOE_L, MOE_H, 2 * MOE_N, dtype=torch.bfloat16)
        w2 = torch.randn(MOE_NRANKS, MOE_L, MOE_N, MOE_H, dtype=torch.bfloat16)
        expert_idx = _injective_expert_idx()
        probs = torch.softmax(torch.randn(MOE_NRANKS, MOE_M, MOE_TOPK), dim=-1)

        recv_x = torch.zeros(MOE_NRANKS, MOE_L, MOE_RECV_MAX, MOE_H, dtype=torch.bfloat16)
        recv_y = torch.zeros_like(recv_x)
        recv_count = torch.zeros(MOE_NRANKS, MOE_L, dtype=torch.int32)
        expanded_row_idx = torch.zeros(MOE_NRANKS, MOE_M, MOE_TOPK, dtype=torch.int32)
        out = torch.zeros(MOE_NRANKS, MOE_M, MOE_H, dtype=torch.bfloat16)
        expert_token_nums = torch.zeros(MOE_NRANKS, MOE_L, dtype=torch.int32)

        compiled(
            x,
            w1,
            w2,
            expert_idx,
            probs,
            recv_x,
            recv_y,
            recv_count,
            expanded_row_idx,
            out,
            expert_token_nums,
        )

        expected_out, expected_counts = golden_dispatch_ffn_combine(
            x,
            w1,
            w2,
            expert_idx,
            probs,
            max_ffn_rows_per_expert=MOE_FFN_ROWS,
        )
        assert torch.equal(expert_token_nums, expected_counts)
        assert torch.allclose(
            out.to(torch.float32),
            expected_out.to(torch.float32),
            rtol=0.15,
            atol=0.15,
        ), f"max diff {(out - expected_out).abs().max().item()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
