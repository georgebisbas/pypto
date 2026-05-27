# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""ST: L3 MoE combine+unpermute stage (HOST SubWorker golden)."""

import sys

import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

from tests.st.distributed.l3_moe_reference import (
    MOE_H,
    MOE_L,
    MOE_M,
    MOE_NRANKS,
    MOE_RECV_MAX,
    MOE_TOPK,
    compute_combine_unpermute_golden,
    compute_dispatch_golden,
)
from tests.st.distributed.l3_moe_unpermute import build_l3_moe_unpermute_program


def _fixed_expert_idx(seed: int = 11) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    idx = torch.zeros(MOE_NRANKS, MOE_M, MOE_TOPK, dtype=torch.int32)
    for r in range(MOE_NRANKS):
        for t in range(MOE_M):
            perm = torch.randperm(MOE_NRANKS * MOE_L, generator=gen)[:MOE_TOPK]
            idx[r, t, :] = perm.to(torch.int32)
    return idx


class TestL3MoeUnpermute:
    def test_moe_unpermute_host_golden(self, test_config, device_ids):
        if len(device_ids) < 2:
            pytest.skip(f"needs 2 devices, got {device_ids}")

        program = build_l3_moe_unpermute_program()
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:2],
                num_sub_workers=1,
            ),
        )

        x = torch.randn(MOE_NRANKS, MOE_M, MOE_H, dtype=torch.bfloat16)
        expert_idx = _fixed_expert_idx()
        probs = torch.softmax(torch.randn(MOE_NRANKS, MOE_M, MOE_TOPK), dim=-1)

        xs = [x[r] for r in range(MOE_NRANKS)]
        recv_x, recv_count, expanded_row_idx = compute_dispatch_golden(xs, expert_idx)
        recv_y = torch.stack([recv_x[r].clone() for r in range(MOE_NRANKS)])
        recv_count_t = torch.stack(recv_count)

        out = torch.zeros(MOE_NRANKS, MOE_M, MOE_H, dtype=torch.bfloat16)
        compiled(recv_y, recv_count_t, expert_idx, probs, expanded_row_idx, out)

        expected = compute_combine_unpermute_golden(
            [recv_y[r] for r in range(MOE_NRANKS)],
            [recv_count_t[r] for r in range(MOE_NRANKS)],
            expert_idx,
            probs,
            expanded_row_idx,
        )
        for r in range(MOE_NRANKS):
            assert torch.allclose(
                out[r].to(torch.float32),
                expected[r].to(torch.float32),
                rtol=1e-2,
                atol=1e-2,
            ), f"rank {r} unpermute mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
