# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""ST: L3 MoE dispatch stage (host routing golden)."""

import sys

import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

from tests.st.distributed.l3_moe_dispatch import build_l3_moe_dispatch_program
from tests.st.distributed.l3_moe_reference import (
    MOE_E_GLOBAL,
    MOE_H,
    MOE_M,
    MOE_NRANKS,
    compute_dispatch_golden,
)


def _fixed_expert_idx(seed: int = 7) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    idx = torch.zeros(MOE_NRANKS, MOE_M, 2, dtype=torch.int32)
    for r in range(MOE_NRANKS):
        for t in range(MOE_M):
            perm = torch.randperm(MOE_E_GLOBAL, generator=gen)[:2]
            idx[r, t, :] = perm.to(torch.int32)
    return idx


class TestL3MoeDispatch:
    def test_moe_dispatch_host_routing(self, test_config, device_ids):
        if len(device_ids) < 2:
            pytest.skip(f"needs 2 devices, got {device_ids}")

        program = build_l3_moe_dispatch_program()
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
        recv_x = torch.zeros(MOE_NRANKS, 2, MOE_M * 2, MOE_H, dtype=torch.bfloat16)

        compiled(x, expert_idx, recv_x)

        xs = [x[r] for r in range(MOE_NRANKS)]
        expected_recv, _, _ = compute_dispatch_golden(xs, expert_idx)
        for r in range(MOE_NRANKS):
            exp = expected_recv[r].to(torch.float32)
            got = recv_x[r].to(torch.float32)
            assert torch.allclose(got[:, : exp.shape[1], :], exp, rtol=1e-2, atol=1e-2), (
                f"rank {r} dispatch mismatch"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
