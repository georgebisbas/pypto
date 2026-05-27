# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed ST — hierarchical split-K GEMM + allreduce (K-shard across chips).

Golden: ``outputs[r] == sum_s matmul(A[:, s*K_s:(s+1)*K_s], B[s*K_s:(s+1)*K_s, :])`` for every rank.
"""

import sys

import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

from tests.st.distributed.l3_hier_split_k_gemm import (
    DEFAULT_K,
    DEFAULT_SPLIT,
    M0,
    N,
    build_l3_hier_split_k_allreduce_gemm_program,
)


def _golden_hier_split_k_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Replicated full GEMM from K-sharded inputs ``a[r]: [M0, K_s]``, ``b[r]: [K_s, N]``."""
    nranks = a.shape[0]
    partials = [torch.matmul(a[r], b[r]) for r in range(nranks)]
    reduced = sum(partials)
    return torch.stack([reduced] * nranks)


class TestL3HierSplitKGemm:
    """Hierarchical split-K on each chip + 2-rank window allreduce."""

    def test_hier_split_k_allreduce_gemm(self, test_config, device_ids):
        if len(device_ids) < 2:
            pytest.skip(f"hier split-K needs 2 devices, got {device_ids}")

        nranks = 2
        k = DEFAULT_K
        k_s = k // nranks
        program = build_l3_hier_split_k_allreduce_gemm_program(
            nranks=nranks,
            m0=M0,
            k=k,
            n=N,
            split=DEFAULT_SPLIT,
        )
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:nranks],
                num_sub_workers=0,
            ),
        )

        torch.manual_seed(11)
        a = torch.randn(nranks, M0, k_s, dtype=torch.float32)
        b = torch.randn(nranks, k_s, N, dtype=torch.float32)
        partials = torch.zeros(nranks, M0, N, dtype=torch.float32)
        outputs = torch.zeros(nranks, M0, N, dtype=torch.float32)

        compiled(a, b, partials, outputs)

        expected = _golden_hier_split_k_gemm(a, b)
        assert torch.allclose(outputs, expected, rtol=1e-3, atol=1e-3), (
            f"hier split-K mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )

    def test_hier_split_k_allreduce_gemm_determinism_interchip(self, test_config, device_ids):
        """Inter-chip reduce order is fixed; intra-chip atomic assemble may vary at ulp level."""
        if len(device_ids) < 2:
            pytest.skip(f"hier split-K needs 2 devices, got {device_ids}")

        nranks = 2
        program = build_l3_hier_split_k_allreduce_gemm_program(nranks=nranks)
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:nranks],
                num_sub_workers=0,
            ),
        )

        torch.manual_seed(42)
        k_s = DEFAULT_K // nranks
        a = torch.randn(nranks, M0, k_s, dtype=torch.float32)
        b = torch.randn(nranks, k_s, N, dtype=torch.float32)
        expected = _golden_hier_split_k_gemm(a, b)

        for _ in range(5):
            partials = torch.zeros(nranks, M0, N, dtype=torch.float32)
            outputs = torch.zeros(nranks, M0, N, dtype=torch.float32)
            compiled(a, b, partials, outputs)
            assert torch.allclose(outputs, expected, rtol=1e-3, atol=1e-3), (
                "hier split-K determinism vs golden failed"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
