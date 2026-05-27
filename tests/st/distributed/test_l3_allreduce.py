# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: 2-rank allreduce — PyPTO port of ``runtime/examples/workers/l3/allreduce_distributed``.

Mirrors the 4-phase pattern of the runtime example's
``kernels/aiv/allreduce_kernel.cpp``. ``reduce_step`` is imported from
``l3_common`` (same kernel as ``l3_allreduce_gemm``).

Golden: ``outputs[r] == inputs[0] + inputs[1]`` for every rank ``r``.
"""

import sys

import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

from tests.st.distributed.l3_common import build_l3_allreduce_only_program

SIZE = 256  # matches ALLREDUCE_COUNT in runtime allreduce_kernel.cpp


class TestL3AllReduce:
    """L3 distributed runtime: 2-rank allreduce via stage-in + notify/wait + remote_load."""

    def test_allreduce(self, test_config, device_ids):
        if len(device_ids) < 2:
            pytest.skip(f"allreduce needs 2 devices, got {device_ids}")

        program = build_l3_allreduce_only_program(nranks=2, m0=1, n=SIZE)
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:2],
                num_sub_workers=0,
            ),
        )

        inputs = torch.stack(
            [
                torch.arange(SIZE, dtype=torch.float32).reshape(1, SIZE),
                torch.arange(100.0, 100.0 + SIZE, dtype=torch.float32).reshape(1, SIZE),
            ]
        )
        outputs = torch.zeros((2, 1, SIZE), dtype=torch.float32)

        compiled(inputs, outputs)

        reduced = inputs[0] + inputs[1]
        expected = torch.stack([reduced, reduced])
        assert torch.allclose(outputs, expected), (
            f"allreduce mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
