# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""ST: MoE combine identity stage (single-rank CHIP)."""

import sys

import pytest
import torch
from pypto import ir

from tests.st.distributed.l3_moe_combine import build_l3_moe_combine_program
from tests.st.distributed.l3_moe_reference import MOE_H, MOE_L, MOE_NRANKS, MOE_RECV_MAX


class TestL3MoeCombine:
    def test_moe_combine_identity(self, test_config):
        program = build_l3_moe_combine_program()
        compiled = ir.compile(program, platform=test_config.platform)

        recv_y = torch.randn(MOE_NRANKS, MOE_L, MOE_RECV_MAX, MOE_H, dtype=torch.bfloat16)
        out = torch.zeros_like(recv_y)
        compiled(recv_y, out)
        assert torch.equal(recv_y, out)


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
