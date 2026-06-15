# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: 2-rank broadcast via ``pld.tensor.broadcast`` intrinsic.

Same on-board semantics as ``test_l3_broadcast.py`` — but the InCore body
calls the new composite intrinsic rather than hand-rolling notify/wait/remote_load.

Golden: every rank's output equals root's input.  Non-root inputs must not
appear in outputs.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 64
ROOT_RANK = 0


def _expected_broadcast(inputs: torch.Tensor, root: int = ROOT_RANK) -> torch.Tensor:
    """Root row replicated on every rank."""
    root_row = inputs[root, 0]
    return torch.stack([root_row, root_row]).unsqueeze(1)


def _build_broadcast_program():
    """Build a 2-rank broadcast program using the intrinsic."""

    @pl.program
    class BroadcastIntrinsic:
        @pl.function(type=pl.FunctionType.InCore)
        def broadcast_step(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            # Phase 1: root only stages data.
            if my_rank == ROOT_RANK:
                local = pl.load(inp, [0, 0], [1, SIZE])
                pl.store(local, [0, 0], data)

            # Phases 2-3: barrier + broadcast — one call.
            data = pld.tensor.broadcast(data, signal, root=ROOT_RANK)

            # Stage-out: every rank reads root's data.
            acc = pl.load(data, [0, 0], [1, SIZE])
            return pl.store(acc, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            return self.broadcast_step(inp, out, data, signal, my_rank)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[2, 1, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[2, 1, SIZE], pl.FP32]],
        ) -> pl.Tensor[[2, 1, SIZE], pl.FP32]:
            data_buf = pld.alloc_window_buffer(SIZE * 4)
            signal_buf = pld.alloc_window_buffer(4)

            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
                sig = pld.window(signal_buf, [1, 1], dtype=pl.INT32)
                self.chip_orch(inputs[r], outputs[r], data, sig, r, device=r)
            return outputs

    return BroadcastIntrinsic


class TestL3TensorBroadcastIntrinsic:
    """L3 distributed runtime: broadcast via ``pld.tensor.broadcast``."""

    def test_broadcast(self, test_config, device_ids):
        if len(device_ids) < 2:
            pytest.skip(f"broadcast needs 2 devices, got {device_ids}")

        program = _build_broadcast_program()
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

        expected = _expected_broadcast(inputs)
        assert torch.allclose(outputs, expected), (
            f"broadcast mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )
        assert not torch.allclose(outputs[0], inputs[1]), "non-root input leaked into output"


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
