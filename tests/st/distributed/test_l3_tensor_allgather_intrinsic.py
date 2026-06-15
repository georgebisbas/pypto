# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: 2-rank allgather via ``pld.tensor.allgather`` intrinsic.

Validates the composite allgather intrinsic produces the same rank-ordered
concatenation on every rank as the hand-written ``test_l3_allgather.py``.

The intrinsic accepts three arguments: ``local_data`` (Tile [1, SIZE]),
``target`` (DistributedTensor [NR, SIZE] staging window), and ``signal``.
It handles stage-in internally, synchronises, remote-loads peers, and
returns the concatenated result as a Tile.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 64
NR = 2  # static for this test


def _expected_allgather(inputs: torch.Tensor) -> torch.Tensor:
    """Rank-ordered concatenation; identical on every rank."""
    gathered = torch.cat([inputs[r, 0] for r in range(inputs.shape[0])])
    return torch.stack([gathered, gathered]).unsqueeze(1)


def _build_allgather_program():
    """Build a 2-rank allgather program using the intrinsic."""

    @pl.program
    class AllGatherIntrinsic:
        @pl.function(type=pl.FunctionType.InCore)
        def gather_step(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, NR * SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[NR, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, NR * SIZE], pl.FP32]:
            # Prepare local chunk as a Tile.
            chunk = pl.load(inp, [0, 0], [1, SIZE])

            # Allgather — intrinsic handles stage-in, sync, remote-loads.
            gathered = pld.tensor.allgather(chunk, data, signal)

            # Stage-out: store the concatenated Tile to output.
            return pl.store(gathered, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, NR * SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[NR, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, NR * SIZE], pl.FP32]:
            return self.gather_step(inp, out, data, signal, my_rank)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[2, 1, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[2, 1, NR * SIZE], pl.FP32]],
        ) -> pl.Tensor[[2, 1, NR * SIZE], pl.FP32]:
            data_buf = pld.alloc_window_buffer(NR * SIZE * 4)
            signal_buf = pld.alloc_window_buffer(NR * 4)

            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [NR, SIZE], dtype=pl.FP32)
                sig = pld.window(signal_buf, [NR, 1], dtype=pl.INT32)
                self.chip_orch(inputs[r], outputs[r], data, sig, r, device=r)
            return outputs

    return AllGatherIntrinsic


class TestL3TensorAllGatherIntrinsic:
    """L3 distributed runtime: allgather via ``pld.tensor.allgather``."""

    def test_allgather(self, test_config, device_ids):
        if len(device_ids) < 2:
            pytest.skip(f"allgather needs 2 devices, got {device_ids}")

        program = _build_allgather_program()
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
        outputs = torch.zeros((2, 1, NR * SIZE), dtype=torch.float32)

        compiled(inputs, outputs)

        expected = _expected_allgather(inputs)
        assert torch.allclose(outputs, expected), (
            f"allgather mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
