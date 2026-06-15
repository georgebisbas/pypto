# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: 2-rank reduce-scatter via ``pld.tensor.reduce_scatter`` intrinsic.

Same on-board semantics as ``test_l3_reduce_scatter.py``.  Target shape
[NR, SIZE] — each rank stages all NR chunks before the call; after the call
rank r's row holds the element-wise sum of chunk r across all ranks.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 64
NRANKS = 2


def _expected_reduce_scatter(inputs: torch.Tensor) -> torch.Tensor:
    """Per-rank golden: sum of chunk r across all ranks."""
    chunks = [
        inputs[0, 0, r * SIZE : (r + 1) * SIZE] + inputs[1, 0, r * SIZE : (r + 1) * SIZE]
        for r in range(NRANKS)
    ]
    return torch.stack(chunks).reshape(NRANKS, 1, SIZE)


def _build_reduce_scatter_program():
    """Build a 2-rank reduce-scatter program using the intrinsic."""

    @pl.program
    class ReduceScatterIntrinsic:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            inp: pl.Tensor[[1, NRANKS * SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[NRANKS, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[NRANKS, 1], pl.INT32]],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            # Stage-in: write each chunk at its row.
            for j in pl.range(NRANKS):
                chunk = pl.load(inp, [0, j * SIZE], [1, SIZE])
                pl.store(chunk, [j, 0], data)

            # Reduce-scatter — one call.
            data = pld.tensor.reduce_scatter(data, signal, op=pld.ReduceOp.Sum)

            # Stage-out: read my reduced chunk.
            acc = pl.load(data, [my_rank, 0], [1, SIZE])
            return pl.store(acc, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, NRANKS * SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[NRANKS, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[NRANKS, 1], pl.INT32]],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            return self.reduce_step(inp, out, data, signal, my_rank)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[2, 1, NRANKS * SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[2, 1, SIZE], pl.FP32]],
        ) -> pl.Tensor[[2, 1, SIZE], pl.FP32]:
            data_buf = pld.alloc_window_buffer(NRANKS * SIZE * 4)
            signal_buf = pld.alloc_window_buffer(NRANKS * 4)

            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [NRANKS, SIZE], dtype=pl.FP32)
                sig = pld.window(signal_buf, [NRANKS, 1], dtype=pl.INT32)
                self.chip_orch(inputs[r], outputs[r], data, sig, r, device=r)
            return outputs

    return ReduceScatterIntrinsic


class TestL3TensorReduceScatterIntrinsic:
    """L3 distributed runtime: reduce-scatter via ``pld.tensor.reduce_scatter``."""

    def test_reduce_scatter(self, test_config, device_ids):
        if len(device_ids) < 2:
            pytest.skip(f"reduce-scatter needs 2 devices, got {device_ids}")

        program = _build_reduce_scatter_program()
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
                torch.arange(NRANKS * SIZE, dtype=torch.float32).reshape(1, NRANKS * SIZE),
                torch.arange(100.0, 100.0 + NRANKS * SIZE, dtype=torch.float32).reshape(1, NRANKS * SIZE),
            ]
        )
        outputs = torch.zeros((2, 1, SIZE), dtype=torch.float32)

        compiled(inputs, outputs)

        expected = _expected_reduce_scatter(inputs)
        assert torch.allclose(outputs, expected), (
            f"reduce-scatter mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
