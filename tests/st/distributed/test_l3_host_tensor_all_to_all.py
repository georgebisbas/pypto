# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed ST: host-orchestrator ``pld.tensor.all_to_all`` builtin dispatch."""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 64
NR = pl.dynamic("NR")


def _expected_all_to_all(inputs: torch.Tensor) -> torch.Tensor:
    """Golden matching simpler: output[src, j] = src*1000 + rank*100 + j."""
    nranks = inputs.shape[0]
    outputs = torch.zeros((nranks, nranks, SIZE), dtype=torch.float32)
    for rank in range(nranks):
        for src in range(nranks):
            for j in range(SIZE):
                outputs[rank, src, j] = float(src * 1000 + rank * 100 + j)
    return outputs


def _make_rank_inputs(n_ranks: int) -> torch.Tensor:
    """Each rank r fills input[r, d, j] = r*1000 + d*100 + j (chunk for dest d)."""
    rows = torch.zeros((n_ranks, n_ranks, SIZE), dtype=torch.float32)
    for r in range(n_ranks):
        for d in range(n_ranks):
            for j in range(SIZE):
                rows[r, d, j] = float(r * 1000 + d * 100 + j)
    return rows


@pl.program
class HostTensorAllToAll:
    @pl.function(type=pl.FunctionType.InCore)
    def publish_step(
        self,
        inp: pl.Tensor[[NR, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[NR, SIZE], pl.FP32]],
    ) -> pld.DistributedTensor[[NR, SIZE], pl.FP32]:
        # Stage each destination chunk: input[dest, :] → data[dest, 0]
        for dest in pl.range(NR):
            local = pl.load(inp, [dest, 0], [1, SIZE])
            data = pl.store(local, [dest, 0], data)
        return data

    @pl.function(type=pl.FunctionType.Orchestration)
    def publish_orch(
        self,
        inp: pl.Tensor[[NR, SIZE], pl.FP32],
        data: pl.InOut[pld.DistributedTensor[[NR, SIZE], pl.FP32]],
    ) -> pld.DistributedTensor[[NR, SIZE], pl.FP32]:
        return self.publish_step(inp, data)

    @pl.function(type=pl.FunctionType.InCore)
    def consume_step(
        self,
        data: pld.DistributedTensor[[NR, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[NR, SIZE], pl.FP32]],
    ) -> pl.Tensor[[NR, SIZE], pl.FP32]:
        # Read exchanged result: out[src, :] = data[src, 0]
        for src in pl.range(NR):
            chunk = pl.load(data, [src, 0], [1, SIZE])
            out = pl.store(chunk, [src, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def consume_orch(
        self,
        data: pld.DistributedTensor[[NR, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[NR, SIZE], pl.FP32]],
    ) -> pl.Tensor[[NR, SIZE], pl.FP32]:
        return self.consume_step(data, out)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        inputs: pl.Tensor[[NR, NR, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[NR, NR, SIZE], pl.FP32]],
    ) -> pl.Tensor[[NR, NR, SIZE], pl.FP32]:
        data_buf = pld.alloc_window_buffer(NR * SIZE * 4)
        signal_buf = pld.alloc_window_buffer(pld.world_size() * 4)

        for r in pl.range(pld.world_size()):
            data = pld.window(data_buf, [NR, SIZE], dtype=pl.FP32)
            self.publish_orch(inputs[r], data, device=r)

        data = pld.window(data_buf, [NR, SIZE], dtype=pl.FP32)
        signal = pld.window(signal_buf, [pld.world_size()], dtype=pl.INT32)
        data = pld.tensor.all_to_all(data, signal)

        for r in pl.range(pld.world_size()):
            self.consume_orch(data, outputs[r], device=r)

        return outputs


class TestL3HostTensorAllToAll:
    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_host_tensor_all_to_all(self, test_config, device_ids, n_ranks):
        if len(device_ids) < n_ranks:
            pytest.skip(f"all-to-all P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        compiled = ir.compile(
            HostTensorAllToAll,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )

        inputs = _make_rank_inputs(n_ranks)
        outputs = torch.zeros((n_ranks, n_ranks, SIZE), dtype=torch.float32)

        compiled(inputs, outputs)

        expected = _expected_all_to_all(inputs)
        assert torch.allclose(outputs, expected), (
            f"host all-to-all P={n_ranks} mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
