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


def _build_a2a_program(n_ranks: int):
    """Build an N-rank program identical to the InCore ST pattern."""
    nr = n_ranks

    @pl.program
    class HostTensorAllToAll:
        @pl.function(type=pl.FunctionType.InCore)
        def exchange_step(
            self,
            inp: pl.Tensor[[nr, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[nr, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pl.Tensor[[nr, SIZE], pl.FP32]:
            result = pld.tensor.all_to_all(inp, data, signal, out)
            return result

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[nr, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[nr, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pl.Tensor[[nr, SIZE], pl.FP32]:
            return self.exchange_step(inp, out, data, signal)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[nr, nr, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[nr, nr, SIZE], pl.FP32]],
        ) -> pl.Tensor[[nr, nr, SIZE], pl.FP32]:
            data_buf = pld.alloc_window_buffer(nr * SIZE * 4)
            signal_buf = pld.alloc_window_buffer(nr * 4)

            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [nr, SIZE], dtype=pl.FP32)
                sig = pld.window(signal_buf, [nr, 1], dtype=pl.INT32)
                self.chip_orch(inputs[r], outputs[r], data, sig, device=r)
            return outputs

    return HostTensorAllToAll


class TestL3HostTensorAllToAll:
    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_host_tensor_all_to_all(self, test_config, device_ids, n_ranks):
        if len(device_ids) < n_ranks:
            pytest.skip(f"all-to-all P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        program = _build_a2a_program(n_ranks)
        compiled = ir.compile(
            program,
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


def _build_a2a_builtin_program(n_ranks: int):
    """Build an N-rank program that exercises the 2-arg HOST builtin path.

    Pattern matches the allreduce HOST test: publish (stage input into the
    HCCL window), call the 2-arg builtin (barrier + exchange), then consume
    (read results back from the window).  Each device's HCCL window is
    private, so the kernel writes exchange results back to ``data[src*C]``
    in-place without cross-device write races.
    """
    nr = n_ranks

    @pl.program
    class HostTensorAllToAllBuiltin:
        @pl.function(type=pl.FunctionType.InCore)
        def publish_step(
            self,
            inp: pl.Tensor[[nr, SIZE], pl.FP32],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
        ) -> pld.DistributedTensor[[nr, SIZE], pl.FP32]:
            for d in pl.range(nr):
                local = pl.load(inp, [d, 0], [1, SIZE])
                data = pl.store(local, [d, 0], data)
            return data

        @pl.function(type=pl.FunctionType.Orchestration)
        def publish_orch(
            self,
            inp: pl.Tensor[[nr, SIZE], pl.FP32],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
        ) -> pld.DistributedTensor[[nr, SIZE], pl.FP32]:
            return self.publish_step(inp, data)

        @pl.function(type=pl.FunctionType.InCore)
        def consume_step(
            self,
            data: pld.DistributedTensor[[nr, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[nr, SIZE], pl.FP32]],
        ) -> pl.Tensor[[nr, SIZE], pl.FP32]:
            for src in pl.range(nr):
                row = pl.load(data, [src, 0], [1, SIZE])
                out = pl.store(row, [src, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def consume_orch(
            self,
            data: pld.DistributedTensor[[nr, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[nr, SIZE], pl.FP32]],
        ) -> pl.Tensor[[nr, SIZE], pl.FP32]:
            return self.consume_step(data, out)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[nr, nr, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[nr, nr, SIZE], pl.FP32]],
        ) -> pl.Tensor[[nr, nr, SIZE], pl.FP32]:
            data_buf = pld.alloc_window_buffer(nr * SIZE * 4)
            signal_buf = pld.alloc_window_buffer(nr * 4)

            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [nr, SIZE], dtype=pl.FP32)
                self.publish_orch(inputs[r], data, device=r)

            data = pld.window(data_buf, [nr, SIZE], dtype=pl.FP32)
            signal = pld.window(signal_buf, [nr], dtype=pl.INT32)
            data = pld.tensor.all_to_all(data, signal)

            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [nr, SIZE], dtype=pl.FP32)
                self.consume_orch(data, outputs[r], device=r)

            return outputs

    return HostTensorAllToAllBuiltin


class TestL3HostTensorAllToAllBuiltin:
    """L3 distributed runtime: 2-arg HOST builtin ``pld.tensor.all_to_all(data, signal)``.

    Exercises the builtin kernel template via the HOST lowering path.  The
    host orchestrator stages input into the per-device private HCCL window,
    calls the 2-arg builtin (barrier + exchange), then reads results back.
    Validates that the exchange produces correct personalized output per rank.
    """

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_host_tensor_all_to_all_builtin(self, test_config, device_ids, n_ranks):
        if len(device_ids) < n_ranks:
            pytest.skip(f"all-to-all builtin P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        program = _build_a2a_builtin_program(n_ranks)
        compiled = ir.compile(
            program,
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
            f"host all-to-all builtin P={n_ranks} mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
