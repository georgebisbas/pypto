# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: 2-rank peer-swap via ``pld.tensor.barrier`` intrinsic.

Validates that the composite barrier correctly serialises cross-rank
access: each rank writes its own data, calls ``pld.tensor.barrier(signal)``,
then reads the peer's data.  Without the barrier, the read could observe
stale / zero data.  With the intrinsic, the peer swap is guaranteed.

Golden: ``outputs[0] == inputs[1]`` and ``outputs[1] == inputs[0]``.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 64


def _expected_peer_swap(inputs: torch.Tensor) -> torch.Tensor:
    """Rank 0 gets rank 1's input, rank 1 gets rank 0's input."""
    return torch.stack([inputs[1], inputs[0]])


def _build_barrier_peer_swap_program():
    """Build a 2-rank peer-swap program using the barrier intrinsic."""

    @pl.program
    class BarrierPeerSwap:
        @pl.function(type=pl.FunctionType.InCore)
        def swap_step(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            # Stage-in: write my data into the window.
            local = pl.load(inp, [0, 0], [1, SIZE])
            pl.store(local, [0, 0], data)

            # Barrier — ensure both ranks have staged before anyone reads.
            signal = pld.tensor.barrier(signal)

            # Read peer's data after the barrier guarantees it's staged.
            recv = pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[1, SIZE])
            return pl.store(recv, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            return self.swap_step(inp, out, data, signal, peer)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[2, 1, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[2, 1, SIZE], pl.FP32]],
        ) -> pl.Tensor[[2, 1, SIZE], pl.FP32]:
            data_buf = pld.alloc_window_buffer(SIZE * 4)  # 1xSIZE x FP32
            signal_buf = pld.alloc_window_buffer(4)  # 1x1 x INT32

            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
                sig = pld.window(signal_buf, [1, 1], dtype=pl.INT32)
                self.chip_orch(
                    inputs[r],
                    outputs[r],
                    data,
                    sig,
                    (r + 1) % pld.world_size(),
                    device=r,
                )
            return outputs

    return BarrierPeerSwap


class TestL3TensorBarrierIntrinsic:
    """L3 distributed runtime: peer swap via ``pld.tensor.barrier``."""

    def test_barrier_peer_swap(self, test_config, device_ids):
        if len(device_ids) < 2:
            pytest.skip(f"barrier peer-swap needs 2 devices, got {device_ids}")

        program = _build_barrier_peer_swap_program()
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:2],
                num_sub_workers=0,
            ),
        )

        # Rank 0: [0, 1, …, SIZE-1]; Rank 1: [100, 101, …].
        inputs = torch.stack(
            [
                torch.arange(SIZE, dtype=torch.float32).reshape(1, SIZE),
                torch.arange(100.0, 100.0 + SIZE, dtype=torch.float32).reshape(1, SIZE),
            ]
        )
        outputs = torch.zeros((2, 1, SIZE), dtype=torch.float32)

        compiled(inputs, outputs)

        expected = _expected_peer_swap(inputs)
        assert torch.allclose(outputs, expected), (
            f"barrier peer-swap mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )
        # Sanity: outputs are NOT the identity mapping.
        assert not torch.allclose(outputs[0], inputs[0]), "rank 0 still has its own input"
        assert not torch.allclose(outputs[1], inputs[1]), "rank 1 still has its own input"


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
