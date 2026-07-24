# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed ST: host-orchestrator ``pld.tensor.all_to_all_v`` builtin dispatch.

Validates the HOST-level variable-size all-to-all collective lowers through
``LowerHostTensorCollectives`` and produces correct rank-ordered personalized
exchange via the hand-written ``builtin.tensor.all_to_all_v`` kernel.

The HOST lowering path detects ``pld.tensor.all_to_all_v`` in ``host_orch`` and
lowers it to ``builtin.tensor.all_to_all_v`` per chip.  The exchange uses a
push-based TPUT pattern with TWO DISTINCT windows:

  1. **Stage** (``stage_step``): each rank writes its per-destination variable-size
     chunks into ``stage_buf`` — a flat 2D window [NR*MAX_RECV, SIZE] used ONLY as
     a TPUT source.  Also populates ``signal[dest]`` with the send count for each
     peer.
  2. **All-to-all-v** (``builtin.tensor.all_to_all_v``): kernel reads
     ``signal[dest]`` at runtime and pushes that many rows from ``stage_buf`` to
     each peer's ``data_buf`` window via in-kernel TPUT, then synchronises
     visibility.
  3. **Consume** (``consume_step``): each rank reads back its own ``data_buf``
     window via ``pl.load`` (peers already placed their chunks there via
     in-kernel TPUT).

``stage_buf`` and ``data_buf`` must be separate windows — reusing one buffer
for both roles is a genuine cross-process data race.

ST coverage: P=2 and P=4 (skips when fewer devices are available).
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 64
MAX_RECV = 4


def _build_host_all_to_all_v_program(n_ranks: int, max_recv: int):
    """Build an N-rank host-orchestrator variable-size all-to-all program."""
    nr = n_ranks
    mr = max_recv
    total = nr * mr

    @pl.program
    class HostTensorAllToAllV:
        @pl.function(type=pl.FunctionType.InCore)
        def stage_step(
            self,
            inp: pl.Tensor[[total, SIZE], pl.FP32],
            stage: pl.Out[pld.DistributedTensor[[total, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ):
            # Write per-destination variable-size chunks into the flat staging
            # window and populate signal[dest] with the send count.
            # Each rank sends (nr - dest) rows to peer dest.
            for dest in pl.range(nr):
                n_rows = nr - dest
                # Store the send count in signal[dest, 0].
                signal[dest, 0] = pl.cast(pl.INT32, n_rows)
                for r in pl.range(mr):
                    if r < n_rows:
                        chunk = pl.load(inp, [dest * mr + r, 0], [1, SIZE])
                        stage = pl.store(chunk, [dest * mr + r, 0], stage)
                    else:
                        # Zero-fill remaining rows up to MAX_RECV.
                        zero = pl.zeros([1, SIZE], dtype=pl.FP32)
                        stage = pl.store(zero, [dest * mr + r, 0], stage)

        @pl.function(type=pl.FunctionType.Orchestration)
        def stage_orch(
            self,
            inp: pl.Tensor[[total, SIZE], pl.FP32],
            stage: pl.Out[pld.DistributedTensor[[total, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ):
            self.stage_step(inp, stage, signal)

        @pl.function(type=pl.FunctionType.InCore)
        def consume_step(
            self,
            data: pld.DistributedTensor[[total, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[total, SIZE], pl.FP32]],
        ) -> pl.Tensor[[total, SIZE], pl.FP32]:
            for flat_row in pl.range(total):
                row = pl.load(data, [flat_row, 0], [1, SIZE])
                out = pl.store(row, [flat_row, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def consume_orch(
            self,
            data: pld.DistributedTensor[[total, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[total, SIZE], pl.FP32]],
        ) -> pl.Tensor[[total, SIZE], pl.FP32]:
            return self.consume_step(data, out)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[nr, total, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[nr, total, SIZE], pl.FP32]],
        ) -> pl.Tensor[[nr, total, SIZE], pl.FP32]:
            stage_buf = pld.alloc_window_buffer(total * SIZE * pl.FP32.get_byte())
            data_buf = pld.alloc_window_buffer(total * SIZE * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())

            for r in pl.range(pld.world_size()):
                stage = pld.window(stage_buf, [total, SIZE], dtype=pl.FP32)
                sig = pld.window(signal_buf, [nr, 1], dtype=pl.INT32)
                self.stage_orch(inputs[r], stage, sig, device=r)

            stage = pld.window(stage_buf, [total, SIZE], dtype=pl.FP32)
            data = pld.window(data_buf, [total, SIZE], dtype=pl.FP32)
            signal = pld.window(signal_buf, [nr, 1], dtype=pl.INT32)
            data = pld.tensor.all_to_all_v(stage, data, signal)

            for r in pl.range(pld.world_size()):
                self.consume_orch(data, outputs[r], device=r)

            return outputs

    return HostTensorAllToAllV


class TestL3HostTensorAllToAllV:
    """L3 distributed runtime: HOST-level variable-size all-to-all via builtin dispatch."""

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_host_tensor_all_to_all_v(self, test_config, device_ids, n_ranks):
        """Compile and run host-level variable-size all-to-all for P in {2, 4}.

        Each rank sends ``n_ranks - dest`` rows to each peer (variable counts).
        MAX_RECV=4 is the compile-time capacity.
        """
        if len(device_ids) < n_ranks:
            pytest.skip(f"host all_to_all_v P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        nr = n_ranks
        mr = MAX_RECV
        total = nr * mr

        program = _build_host_all_to_all_v_program(nr, mr)
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:nr],
                num_sub_workers=0,
            ),
        )

        variant_dir = compiled.output_dir / "next_levels" / "builtin.tensor.all_to_all_v__fp32"
        assert variant_dir.is_dir(), f"expected {variant_dir}"
        assert (variant_dir / "kernel_config.py").is_file()

        # Build inputs: 3D host view [nr, total, SIZE] = per-rank flat 2D.
        # Rank r sends to dest d: rows dest*mr+k for k=0..nr-d-1.
        # Value = r*1000 + d*100 + k*10 + j%10.
        inputs = torch.zeros((nr, total, SIZE), dtype=torch.float32)
        for r in range(nr):
            for d in range(nr):
                n_rows = nr - d  # variable send count
                base = d * mr
                for k in range(n_rows):
                    for j in range(SIZE):
                        inputs[r, base + k, j] = float(r * 1000 + d * 100 + k * 10 + j % 10)

        outputs = torch.zeros((nr, total, SIZE), dtype=torch.float32)

        compiled(inputs, outputs)

        # Golden validation:
        # Rank rank receives from src the chunk that src sent to dest=rank.
        # Flat 2D layout: rows src*mr+k hold what src pushed for peer dest=rank.
        for rank in range(nr):
            for src in range(nr):
                n_rows = nr - rank  # what src sent to dest=rank
                base = src * mr
                for k in range(n_rows):
                    expected_row = inputs[src, rank * mr + k, :]
                    got_row = outputs[rank, base + k, :]
                    assert torch.allclose(got_row, expected_row, atol=1e-5), (
                        f"P={nr} rank={rank} src={src} row={k}: "
                        f"max diff = {(got_row - expected_row).abs().max().item()}"
                    )
                # Pad rows beyond n_rows should be zero
                for k in range(n_rows, mr):
                    got_row = outputs[rank, base + k, :]
                    assert torch.allclose(got_row, torch.zeros(SIZE), atol=1e-5), (
                        f"P={nr} rank={rank} src={src} pad row={k}: expected zeros, "
                        f"max = {got_row.abs().max().item()}"
                    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
