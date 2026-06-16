# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: N-rank broadcast — 2D ``[NR, SIZE]`` layout.

Uses the **dynamic loop + static tile** pattern with ``NR = pl.dynamic("NR")``.
See ``test_l3_allgather.py`` for the full pattern explanation.

* **Phase 1 (stage-in)** — root writes its data into scratch.
* **Phase 2 (barrier)** — notify-all / wait-all (N-rank mesh).
* **Phase 3 (broadcast)** — each rank remote_loads root's scratch → local output.

Golden: every rank's output equals the root's input.

ST coverage: P=2 and P=4.
"""

# pyright: reportUndefinedVariable=false

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 64  # elements per rank
ROOT_RANK = 0
NR = pl.dynamic("NR")


def _expected_broadcast(inputs: torch.Tensor, root: int = ROOT_RANK) -> torch.Tensor:
    """Every rank gets the root's input.

    inputs shape: [NR, NR, SIZE]
    output shape: [NR, NR, SIZE] — all ranks have root's data
    """
    root_data = inputs[root].clone()
    return torch.stack([root_data] * inputs.shape[0])


def _make_rank_inputs(n_ranks: int) -> torch.Tensor:
    """Only root has meaningful data; others are zeroed."""
    data = torch.zeros((n_ranks, n_ranks, SIZE), dtype=torch.float32)
    for c in range(n_ranks):
        data[ROOT_RANK, c] = torch.arange(c * 100.0, c * 100.0 + SIZE, dtype=torch.float32)
    return data


@pl.program
class BroadcastN:
    """Mesh broadcast with dynamic rank count NR."""

    @pl.function(type=pl.FunctionType.InCore)
    def bcast_step(
        self,
        inp: pl.Tensor[[NR, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[NR, SIZE], pl.FP32]],
        scratch: pl.InOut[pld.DistributedTensor[[NR, SIZE], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
        root: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[NR, SIZE], pl.FP32]:
        """Stage-in (root only) → barrier → remote_load root → store."""
        ctx = pld.get_comm_ctx(scratch)
        my_rank = pld.rank(ctx)
        nranks = pld.nranks(ctx)

        # Phase 1: stage-in — only root writes its chunk into scratch.
        if my_rank == root:
            for c in pl.range(nranks):
                chunk = pl.load(inp, [c, 0], [1, SIZE])  # Tile[1, SIZE] ← static
                pl.store(chunk, [c, 0], scratch)

        # Phase 2: barrier — notify every peer, wait on every peer slot.
        for peer in pl.range(nranks):
            if peer != my_rank:
                pld.system.notify(
                    signal,
                    peer=peer,
                    offsets=[my_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
        for src in pl.range(nranks):
            if src != my_rank:
                pld.system.wait(
                    signal=signal,
                    offsets=[src, 0],
                    expected=1,
                    cmp=pld.WaitCmp.Ge,
                )

        # Phase 3: broadcast — each rank reads all chunks from root's scratch.
        for c in pl.range(nranks):
            recv = pld.tile.remote_load(scratch, peer=root, offsets=[c, 0], shape=[1, SIZE])
            pl.store(recv, [c, 0], out)  # Tile[1, SIZE] ← static

        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch(
        self,
        inp: pl.Tensor[[NR, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[NR, SIZE], pl.FP32]],
        scratch: pl.InOut[pld.DistributedTensor[[NR, SIZE], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
        root: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[NR, SIZE], pl.FP32]:
        """Per-device orchestration wrapper around ``bcast_step``."""
        return self.bcast_step(inp, out, scratch, signal, root)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        inputs: pl.Tensor[[NR, NR, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[NR, NR, SIZE], pl.FP32]],
    ) -> pl.Tensor[[NR, NR, SIZE], pl.FP32]:
        """Launch one chip orchestration per rank with shared window buffers."""
        scratch_buf = pld.alloc_window_buffer(pld.world_size() * SIZE * 4)
        signal_buf = pld.alloc_window_buffer(pld.world_size() * 4)

        for r in pl.range(pld.world_size()):
            scratch = pld.window(scratch_buf, [pld.world_size(), SIZE], dtype=pl.FP32)
            signal = pld.window(signal_buf, [pld.world_size(), 1], dtype=pl.INT32)
            self.chip_orch(
                inputs[r],
                outputs[r],
                scratch,
                signal,
                ROOT_RANK,
                device=r,
            )
        return outputs


class TestL3Broadcast:
    """L3 distributed runtime: N-rank broadcast (2D layout)."""

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_broadcast(self, test_config, device_ids, n_ranks):
        """Compile and run broadcast for P=2 or P=4."""
        if len(device_ids) < n_ranks:
            pytest.skip(f"broadcast P={n_ranks} needs {n_ranks} devices")

        compiled = ir.compile(
            BroadcastN,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )

        inputs = _make_rank_inputs(n_ranks)
        outputs = torch.zeros((n_ranks, n_ranks, SIZE), dtype=torch.float32)
        compiled(inputs, outputs)

        expected = _expected_broadcast(inputs)
        assert torch.allclose(outputs, expected), (
            f"broadcast P={n_ranks} mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
