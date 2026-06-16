# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: dynamic-rank allgather — 2D ``[NR, SIZE]`` layout.

No composite dimensions, no DimExpr IR node — uses bare ``pl.dynamic("NR")``
with ``[NR, SIZE]`` 2D layout. All tile shapes are static ``[1, SIZE]``.

* **Phase 1 (stage-in)** — copy local input into scratch slot.
* **Phase 2 (barrier)** — notify-all / wait-all (matching allreduce pattern).
* **Phase 3 (gather)** — ``pld.tile.remote_load`` each peer's scratch →
  ``pl.store`` into ``out[r, :]``.

Golden: every rank sees rank-ordered concatenation, shape ``[NR, SIZE]``.

ST coverage: P=2 (default CI) and P=4 (any 4-device host). One program for both.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 64  # elements per rank
NR = pl.dynamic("NR")


def _expected_allgather(inputs: torch.Tensor) -> torch.Tensor:
    """Rank-ordered concatenation; identical result on every rank.

    inputs shape: [NR, 1, SIZE]
    output shape: [NR, NR, SIZE] — each rank gets full [NR, SIZE] matrix
    """
    n_ranks = inputs.shape[0]
    gathered = inputs.reshape(n_ranks, SIZE)
    return torch.stack([gathered] * n_ranks)


def _make_rank_inputs(n_ranks: int) -> torch.Tensor:
    """Distinct per-rank tensors so the golden concat is non-trivial."""
    rows = [
        torch.arange(r * 100.0, r * 100.0 + SIZE, dtype=torch.float32).reshape(1, SIZE)
        for r in range(n_ranks)
    ]
    return torch.stack(rows)


@pl.program
class AllGatherDynamic:
    """Dynamic-rank allgather with 2D ``[NR, SIZE]`` layout."""

    @pl.function(type=pl.FunctionType.InCore)
    def gather_step(
        self,
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[NR, SIZE], pl.FP32]],
        scratch: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
    ) -> pl.Tensor[[NR, SIZE], pl.FP32]:
        """Stage-in → barrier → remote_load all peers → store in 2D output."""
        ctx = pld.get_comm_ctx(scratch)
        my_rank = pld.rank(ctx)
        nranks = pld.nranks(ctx)

        # Phase 1: stage-in — copy local input into this rank's scratch slot.
        local = pl.load(inp, [0, 0], [1, SIZE])  # Tile[1, SIZE] ← static
        scratch = pl.store(local, [0, 0], scratch)

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

        # Phase 3: gather — read each rank's scratch, store into out[r, :].
        for r in pl.range(nranks):
            recv = pld.tile.remote_load(scratch, peer=r, offsets=[0, 0], shape=[1, SIZE])
            pl.store(recv, [r, 0], out)  # Tile[1, SIZE] ← static

        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch(
        self,
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[NR, SIZE], pl.FP32]],
        scratch: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
    ) -> pl.Tensor[[NR, SIZE], pl.FP32]:
        """Per-device orchestration wrapper around ``gather_step``."""
        return self.gather_step(inp, out, scratch, signal)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        inputs: pl.Tensor[[NR, 1, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[NR, NR, SIZE], pl.FP32]],
    ) -> pl.Tensor[[NR, NR, SIZE], pl.FP32]:
        """Launch one chip orchestration per rank with shared window buffers."""
        scratch_buf = pld.alloc_window_buffer(SIZE * 4)  # 1×SIZE × FP32
        signal_buf = pld.alloc_window_buffer(pld.world_size() * 4)  # NR×1 × INT32

        for r in pl.range(pld.world_size()):
            scratch = pld.window(scratch_buf, [1, SIZE], dtype=pl.FP32)
            signal = pld.window(signal_buf, [pld.world_size(), 1], dtype=pl.INT32)
            self.chip_orch(
                inputs[r],
                outputs[r],
                scratch,
                signal,
                device=r,
            )
        return outputs


class TestL3AllGather:
    """L3 distributed runtime: dynamic-rank allgather (2D layout, no DimExpr)."""

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_allgather(self, test_config, device_ids, n_ranks):
        """Compile and run allgather for P=2 or P=4; skip when devices are scarce."""
        if len(device_ids) < n_ranks:
            pytest.skip(f"allgather P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        compiled = ir.compile(
            AllGatherDynamic,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )

        inputs = _make_rank_inputs(n_ranks)
        outputs = torch.zeros((n_ranks, n_ranks, SIZE), dtype=torch.float32)
            ]
        compiled(inputs, outputs)

        expected = _expected_allgather(inputs)
        assert torch.allclose(outputs, expected), (
            f"allgather P={n_ranks} mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
