# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: N-rank reduce_scatter — 2D ``[NR, SIZE]`` layout.

Uses the **dynamic loop + static tile** pattern for rank-count polymorphism:

- **Dynamic loops**: ``pl.range(nranks)`` where ``nranks`` comes from
  ``pld.nranks(ctx)`` at runtime — adapts to any rank count.
- **Static tiles**: every ``pl.load`` / ``pl.store`` / ``pld.tile.remote_load``
  uses a fixed shape (``[1, SIZE]``) — tiles are always compile-time constants
  and never depend on the number of ranks.
- **Type annotations**: ``NR = pl.dynamic("NR")`` is a bare ``Var`` in
  parameter type shapes (e.g. ``Tensor[[NR, SIZE]]``), never a composite
  expression (``NR * SIZE``).  The same compiled binary works for P=2 and P=4
  because NR is recovered from the input tensor shape at runtime.

* **Phase 1 (stage-in)** — each rank writes its own column into scratch.
* **Phase 2 (barrier)** — notify-all / wait-all (N-rank mesh).
* **Phase 3 (reduce)** — accumulate every peer's chunk via
  ``pld.tile.remote_load`` + ``pl.add``.
* **Phase 4 (stage-out)** — store reduced accumulator → local output.

Golden: each rank gets its own reduced chunk.

ST coverage: P=2 and P=4. One program body for both.
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
NR = pl.dynamic("NR")


def _expected_reduce_scatter(inputs: torch.Tensor) -> torch.Tensor:
    """Element-wise sum of all rank inputs, scattered to per-rank chunks.

    inputs shape: [NR, NR, SIZE] (each rank has FULL data from all ranks)
    output shape: [NR, 1, SIZE]  (each rank gets its own reduced chunk)
    """
    n_ranks = inputs.shape[0]
    result = torch.zeros((n_ranks, 1, SIZE), dtype=inputs.dtype)
    for r in range(n_ranks):
        result[r, 0] = inputs[:, r, :].sum(dim=0)
    return result


def _make_rank_inputs(n_ranks: int) -> torch.Tensor:
    """Distinct per-rank × per-column tensors."""
    data = torch.zeros((n_ranks, n_ranks, SIZE), dtype=torch.float32)
    for rank in range(n_ranks):
        for col in range(n_ranks):
            base = (rank * n_ranks + col) * 100.0
            data[rank, col] = torch.arange(base, base + SIZE, dtype=torch.float32)
    return data


@pl.program
class ReduceScatterDynamic:
    """Dynamic-rank reduce_scatter with 2D ``[NR, SIZE]`` layout."""

    @pl.function(type=pl.FunctionType.InCore)
    def reduce_step(
        self,
        inp: pl.Tensor[[NR, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
        scratch: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
    ) -> pl.Tensor[[1, SIZE], pl.FP32]:
        """Stage-in → barrier → accumulate peer chunks → store my chunk."""
        ctx = pld.get_comm_ctx(scratch)
        my_rank = pld.rank(ctx)
        nranks = pld.nranks(ctx)

        # Phase 1: stage-in — copy my column into scratch.
        local = pl.load(inp, [my_rank, 0], [1, SIZE])  # Tile[1, SIZE] ← static
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

        # Phase 3: accumulate — load my slice, add every peer's slice via remote_load.
        acc = pl.load(scratch, [0, 0], [1, SIZE])
        for peer in pl.range(nranks):
            if peer != my_rank:
                recv = pld.tile.remote_load(scratch, peer=peer, offsets=[0, 0], shape=[1, SIZE])
                acc = pl.add(acc, recv)

        # Phase 4: stage-out — reduced accumulator → local output.
        return pl.store(acc, [0, 0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch(
        self,
        inp: pl.Tensor[[NR, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
        scratch: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
    ) -> pl.Tensor[[1, SIZE], pl.FP32]:
        """Per-device orchestration wrapper around ``reduce_step``."""
        return self.reduce_step(inp, out, scratch, signal)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        inputs: pl.Tensor[[NR, NR, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[NR, 1, SIZE], pl.FP32]],
    ) -> pl.Tensor[[NR, 1, SIZE], pl.FP32]:
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


class TestL3ReduceScatter:
    """L3 distributed runtime: N-rank reduce_scatter (2D layout, dynamic NR)."""

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_reduce_scatter(self, test_config, device_ids, n_ranks):
        """Compile and run reduce_scatter for P=2 or P=4."""
        if len(device_ids) < n_ranks:
            pytest.skip(f"reduce_scatter P={n_ranks} needs {n_ranks} devices")

        compiled = ir.compile(
            ReduceScatterDynamic,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )

        inputs = _make_rank_inputs(n_ranks)
        outputs = torch.zeros((n_ranks, 1, SIZE), dtype=torch.float32)

        compiled(inputs, outputs)

        expected = _expected_reduce_scatter(inputs)
        assert torch.allclose(outputs, expected), (
            f"reduce_scatter P={n_ranks} mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
