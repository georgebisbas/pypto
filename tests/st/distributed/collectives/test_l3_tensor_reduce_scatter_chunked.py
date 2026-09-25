# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: ``pld.tensor.reduce_scatter`` at extents beyond one VEC chunk.

``test_l3_tensor_reduce_scatter_intrinsic.py`` pins the single-tile schedule —
``SIZE = 64``, well under one FP32 chunk — and the reduce-op variants.  This file
covers what that leaves untested:

* extents that need the chunk loop (``SIZE > 4096``), including extents that do
  not divide evenly by the chunk, so the ragged final chunk and its
  ``tile.set_validshape`` narrowing are exercised — and one whose byte width is
  not 32-byte aligned, which the single-tile schedule cannot even allocate;
* calling ``reduce_scatter`` **twice on one signal**, which only completes if the
  first call's epilogue restored the signal generation counter it consumed.
  ``test_l3_tensor_collective_signal_reuse.py`` already covers signal reuse, but
  at ``SIZE = 64``, i.e. the single-tile path whose epilogue subtracts a
  compile-time constant. The chunked path's epilogue subtracts the runtime
  ``1 + ceil(SIZE / chunk_cols)``, and only a chunked extent exercises that.

Only ``sum`` is used: the reduce-op variants are already pinned by the sibling
file, so repeating them here would add cost without coverage.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir import DistributedConfig

# FP32 VEC chunk emitted by the composite lowering (16 KiB tile).
CHUNK = 4096

# (SIZE, label) — every entry exceeds one chunk, so none of them can silently
# fall back to the single-tile path.  They span an extent whose byte width is
# not 32-byte aligned, fewer-than-two chunks with a ragged tail, an exact chunk
# multiple, many chunks with a ragged tail, and the slab width of the motivating
# rail (TP = 4, DECODE_MAX_TOKENS = 192 -> width <= 48, size = width * D).
CHUNKED_EXTENTS = [
    (5120, "sub_two_chunks"),
    (3 * CHUNK, "three_chunks_exact"),
    (25600, "many_chunks_ragged"),
    (48 * 5120, "w48_full_slab"),
    (CHUNK + 1, "unaligned_chunk_plus_one"),
]


def _expected_reduce_scatter(inputs: torch.Tensor, size: int) -> torch.Tensor:
    """Per-rank golden: element-wise sum of chunk r across all ranks.

    Both calls in the program compute this same value, because the program
    re-stages the window from ``inp`` between them.
    """
    n_ranks = inputs.shape[0]
    rows = inputs.reshape(n_ranks, n_ranks, size)
    return rows.sum(dim=0)


def _make_rank_inputs(n_ranks: int, size: int) -> torch.Tensor:
    """Distinct per-rank tensors holding ``n_ranks`` contiguous chunks of ``size``.

    Values are integer-valued and bounded (``< n_ranks * size + offset``), so each
    partial sum stays well inside FP32's exact-integer range: the comparison can be
    bit-exact regardless of the order the kernel accumulates in.
    """
    rows = [
        torch.arange(r * 100.0, r * 100.0 + n_ranks * size, dtype=torch.float32).reshape(1, n_ranks * size)
        for r in range(n_ranks)
    ]
    return torch.stack(rows)


def _build_reduce_scatter_chunked_program(n_ranks: int, size: int):
    """Build the N-rank chunked/reuse program at call time.

    Deferred construction lets this file collect even if the embedded body is
    rejected by the parser.
    """
    nr = n_ranks
    sz = size

    @pl.program
    class ReduceScatterChunkedReuse:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            inp: pl.Tensor[[1, nr * sz], pl.FP32],
            out: pl.Out[pl.Tensor[[2, sz], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[nr, sz], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[2, sz], pl.FP32]:
            # Stage-in: this rank writes chunk j of its input to window row j.
            # Staged in VEC-sized pieces because a row whose byte width is not
            # 32-byte aligned (SIZE = 4097) cannot be allocated as one tile.
            for j in pl.range(nr):
                for c in pl.range(0, sz, CHUNK):
                    staged = pl.load(inp, [0, j * sz + c], [1, CHUNK], [1, pl.min(CHUNK, sz - c)])
                    pl.store(staged, [j, c], data)

            # First collective: after it, rank r's own row holds chunk r reduced
            # across every rank.
            data = pld.tensor.reduce_scatter(data, signal, op=pld.ReduceOp.Sum)
            for c in pl.range(0, sz, CHUNK):
                first = pl.load(data, [my_rank, c], [1, CHUNK], [1, pl.min(CHUNK, sz - c)])
                pl.store(first, [0, c], out)

            # Restore the window so the second call sees exactly the first
            # call's input.  Safe against the collective's WAR barrier: that
            # barrier already guaranteed every peer finished reading row r.
            for j in pl.range(nr):
                for c in pl.range(0, sz, CHUNK):
                    restaged = pl.load(inp, [0, j * sz + c], [1, CHUNK], [1, pl.min(CHUNK, sz - c)])
                    pl.store(restaged, [j, c], data)

            # Second collective on the same signal.  It can only complete if the
            # first call's epilogue reset the signal generation counter, and it
            # must reproduce the first call's result.
            data = pld.tensor.reduce_scatter(data, signal, op=pld.ReduceOp.Sum)
            for c in pl.range(0, sz, CHUNK):
                second = pl.load(data, [my_rank, c], [1, CHUNK], [1, pl.min(CHUNK, sz - c)])
                pl.store(second, [1, c], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, nr * sz], pl.FP32],
            out: pl.Out[pl.Tensor[[2, sz], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[nr, sz], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[2, sz], pl.FP32]:
            return self.reduce_step(inp, out, data, signal, my_rank)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[nr, 1, nr * sz], pl.FP32],
            outputs: pl.Out[pl.Tensor[[nr, 2, sz], pl.FP32]],
        ) -> pl.Tensor[[nr, 2, sz], pl.FP32]:
            data_buf = pld.alloc_window_buffer(nr * sz * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())

            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [nr, sz], dtype=pl.FP32)
                sig = pld.window(signal_buf, [nr, 1], dtype=pl.INT32)
                self.chip_orch(inputs[r], outputs[r], data, sig, r, device=r)
            return outputs

    return ReduceScatterChunkedReuse


class TestL3TensorReduceScatterChunked:
    """L3 distributed runtime: chunked reduce-scatter and its signal-reuse contract."""

    @pytest.mark.parametrize(("size", "label"), CHUNKED_EXTENTS, ids=[c[1] for c in CHUNKED_EXTENTS])
    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_reduce_scatter_chunked_and_reused(self, test_config, device_ids, n_ranks, size, label):
        """Run P=2/P=4 chunked reduce-scatter twice on one signal; skip when devices are scarce."""
        if len(device_ids) < n_ranks:
            pytest.skip(f"reduce-scatter P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        program = _build_reduce_scatter_chunked_program(n_ranks, size)
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )

        inputs = _make_rank_inputs(n_ranks, size)
        outputs = torch.zeros((n_ranks, 2, size), dtype=torch.float32)

        compiled(inputs, outputs)

        expected = _expected_reduce_scatter(inputs, size)
        for call in range(2):
            got = outputs[:, call, :]
            assert torch.equal(got, expected), (
                f"reduce-scatter chunked ({label}, SIZE={size}) P={n_ranks} call {call + 1} mismatch: "
                f"max diff = {(got - expected).abs().max().item()}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
