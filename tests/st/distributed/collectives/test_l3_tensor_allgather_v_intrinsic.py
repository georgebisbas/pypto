# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed ST: variable-size all-gather via ``pld.tensor.allgather_v`` intrinsic.

The variable-size counterpart of ``pld.tensor.allgather``, which fixes every
rank's contribution at a compile-time ``[1, SIZE]``. Here each rank contributes
``rows = clamp(send_count, 0, MAX_ROWS)`` rows read at runtime, so the
contribution may be data-dependent — a context-parallel rank's token count, for
example.

Five arguments, flat 2D layouts for ptoas compatibility:
  - ``local_data`` (Tensor [MAX_ROWS, SIZE]) — this rank's rows
  - ``target`` (DistributedTensor [NR*MAX_ROWS, SIZE]) — gathered result window
  - ``signal`` (DistributedTensor INT32 [NR, 1]) — self-clearing barrier
  - ``send_count`` (Tensor INT32 [1, 1]) — this rank's row count, clamped to MAX_ROWS
  - ``recv_counts`` (DistributedTensor INT32 [NR, 1]) — after the barrier,
    ``recv_counts[src, 0]`` holds how many rows ``src`` sent

Unlike ``all_to_all_v`` every peer receives the *same* rows, so there is a single
count rather than a per-destination vector.

What only an execution test can establish, and the lowering UTs cannot: that the
gathered rows land at the right offsets, that ``recv_counts`` publishes what was
actually sent, and that the two-sided clamp holds — an over-capacity count must
not spill into the next rank's slice, and a zero count must still publish 0.

ST coverage: **P=2** (default CI / 2-device hosts) and **P=4** (any four devices).
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 64
MAX_ROWS = 4


def _build_allgather_v_program(n_ranks: int, max_rows: int):
    """Build an N-rank variable-size all-gather program."""
    nr = n_ranks
    mr = max_rows
    total = nr * mr

    @pl.program
    class AllGatherVIntrinsicNRank:
        """N-rank variable-size all-gather with the window-as-result pattern."""

        @pl.function(type=pl.FunctionType.InCore)
        def gather_step(
            self,
            local: pl.Tensor[[mr, SIZE], pl.FP32],
            count: pl.Tensor[[1, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[total, SIZE], pl.FP32]],
            recv_out: pl.Out[pl.Tensor[[nr, 1], pl.INT32]],
            data: pl.InOut[pld.DistributedTensor[[total, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
            recv_counts: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> tuple[pl.Tensor[[total, SIZE], pl.FP32], pl.Tensor[[nr, 1], pl.INT32]]:
            """Push this rank's rows to every peer, barrier, read back by recv_counts."""
            result = pld.tensor.allgather_v(local, data, signal, count, recv_counts)

            # Read back for host-side verification, bounded by the *published*
            # recv_counts rather than a hardcoded formula — that is the intended
            # device-side consumer pattern, and it is what makes the clamp
            # assertions below non-vacuous.
            #
            # Both loops target `out` and their row ranges partition each
            # source's capacity slot, so every row is copied exactly once and
            # `out` ends up FULLY written. That matters: a `pl.Out` tensor is
            # write-only on device (its host buffer is never uploaded), so any
            # row the kernel skipped would read back as undefined memory rather
            # than as window content.
            #   [base, base + recv_counts[src])   valid — checked against golden
            #   [base + recv_counts[src], base+mr) tail — never pushed; the
            #                                      bounded-transfer evidence
            for src in pl.range(nr):
                n_rows_i32 = pl.read(recv_counts, [src, 0])
                # Scalar read/write — a [1,1] INT32 tile.load fails ptoas
                # 32-byte row alignment (4 bytes).
                pl.write(recv_out, [src, 0], n_rows_i32)
                n_rows = pl.cast(n_rows_i32, pl.INDEX)
                base = src * mr
                for r in pl.range(n_rows):
                    flat_row = base + r
                    chunk = pl.load(result, [flat_row, 0], [1, SIZE])
                    pl.store(chunk, [flat_row, 0], out)
                for r in pl.range(n_rows, mr):
                    flat_row = base + r
                    chunk = pl.load(result, [flat_row, 0], [1, SIZE])
                    pl.store(chunk, [flat_row, 0], out)
            return out, recv_out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            local: pl.Tensor[[mr, SIZE], pl.FP32],
            count: pl.Tensor[[1, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[total, SIZE], pl.FP32]],
            recv_out: pl.Out[pl.Tensor[[nr, 1], pl.INT32]],
            data: pl.InOut[pld.DistributedTensor[[total, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
            recv_counts: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> tuple[pl.Tensor[[total, SIZE], pl.FP32], pl.Tensor[[nr, 1], pl.INT32]]:
            """Chip orchestration: dispatch to gather_step with bound windows."""
            return self.gather_step(local, count, out, recv_out, data, signal, recv_counts)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            locals_: pl.Tensor[[nr, mr, SIZE], pl.FP32],
            counts: pl.Tensor[[nr, 1, 1], pl.INT32],
            outputs: pl.Out[pl.Tensor[[nr, total, SIZE], pl.FP32]],
            recv_outputs: pl.Out[pl.Tensor[[nr, nr, 1], pl.INT32]],
        ) -> tuple[pl.Tensor[[nr, total, SIZE], pl.FP32], pl.Tensor[[nr, nr, 1], pl.INT32]]:
            """HOST orchestrator: allocate windows once, loop over ranks."""
            data_buf = pld.alloc_window_buffer(total * SIZE * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())
            recv_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())

            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [total, SIZE], dtype=pl.FP32)
                sig = pld.window(signal_buf, [nr, 1], dtype=pl.INT32)
                recv = pld.window(recv_buf, [nr, 1], dtype=pl.INT32)
                self.chip_orch(
                    locals_[r],
                    counts[r],
                    outputs[r],
                    recv_outputs[r],
                    data,
                    sig,
                    recv,
                    device=r,
                )
            return outputs, recv_outputs

    return AllGatherVIntrinsicNRank


def _run(program, test_config, device_ids, nr, mr, host_counts):
    """Compile and run; returns (locals_, outputs, recv_outputs)."""
    total = nr * mr
    compiled = ir.compile(
        program,
        platform=test_config.platform,
        distributed_config=DistributedConfig(device_ids=device_ids[:nr], num_sub_workers=0),
    )

    # Rank r's row k carries r*1000 + k*10 + j%10, so a misplaced row is
    # traceable to the rank and row it came from rather than just "wrong".
    locals_ = torch.zeros((nr, mr, SIZE), dtype=torch.float32)
    for r in range(nr):
        for k in range(mr):
            for j in range(SIZE):
                locals_[r, k, j] = float(r * 1000 + k * 10 + j % 10)

    counts = torch.zeros((nr, 1, 1), dtype=torch.int32)
    for r in range(nr):
        counts[r, 0, 0] = host_counts[r]

    outputs = torch.zeros((nr, total, SIZE), dtype=torch.float32)
    recv_outputs = torch.zeros((nr, nr, 1), dtype=torch.int32)
    compiled(locals_, counts, outputs, recv_outputs)
    return locals_, outputs, recv_outputs


def _check(nr, mr, host_counts, locals_, outputs, recv_outputs):
    """Every rank must hold every source's clamped rows at src*mr."""
    for r in range(nr):
        expected = max(0, min(int(host_counts[r]), mr))
        for rank in range(nr):
            got = int(recv_outputs[rank, r, 0].item())
            assert got == expected, (
                f"P={nr} rank={rank} src={r}: recv_counts={got} != clamp({host_counts[r]}, 0, {mr})={expected}"
            )
            for k in range(expected):
                got_row = outputs[rank, r * mr + k, :]
                assert torch.allclose(got_row, locals_[r, k, :], atol=1e-5), (
                    f"P={nr} rank={rank} src={r} row={k}: max diff = "
                    f"{(got_row - locals_[r, k, :]).abs().max().item()}"
                )


class TestL3TensorAllGatherVIntrinsic:
    """L3 distributed runtime: variable-size all-gather via ``pld.tensor.allgather_v``."""

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_allgather_v_uneven_counts(self, test_config, device_ids, n_ranks):
        """Every rank contributes a different number of rows, and all agree on the result."""
        if len(device_ids) < n_ranks:
            pytest.skip(f"allgather_v P={n_ranks} needs {n_ranks} devices, got {device_ids}")
        nr, mr = n_ranks, MAX_ROWS
        # 1, 2, ... — at P=4 the last rank sends exactly MAX_ROWS, covering the
        # full-capacity boundary alongside the partial ones.
        host_counts = [(r % mr) + 1 for r in range(nr)]
        program = _build_allgather_v_program(nr, mr)
        locals_, outputs, recv = _run(program, test_config, device_ids, nr, mr, host_counts)
        _check(nr, mr, host_counts, locals_, outputs, recv)

    def test_allgather_v_zero_and_over_capacity_counts_are_clamped(self, test_config, device_ids):
        """A zero count still publishes 0; an over-capacity count is capped at MAX_ROWS.

        The upper clamp is the load-bearing one: without it a count above
        MAX_ROWS would push past this rank's slot into the next rank's, so the
        check that rank 1's rows are intact is what proves the clamp holds.
        """
        nr, mr = 2, MAX_ROWS
        if len(device_ids) < nr:
            pytest.skip(f"allgather_v P={nr} needs {nr} devices, got {device_ids}")
        # rank 0 sends nothing; rank 1 asks for more than capacity.
        host_counts = [0, mr + 3]
        program = _build_allgather_v_program(nr, mr)
        locals_, outputs, recv = _run(program, test_config, device_ids, nr, mr, host_counts)
        _check(nr, mr, host_counts, locals_, outputs, recv)
