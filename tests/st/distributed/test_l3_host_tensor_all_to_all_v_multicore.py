# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 ST: HOST-orchestrated ``pld.tensor.all_to_all_v`` with ``core_num > 1``.

RFC #2521 K2. The entry derives the admitted block count
``B = CalAllToAllVBlocks(P, L)`` from the rank count and the requested
``core_num = L``, rejects a signal narrower than ``B``, and launches exactly
``B`` blocks with ``require_sync_start``. This ST is the on-device counterpart
of the unit tests, which only assert the *generated source text* of that
derivation.

What each case proves, and what it does not:

* **Correctness under ``B > 1``** — the personalised exchange still produces the
  right rank-ordered result when more than one block participates. This is the
  main regression barrier: the barrier is addressed per block
  (``peer * stride + block_idx``), so a broken lane scheme hangs or mis-syncs.
* **Every admitted block finished** — after the call each rank reads back the
  ``B`` signal lanes it used and asserts they are all zero. The kernel's credit
  epilogue drops both of a call's credits (one ``AtomicAdd(-2)`` per lane), so a
  lane at zero proves the block owning it started *and* completed its epilogue.
* **Signal reuse across calls** — two back-to-back calls through ONE signal at
  ``B > 1``. Stale credits from round 1 would let round 2's barrier pass
  spuriously and the receive would read peers' data too early, which the output
  check catches. Each round carries a distinct value offset so a stale round-1
  result cannot match its golden.
* **``stride < B`` is rejected before launch** — the negative case. Requesting
  ``L`` that maps to more blocks than the signal is wide must fail explicitly
  rather than hang or silently under-deliver.

Deliberately NOT checked here: **the launch width itself.** Setting
``launch_spec(B)`` is pinned at codegen level by
``test_host_orch_distributed.py``, and the data path cannot distinguish
"launched B" from "launched L and let the tail return", because
``active_blocks = min(block_num, stride)`` makes the extra blocks
indistinguishable from never-launched ones. Observing the actual launch width
needs the runtime's DFX surface — RFC #2521 §13.3's "requested L, launched B,
and active lanes are observable", which is frozen item #10 and not implemented.
Until it is, this ST is the strongest available evidence and not a substitute
for that observation.

ST coverage: P=2 and P=4 (skips when fewer devices are available).
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir import DistributedConfig

SIZE = 64
MAX_RECV = 4


def _admitted_blocks(n_ranks: int, core_num: int) -> int:
    """RFC #2521 frozen item #3's mapping, mirrored here for the expectations.

    Kept as an independent restatement of ``CalAllToAllVBlocks`` so a case's
    expected block count does not come from the code under test.
    """
    if n_ranks <= 0 or core_num <= 0:
        raise ValueError(f"need positive P and L, got P={n_ranks}, L={core_num}")
    return core_num if core_num < n_ranks else (core_num // n_ranks) * n_ranks


def _build_host_all_to_all_v_multicore_program(
    n_ranks: int,
    max_recv: int,
    core_num: int,
    signal_stride: int,
):
    """Build an N-rank HOST-orchestrated ``all_to_all_v`` program at ``core_num`` blocks."""
    nr = n_ranks
    mr = max_recv
    total = nr * mr
    cores = core_num
    stride = signal_stride
    # A [nr, stride] INT32 tile.load fails ptoas 32-byte row alignment whenever
    # `stride * 4` is not a multiple of 32 (stride=2 is 8 bytes), so stage the
    # readback to a padded column count and bound it with valid_shape — the same
    # idiom the multicore allreduce ST uses for its signal readback.
    signal_stage_cols = ((signal_stride + 7) // 8) * 8
    # The lanes the kernel actually uses: `active_blocks = min(block_num, stride)`
    # with `block_num == B` and `stride >= B`, so exactly B.
    admitted = _admitted_blocks(nr, cores)

    @pl.program
    class HostTensorAllToAllVMulticore:
        """N-rank HOST-orchestrated variable-size all-to-all at multiple blocks."""

        @pl.function(type=pl.FunctionType.InCore)
        def stage_step(
            self,
            inp: pl.Tensor[[total, SIZE], pl.FP32],
            stage: pl.Out[pld.DistributedTensor[[total, SIZE], pl.FP32]],
        ):
            for row in pl.range(total):
                chunk = pl.load(inp, [row, 0], [1, SIZE])
                stage = pl.store(chunk, [row, 0], stage)

        @pl.function(type=pl.FunctionType.Orchestration)
        def stage_orch(
            self,
            inp: pl.Tensor[[total, SIZE], pl.FP32],
            stage: pl.Out[pld.DistributedTensor[[total, SIZE], pl.FP32]],
        ):
            self.stage_step(inp, stage)

        @pl.function(type=pl.FunctionType.InCore)
        def fill_counts_step(
            self,
            counts_row: pl.Tensor[[nr, 1], pl.INT32],
            counts: pl.Out[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ):
            # Scalar read/write — a [1,1] INT32 tile.load/store fails ptoas
            # 32-byte row alignment (4 bytes); same pitfall the InCore
            # all_to_all_v intrinsic ST avoids for recv_counts.
            for d in pl.range(nr):
                v = pl.read(counts_row, [d, 0])
                pl.write(counts, [d, 0], v)

        @pl.function(type=pl.FunctionType.Orchestration)
        def fill_counts_orch(
            self,
            counts_row: pl.Tensor[[nr, 1], pl.INT32],
            counts: pl.Out[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ):
            self.fill_counts_step(counts_row, counts)

        @pl.function(type=pl.FunctionType.InCore)
        def consume_step(
            self,
            data: pld.DistributedTensor[[total, SIZE], pl.FP32],
            recv_counts: pld.DistributedTensor[[nr, 1], pl.INT32],
            signal: pld.DistributedTensor[[nr, stride], pl.INT32],
            out: pl.Out[pl.Tensor[[total, SIZE], pl.FP32]],
            recv_out: pl.Out[pl.Tensor[[nr, 1], pl.INT32]],
            signal_out: pl.Out[pl.Tensor[[nr, stride], pl.INT32]],
        ) -> tuple[pl.Tensor[[total, SIZE], pl.FP32], pl.Tensor[[nr, 1], pl.INT32]]:
            # Two loops whose row ranges partition each sender's slot, so every
            # row of the window reaches `out` exactly once and `out` ends up
            # FULLY written. That matters: a `pl.Out` tensor is write-only on
            # the device (the host buffer is never uploaded), so a row the
            # kernel skipped would come back as undefined memory rather than as
            # window content — and the host-side tail check would be inspecting
            # that, not the window.
            #   [base, base + recv_counts[src])       valid -- checked vs golden
            #   [base + recv_counts[src], base + mr)  tail  -- bounded-transfer check
            for src in pl.range(nr):
                n_rows_i32 = pl.read(recv_counts, [src, 0])
                pl.write(recv_out, [src, 0], n_rows_i32)
                n_rows = pl.cast(n_rows_i32, pl.INDEX)
                base = src * mr
                for r in pl.range(n_rows):
                    flat_row = base + r
                    chunk = pl.load(data, [flat_row, 0], [1, SIZE])
                    out = pl.store(chunk, [flat_row, 0], out)
                for r in pl.range(n_rows, mr):
                    flat_row = base + r
                    chunk = pl.load(data, [flat_row, 0], [1, SIZE])
                    out = pl.store(chunk, [flat_row, 0], out)

            # The kernel's credit epilogue drops both credits each admitted
            # block took on its own lane, so a used lane reads back zero once
            # that block has completed. Lanes >= admitted were never touched by
            # the kernel, so they read back zero as well — which is why the
            # host-side assertion covers the used lanes only and cannot, on its
            # own, distinguish B from L blocks.
            # TWAIT performs the cache invalidation required for a reliable
            # device-side observation of a lane another rank wrote.
            ctx = pld.get_comm_ctx(signal)
            my_rank = pld.rank(ctx)
            for peer in pl.range(nr):
                if peer != my_rank:
                    for lane in pl.range(admitted):
                        pld.system.wait(
                            signal=signal,
                            offsets=[peer, lane],
                            expected=0,
                            cmp=pld.WaitCmp.Eq,
                        )

            signal_values = pl.load(
                signal,
                [0, 0],
                [nr, signal_stage_cols],
                valid_shape=[nr, stride],
            )
            pl.store(signal_values, [0, 0], signal_out)
            return out, recv_out

        @pl.function(type=pl.FunctionType.Orchestration)
        def consume_orch(
            self,
            data: pld.DistributedTensor[[total, SIZE], pl.FP32],
            recv_counts: pld.DistributedTensor[[nr, 1], pl.INT32],
            signal: pld.DistributedTensor[[nr, stride], pl.INT32],
            out: pl.Out[pl.Tensor[[total, SIZE], pl.FP32]],
            recv_out: pl.Out[pl.Tensor[[nr, 1], pl.INT32]],
            signal_out: pl.Out[pl.Tensor[[nr, stride], pl.INT32]],
        ) -> tuple[pl.Tensor[[total, SIZE], pl.FP32], pl.Tensor[[nr, 1], pl.INT32]]:
            return self.consume_step(data, recv_counts, signal, out, recv_out, signal_out)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[nr, total, SIZE], pl.FP32],
            send_counts: pl.Tensor[[nr, nr, 1], pl.INT32],
            outputs: pl.Out[pl.Tensor[[nr, total, SIZE], pl.FP32]],
            recv_outputs: pl.Out[pl.Tensor[[nr, nr, 1], pl.INT32]],
            signal_outputs: pl.Out[pl.Tensor[[nr, nr, stride], pl.INT32]],
        ) -> tuple[pl.Tensor[[nr, total, SIZE], pl.FP32], pl.Tensor[[nr, nr, 1], pl.INT32]]:
            input_buf = pld.alloc_window_buffer(total * SIZE * pl.FP32.get_byte())
            data_buf = pld.alloc_window_buffer(total * SIZE * pl.FP32.get_byte())
            # One INT32 per (peer, admitted block); stride >= B is what the
            # entry validates before it launches.
            signal_buf = pld.alloc_window_buffer(nr * stride * pl.INT32.get_byte())
            # Peers pull ONE word per rank from this window (scalar ld_dev read),
            # so the [NR, 1] INT32 vector is the whole requirement — no fixed-
            # width TLOAD unit set.
            counts_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())
            recv_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())

            for r in pl.range(pld.world_size()):
                stage = pld.window(input_buf, [total, SIZE], dtype=pl.FP32)
                self.stage_orch(inputs[r], stage, device=r)

            for r in pl.range(pld.world_size()):
                counts = pld.window(counts_buf, [nr, 1], dtype=pl.INT32)
                self.fill_counts_orch(send_counts[r], counts, device=r)

            stage = pld.window(input_buf, [total, SIZE], dtype=pl.FP32)
            data = pld.window(data_buf, [total, SIZE], dtype=pl.FP32)
            signal = pld.window(signal_buf, [nr, stride], dtype=pl.INT32)
            counts = pld.window(counts_buf, [nr, 1], dtype=pl.INT32)
            recv = pld.window(recv_buf, [nr, 1], dtype=pl.INT32)
            data = pld.tensor.all_to_all_v(stage, data, signal, counts, recv, core_num=cores)

            for r in pl.range(pld.world_size()):
                self.consume_orch(
                    data, recv, signal, outputs[r], recv_outputs[r], signal_outputs[r], device=r
                )

            return outputs, recv_outputs

    return HostTensorAllToAllVMulticore


def _make_inputs(n_ranks: int, max_recv: int, round_offset: float = 0.0) -> torch.Tensor:
    """Rank r sends to dest d: rows ``dest*mr + k`` carrying ``r*1000 + d*100 + k*10 + j%10``.

    Same golden formula as the single-block HOST ST, offset per round so a stale
    earlier round cannot match the current round's expectation.
    """
    nr = n_ranks
    mr = max_recv
    total = nr * mr
    inputs = torch.zeros((nr, total, SIZE), dtype=torch.float32)
    for r in range(nr):
        for d in range(nr):
            n_rows = nr - d
            base = d * mr
            for k in range(n_rows):
                for j in range(SIZE):
                    inputs[r, base + k, j] = round_offset + float(r * 1000 + d * 100 + k * 10 + j % 10)
    return inputs


def _make_send_counts(n_ranks: int) -> torch.Tensor:
    """Rank r sends ``nr - d`` rows to destination d — variable, runtime-dependent."""
    nr = n_ranks
    send_counts = torch.zeros((nr, nr, 1), dtype=torch.int32)
    for r in range(nr):
        for d in range(nr):
            send_counts[r, d, 0] = nr - d
    return send_counts


class TestL3HostTensorAllToAllVMulticore:
    """L3 distributed runtime: HOST ``all_to_all_v`` with ``core_num > 1`` (RFC #2521 K2)."""

    @pytest.mark.parametrize(
        ("n_ranks", "core_num", "signal_stride"),
        [
            pytest.param(2, 2, 2, id="p2-l2-b2-equals-p"),
            pytest.param(2, 5, 4, id="p2-l5-b4-greater-than-p"),
            pytest.param(4, 2, 2, id="p4-l2-b2-less-than-p"),
            pytest.param(4, 10, 8, id="p4-l10-b8-greater-than-p"),
            pytest.param(2, 2, 3, id="p2-l2-b2-stride-wider-than-b"),
        ],
    )
    def test_multicore_output_and_used_lanes(self, test_config, device_ids, n_ranks, core_num, signal_stride):
        """Correct exchange at ``B > 1``, plus every used lane self-cleared to zero."""
        if len(device_ids) < n_ranks:
            pytest.skip(f"multicore host all_to_all_v P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        nr = n_ranks
        mr = MAX_RECV
        total = nr * mr
        admitted = _admitted_blocks(nr, core_num)
        assert signal_stride >= admitted, "case is malformed: stride must admit B"
        label = f"P={nr} L={core_num} B={admitted} stride={signal_stride}"

        compiled = ir.compile(
            _build_host_all_to_all_v_multicore_program(nr, mr, core_num, signal_stride),
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:nr],
                num_sub_workers=0,
            ),
        )

        inputs = _make_inputs(nr, mr)
        send_counts = _make_send_counts(nr)
        outputs = torch.zeros((nr, total, SIZE), dtype=torch.float32)
        recv_outputs = torch.zeros((nr, nr, 1), dtype=torch.int32)
        signal_outputs = torch.zeros((nr, nr, signal_stride), dtype=torch.int32)

        compiled(inputs, send_counts, outputs, recv_outputs, signal_outputs)

        for rank in range(nr):
            for src in range(nr):
                n_rows = int(send_counts[src, rank, 0].item())
                assert int(recv_outputs[rank, src, 0].item()) == n_rows, (
                    f"{label}: rank={rank} src={src}: recv_counts="
                    f"{int(recv_outputs[rank, src, 0].item())} != expected={n_rows}"
                )
                base = src * mr
                for k in range(n_rows):
                    expected_row = inputs[src, rank * mr + k, :]
                    got_row = outputs[rank, base + k, :]
                    assert torch.allclose(got_row, expected_row, atol=1e-5), (
                        f"{label}: rank={rank} src={src} row={k}: "
                        f"max diff = {(got_row - expected_row).abs().max().item()}"
                    )

        # Every lane the kernel used must be back to zero: the credit epilogue
        # drops both credits per lane, so zero proves the owning block started
        # and completed. Lanes >= admitted are untouched and read zero too, so
        # this asserts the used range only.
        for rank in range(nr):
            for peer in range(nr):
                if peer == rank:
                    continue
                used = signal_outputs[rank, peer, :admitted]
                assert torch.all(used == 0), (
                    f"{label}: receiver={rank} sender={peer} lane not self-cleared: "
                    f"got {signal_outputs[rank, peer].tolist()}"
                )

    @pytest.mark.parametrize(
        ("n_ranks", "core_num", "signal_stride"),
        [
            pytest.param(2, 5, 4, id="reuse-p2-l5-b4"),
            pytest.param(4, 10, 8, id="reuse-p4-l10-b8"),
        ],
    )
    def test_multicore_signal_reuse(self, test_config, device_ids, n_ranks, core_num, signal_stride):
        """Two back-to-back calls through ONE signal at ``B > 1``.

        Output correctness is the definitive proof: if the per-lane epilogue
        fails to drop round 1's credits, round 2's barrier passes spuriously on
        stale credits and the receive reads peers' data too early. Each round
        carries a distinct value offset so a stale round-1 result cannot match
        its own golden.
        """
        if len(device_ids) < n_ranks:
            pytest.skip(f"multicore host all_to_all_v P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        nr = n_ranks
        mr = MAX_RECV
        total = nr * mr
        admitted = _admitted_blocks(nr, core_num)
        rounds = 2
        label = f"reuse P={nr} L={core_num} B={admitted} stride={signal_stride}"

        compiled = ir.compile(
            _build_host_all_to_all_v_multicore_program(nr, mr, core_num, signal_stride),
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:nr],
                num_sub_workers=0,
            ),
        )

        send_counts = _make_send_counts(nr)
        for rd in range(rounds):
            inputs = _make_inputs(nr, mr, round_offset=rd * 10000.0)
            outputs = torch.zeros((nr, total, SIZE), dtype=torch.float32)
            recv_outputs = torch.zeros((nr, nr, 1), dtype=torch.int32)
            signal_outputs = torch.zeros((nr, nr, signal_stride), dtype=torch.int32)

            compiled(inputs, send_counts, outputs, recv_outputs, signal_outputs)

            for rank in range(nr):
                for src in range(nr):
                    n_rows = int(send_counts[src, rank, 0].item())
                    assert int(recv_outputs[rank, src, 0].item()) == n_rows, (
                        f"{label} round {rd}: rank={rank} src={src}: recv_counts="
                        f"{int(recv_outputs[rank, src, 0].item())} != expected={n_rows}"
                    )
                    base = src * mr
                    for k in range(n_rows):
                        expected_row = inputs[src, rank * mr + k, :]
                        got_row = outputs[rank, base + k, :]
                        assert torch.allclose(got_row, expected_row, atol=1e-5), (
                            f"{label} round {rd}: rank={rank} src={src} row={k}: "
                            f"max diff = {(got_row - expected_row).abs().max().item()}"
                        )

    @pytest.mark.parametrize(
        ("n_ranks", "core_num", "signal_stride"),
        [
            pytest.param(4, 10, 4, id="p4-l10-b8-stride4-too-narrow"),
            pytest.param(2, 5, 2, id="p2-l5-b4-stride2-too-narrow"),
        ],
    )
    def test_rejects_signal_stride_below_admitted_blocks(
        self, test_config, device_ids, n_ranks, core_num, signal_stride
    ):
        """A signal narrower than ``B`` must fail before launch, not hang.

        Frozen item #9's completion criterion: ``signal stride < B`` is rejected
        before launch. The check lives in the generated entry, which reports a
        fatal runtime argument error and submits no AIV task — so the failure
        surfaces from the dispatch rather than as a hang or as wrong output.
        """
        if len(device_ids) < n_ranks:
            pytest.skip(f"multicore host all_to_all_v P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        nr = n_ranks
        mr = MAX_RECV
        total = nr * mr
        admitted = _admitted_blocks(nr, core_num)
        assert signal_stride < admitted, "case is malformed: stride must be too narrow for B"

        compiled = ir.compile(
            _build_host_all_to_all_v_multicore_program(nr, mr, core_num, signal_stride),
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:nr],
                num_sub_workers=0,
            ),
        )

        inputs = _make_inputs(nr, mr)
        send_counts = _make_send_counts(nr)
        outputs = torch.zeros((nr, total, SIZE), dtype=torch.float32)
        recv_outputs = torch.zeros((nr, nr, 1), dtype=torch.int32)
        signal_outputs = torch.zeros((nr, nr, signal_stride), dtype=torch.int32)

        with pytest.raises(RuntimeError):
            compiled(inputs, send_counts, outputs, recv_outputs, signal_outputs)


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
