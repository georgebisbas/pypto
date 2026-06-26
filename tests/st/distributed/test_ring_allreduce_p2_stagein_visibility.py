# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Minimal failing example for the P=2 HCCL stage-in visibility race (issue #1826).

WHAT THIS ISOLATES
==================
The full ring allreduce (``test_l3_ring_allreduce.py``) fails intermittently on
NPU/HCCL at P=2 with corrupted output, and the suspicion is that a rank's first
``pld.tile.remote_load`` reads **stale / partially-staged** data from a peer's
HCCL window — i.e. the single notify/wait barrier between stage-in and the first
remote read does not guarantee the peer's stage-in stores are visible across the
fabric before the read.

This example strips the ring down to *only* that hazard:

    Phase 1 (stage-in):  each rank writes a known, rank-distinct pattern into its
                         own ``scratch`` HCCL window.
    Barrier:             ONE global notify-all / wait-all on signal row 0.
    Fetch:               each rank ``remote_load``s its LEFT neighbour's full
                         staged buffer and writes it to ``out``.

No reduce, no accumulate, no allgather, no multi-round, no chunk arithmetic — so
any mismatch points squarely at "remote_load after one barrier did not observe
the peer's fully-staged window". Golden is a pure neighbour shift:
``out[r] == inputs[(r-1) % P]``.

EXPECTED BEHAVIOUR
==================
* **sim** (``--platform=a2a3sim``): PASSES every run — sim uses malloc+shm with no
  fabric write-visibility reordering, so this only validates that the example is a
  correct, runnable program. A sim pass does NOT exonerate the bug.
* **NPU / HCCL** (``--platform=a2a3``): expected to FAIL **intermittently** at P=2
  if the bug exists. The failure is timing-dependent — run many iterations.

REPRODUCIBILITY INSTRUCTIONS
============================
Build/toolchain: pypto built against the pinned PTOAS (``PTOAS_VERSION`` in
``.github/workflows/ci.yml``; v0.45 at time of writing) and the pinned
``--pto-isa-commit``.

1) Validate the example is correct (sim — should PASS):

    pytest tests/st/distributed/test_ring_allreduce_p2_stagein_visibility.py \
      -k "test_stagein_visibility[2]" --forked \
      --platform=a2a3sim --device=0,1 --pto-isa-commit=<isa-commit>

2) Attempt to reproduce the bug (NPU — expected to FAIL intermittently):

    pytest tests/st/distributed/test_ring_allreduce_p2_stagein_visibility.py \
      -k "test_stagein_visibility[2]" --forked \
      --platform=a2a3 --device=0,1

3) Because the race is intermittent, loop it and count failures (NPU):

    for i in $(seq 1 100); do
      pytest tests/st/distributed/test_ring_allreduce_p2_stagein_visibility.py \
        -k "test_stagein_visibility[2]" --forked --platform=a2a3 --device=0,1 -q \
        2>&1 | grep -qE "1 passed" || echo "FAIL iter $i"
    done

   Any "FAIL iter N" line over ~100 iters reproduces the bug. Zero failures over
   100 iters means this minimal schedule does not surface it on this host (the
   real ring may need more in-flight stores — see "AMPLIFIERS" below).

WHAT A FAILURE LOOKS LIKE
=========================
``out[r]`` differs from ``inputs[(r-1) % P]``: typically the receiver sees its
own data, zeros, or a partial mix of the peer's buffer. The assert message prints
``max diff`` and the first mismatching index.

AMPLIFIERS (knobs to widen the race window if step 3 shows zero failures)
=========================================================================
* Increase ``SIZE`` (more bytes must become visible per remote_load).
* Increase ``STAGE_IN_PIECES`` (more in-flight MTE3 stores at the barrier — closer
  to the real ring's chunked stage-in).
* Add P=4 / P=8 to the parametrize list.

ONCE REPRODUCED
===============
With a confirmed MFE we can A/B the candidate fixes (see #1826 / closed #1827):
a consumer-side acquire in PTOAS after ``TWAIT`` vs. a stronger producer fence vs.
the (closed) duplicate-barrier band-aid.

Related: issue #1826 (bug), PR #1827 (closed band-aid), ``test_l3_ring_allreduce.py``.
"""

# pyright: reportUndefinedVariable=false

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 256  # elements per rank window (matches the ring ST)
STAGE_IN_PIECES = 4  # split stage-in into this many MTE3 stores (in-flight at the barrier)


def _make_rank_inputs(n_ranks: int) -> torch.Tensor:
    """Rank-distinct patterns with a large per-rank base so any stale/partial read is obvious."""
    rows = [
        torch.arange(r * 1000.0, r * 1000.0 + SIZE, dtype=torch.float32).reshape(1, SIZE)
        for r in range(n_ranks)
    ]
    return torch.stack(rows)


def _expected_left_shift(inputs: torch.Tensor) -> torch.Tensor:
    """Each rank receives its LEFT neighbour's staged buffer: out[r] = inputs[(r-1) % P]."""
    n = inputs.shape[0]
    return torch.stack([inputs[(r - 1 + n) % n] for r in range(n)])


def _build_stagein_visibility_program(n_ranks: int):
    """Minimal stage-in → one barrier → remote_load(left) program for the given rank count."""
    piece = SIZE // STAGE_IN_PIECES  # Python int — static tile shape

    @pl.program
    class StageInVisibility:
        """Stage a known pattern, barrier once, then fetch the left neighbour's full buffer."""

        @pl.function(type=pl.FunctionType.InCore)
        def stage_and_fetch(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            scratch: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[1, n_ranks], pl.INT32]],
            piece_elems: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            """stage-in (chunked) → single notify/wait barrier (row 0) → remote_load(left)."""
            ctx = pld.get_comm_ctx(scratch)
            my_rank = pld.rank(ctx)
            nranks = pld.nranks(ctx)
            left = (my_rank - 1 + nranks) % nranks

            # Phase 1: stage-in — copy local input into the scratch HCCL window in pieces.
            for p in pl.range(STAGE_IN_PIECES):
                src = pl.load(inp, [0, p * piece_elems], [1, piece])
                pl.store(src, [0, p * piece_elems], scratch)

            # Single global barrier on signal row 0 — notify-all then wait-all.
            # alloc_window_buffer zero-inits, so AtomicAdd(0->1) / WaitGe(1) is safe.
            for peer in pl.range(nranks):
                if peer != my_rank:
                    pld.system.notify(
                        signal,
                        peer=peer,
                        offsets=[0, my_rank],
                        value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )
            for peer in pl.range(nranks):
                if peer != my_rank:
                    pld.system.wait(
                        signal=signal,
                        offsets=[0, peer],
                        expected=1,
                        cmp=pld.WaitCmp.Ge,
                    )

            # The operation under test: read the left neighbour's full staged buffer.
            recv = pld.tile.remote_load(
                scratch,
                peer=left,
                offsets=[0, 0],
                shape=[1, SIZE],
            )
            pl.store(recv, [0, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            scratch: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[1, n_ranks], pl.INT32]],
            piece_elems: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            """Per-device orchestration wrapper around ``stage_and_fetch``."""
            return self.stage_and_fetch(inp, out, scratch, signal, piece_elems)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[n_ranks, 1, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[n_ranks, 1, SIZE], pl.FP32]],
        ) -> pl.Tensor[[n_ranks, 1, SIZE], pl.FP32]:
            """Launch one chip orchestration per rank with shared window buffers."""
            scratch_buf = pld.alloc_window_buffer(SIZE * 4)  # SIZE × FP32
            signal_buf = pld.alloc_window_buffer(n_ranks * 4)  # 1 row × NR × INT32

            piece_elems = SIZE // STAGE_IN_PIECES
            for r in pl.range(n_ranks):
                scratch = pld.window(scratch_buf, [1, SIZE], dtype=pl.FP32)
                signal = pld.window(signal_buf, [1, n_ranks], dtype=pl.INT32)
                self.chip_orch(
                    inputs[r],
                    outputs[r],
                    scratch,
                    signal,
                    piece_elems,
                    device=r,
                )
            return outputs

    return StageInVisibility


class TestRingAllReduceP2StageInVisibility:
    """Minimal stage-in visibility repro for issue #1826 (P=2 HCCL race)."""

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_stagein_visibility(self, test_config, device_ids, n_ranks):
        """Stage a known pattern, barrier once, fetch left neighbour; assert exact match."""
        if len(device_ids) < n_ranks:
            pytest.skip(f"stage-in visibility P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        program = _build_stagein_visibility_program(n_ranks)
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )

        inputs = _make_rank_inputs(n_ranks)
        outputs = torch.zeros((n_ranks, 1, SIZE), dtype=torch.float32)

        compiled(inputs, outputs)

        expected = _expected_left_shift(inputs)
        if not torch.allclose(outputs, expected):
            diff = (outputs - expected).abs()
            flat = diff.reshape(-1)
            first_bad = int(torch.nonzero(flat)[0].item()) if flat.any() else -1
            pytest.fail(
                f"P={n_ranks} stage-in visibility mismatch: max diff = {diff.max().item()}, "
                f"first mismatching flat index = {first_bad}. "
                f"out[r] should equal inputs[(r-1)%P] (left-neighbour shift). "
                f"A nonzero diff means remote_load did not observe the peer's fully-staged "
                f"window after one notify/wait barrier (issue #1826)."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
