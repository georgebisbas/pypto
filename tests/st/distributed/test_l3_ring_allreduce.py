# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: N-rank ring allreduce — chunked reduce-scatter + allgather schedule.

Ports the ring algorithm from simpler ``allreduce_ring_distributed``
([#975](https://github.com/hw-native-sys/simpler/pull/975)) into the PyPTO DSL,
using the same ``NR``-dynamic program pattern as ``test_l3_allreduce.py``
([#1668](https://github.com/hw-native-sys/pypto/pull/1668)).

Schedule (NCCL-style RS + AG, single monolithic InCore kernel):
* **Phase 1 (stage-in)** — partition each rank's input into ``NR`` equal
  chunks in the HCCL-window ``chunks`` buffer.
* **Phase 2 (reduce-scatter)** — ``(NR−1)`` ring steps. Per step ``s ∈ [1, NR−1]``:
  publish send chunk to ``exchange`` buffer, barrier (notify-all / wait-all
  on per-round signal row), ``pld.tile.remote_load`` left neighbour's
  ``chunks[(left − s + NR) % NR]``, ``pl.add`` into local
  ``chunks[(r − s − 1 + NR) % NR]``.
* **Phase 3 (allgather)** — ``(NR−1)`` ring steps. Per step:
  publish to ``exchange``, barrier, ``pld.tile.remote_load`` left neighbour's
  ``chunks[(left − s + 1 + NR) % NR]``, store into local
  ``chunks[(r − s + NR) % NR]``.
* **Phase 4 (stage-out)** — concatenate ``chunks[0..NR−1]`` → local ``out``.

Golden: ``outputs[r] == sum(inputs[*])`` for every rank ``r`` (same element-wise
sum as mesh allreduce — ring is a different schedule, not a different result).

Monolithic InCore kernel (``ring_allreduce``) collapses all 4 phases into a single
device-side entry point, eliminating 2×(P−1) AIV task dispatches per rank vs a
multi-kernel chain.  Barrier notify-all / wait-all uses separate signal rows per
round (``signal`` shape ``[2*(NR−1), NR]`` int32).  The ``exchange`` buffer
(1×CHUNK FP32) provides an explicit commit-point before each barrier, matching
simpler's ``allreduce_ring_kernel.cpp`` publish pattern.

ST coverage: **P=2** (default CI / 2-device hosts) and **P=4** (any four
devices, e.g. ``--device=0,1,2,3`` or ``--device=0-3``). One program body
for both.
"""

# pyright: reportUndefinedVariable=false

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 256  # ALLREDUCE_COUNT in simpler allreduce_ring_kernel.cpp
NR = pl.dynamic("NR")
CHUNK = SIZE // NR  # dynamic — resolves to SIZE // nranks at compile time


def _expected_ring_allreduce(inputs: torch.Tensor) -> torch.Tensor:
    """Replicate the element-wise sum of all rank inputs on every rank."""
    reduced = inputs.sum(dim=0)
    return torch.stack([reduced] * inputs.shape[0])


def _make_rank_inputs(n_ranks: int) -> torch.Tensor:
    """Distinct per-rank tensors so the golden sum is non-trivial."""
    rows = [
        torch.arange(r * 100.0, r * 100.0 + SIZE, dtype=torch.float32).reshape(1, SIZE)
        for r in range(n_ranks)
    ]
    return torch.stack(rows)


@pl.program
class RingAllReduce:
    """Ring allreduce with dynamic rank count ``NR`` — monolithic InCore kernel."""

    @pl.function(type=pl.FunctionType.InCore)
    def ring_allreduce(
        self,
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
        chunks: pl.InOut[pld.DistributedTensor[[NR, CHUNK], pl.FP32]],
        exchange: pl.InOut[pld.DistributedTensor[[1, CHUNK], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[2 * (NR - 1), NR], pl.INT32]],
    ) -> pl.Tensor[[1, SIZE], pl.FP32]:
        """Four-phase ring allreduce: stage-in → RS loop → AG loop → stage-out."""
        ctx = pld.get_comm_ctx(chunks)
        my_rank = pld.rank(ctx)
        nranks = pld.nranks(ctx)
        left = (my_rank - 1 + nranks) % nranks

        # ------------------------------------------------------------------
        # Phase 1: stage-in — partition local input into NR chunk slots.
        # ------------------------------------------------------------------
        for c in pl.range(nranks):
            chunk_data = pl.load(inp, [0, c * CHUNK], [1, CHUNK])
            pl.store(chunk_data, [c, 0], chunks)

        # ------------------------------------------------------------------
        # Phase 2: reduce-scatter — (NR−1) ring steps.
        # Rank r ends with fully reduced chunk at index r.
        # Index math matches allreduce_ring_kernel.cpp:
        #   send_idx      = (r − s + NR) % NR
        #   recv_add_idx  = (r − s − 1 + NR) % NR
        #   left_send_idx = (left − s + NR) % NR
        # ------------------------------------------------------------------
        for s in pl.range(1, nranks):
            send_idx = (my_rank - s + nranks) % nranks
            recv_add_idx = (my_rank - s - 1 + nranks) % nranks
            left_send_idx = (left - s + nranks) % nranks
            round_row = s - 1  # RS rounds occupy rows [0 .. NR-2]

            # Publish send chunk to exchange — commit before barrier.
            send_data = pl.load(chunks, [send_idx, 0], [1, CHUNK])
            pl.store(send_data, [0, 0], exchange)

            # Barrier: notify every peer, wait on every peer slot.
            for peer in pl.range(nranks):
                if peer != my_rank:
                    pld.system.notify(
                        signal,
                        peer=peer,
                        offsets=[round_row, my_rank],
                        value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )
            for src in pl.range(nranks):
                if src != my_rank:
                    pld.system.wait(
                        signal=signal,
                        offsets=[round_row, src],
                        expected=1,
                        cmp=pld.WaitCmp.Ge,
                    )

            # Remote-load left peer's send chunk, accumulate into local recv_add chunk.
            recv = pld.tile.remote_load(chunks, peer=left, offsets=[left_send_idx, 0], shape=[1, CHUNK])
            acc = pl.load(chunks, [recv_add_idx, 0], [1, CHUNK])
            acc = pl.add(acc, recv)
            pl.store(acc, [recv_add_idx, 0], chunks)

        # ------------------------------------------------------------------
        # Phase 3: allgather — (NR−1) ring steps.
        # Every rank collects all reduced chunks from left neighbour.
        # Index math:
        #   send_idx      = (r − s + 1 + NR) % NR
        #   recv_idx      = (r − s + NR) % NR
        #   left_send_idx = (left − s + 1 + NR) % NR
        # ------------------------------------------------------------------
        for s in pl.range(1, nranks):
            send_idx = (my_rank - s + 1 + nranks) % nranks
            recv_idx = (my_rank - s + nranks) % nranks
            left_send_idx = (left - s + 1 + nranks) % nranks
            round_row = (nranks - 1) + (s - 1)  # AG rounds occupy rows [NR-1 .. 2*(NR-1)-1]

            # Publish send chunk to exchange — commit before barrier.
            send_data = pl.load(chunks, [send_idx, 0], [1, CHUNK])
            pl.store(send_data, [0, 0], exchange)

            # Barrier: notify every peer, wait on every peer slot.
            for peer in pl.range(nranks):
                if peer != my_rank:
                    pld.system.notify(
                        signal,
                        peer=peer,
                        offsets=[round_row, my_rank],
                        value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )
            for src in pl.range(nranks):
                if src != my_rank:
                    pld.system.wait(
                        signal=signal,
                        offsets=[round_row, src],
                        expected=1,
                        cmp=pld.WaitCmp.Ge,
                    )

            # Remote-load left peer's send chunk, store into local recv chunk.
            recv = pld.tile.remote_load(chunks, peer=left, offsets=[left_send_idx, 0], shape=[1, CHUNK])
            pl.store(recv, [recv_idx, 0], chunks)

        # ------------------------------------------------------------------
        # Phase 4: stage-out — concatenate reduced chunks → local output.
        # ------------------------------------------------------------------
        for c in pl.range(nranks):
            chunk_data = pl.load(chunks, [c, 0], [1, CHUNK])
            out = pl.store(chunk_data, [0, c * CHUNK], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch(
        self,
        inp: pl.Tensor[[1, SIZE], pl.FP32],
        out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
        chunks: pl.InOut[pld.DistributedTensor[[NR, CHUNK], pl.FP32]],
        exchange: pl.InOut[pld.DistributedTensor[[1, CHUNK], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[2 * (NR - 1), NR], pl.INT32]],
    ) -> pl.Tensor[[1, SIZE], pl.FP32]:
        """Per-device orchestration — single monolithic InCore dispatch."""
        return self.ring_allreduce(inp, out, chunks, exchange, signal)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        inputs: pl.Tensor[[NR, 1, SIZE], pl.FP32],
        outputs: pl.Out[pl.Tensor[[NR, 1, SIZE], pl.FP32]],
    ) -> pl.Tensor[[NR, 1, SIZE], pl.FP32]:
        """Launch one chip orchestration per rank with shared window buffers."""
        nranks = pld.world_size()
        chunks_buf = pld.alloc_window_buffer(nranks * CHUNK * 4)  # NR x CHUNK x FP32
        exchange_buf = pld.alloc_window_buffer(CHUNK * 4)  # 1 x CHUNK x FP32 (publish area)
        signal_buf = pld.alloc_window_buffer(2 * (nranks - 1) * nranks * 4)  # 2*(NR-1) x NR x INT32

        for r in pl.range(nranks):
            chunks = pld.window(chunks_buf, [nranks, CHUNK], dtype=pl.FP32)
            exchange = pld.window(exchange_buf, [1, CHUNK], dtype=pl.FP32)
            signal = pld.window(signal_buf, [2 * (nranks - 1), nranks], dtype=pl.INT32)
            self.chip_orch(inputs[r], outputs[r], chunks, exchange, signal, device=r)
        return outputs


class TestL3RingAllReduce:
    """L3 distributed runtime: N-rank ring allreduce (RS+AG schedule)."""

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_ring_allreduce(self, test_config, device_ids, n_ranks):
        """Compile and run ring allreduce for P=2 or P=4; skip when devices are scarce."""
        if len(device_ids) < n_ranks:
            pytest.skip(f"ring allreduce P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        compiled = ir.compile(
            RingAllReduce,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )

        inputs = _make_rank_inputs(n_ranks)
        outputs = torch.zeros((n_ranks, 1, SIZE), dtype=torch.float32)

        compiled(inputs, outputs)

        expected = _expected_ring_allreduce(inputs)
        assert torch.allclose(outputs, expected), (
            f"ring allreduce P={n_ranks} mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
