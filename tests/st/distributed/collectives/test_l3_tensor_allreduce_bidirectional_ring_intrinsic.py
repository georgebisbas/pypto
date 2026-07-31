# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: N-rank bidirectional ring allreduce via
``pld.tensor.allreduce(mode="bidirectional_ring")``.

The bidirectional ring uses two parallel unidirectional rings on disjoint
halves of the data (ring0 = first half → right neighbour, ring1 = second
half → left neighbour), sharing one O(1) neighbor-only barrier per round.
Same barrier count as the unidirectional ring (2(P−1)+1) but 2x data
throughput per round.

Signal shape: ``[2 * (NR − 1), NR]`` (same as unidirectional ring).
Data shape:  ``[NR, SIZE]`` with ``SIZE % (2*NR) == 0``.

ST coverage: **P=2** and **P=4**, each at several sizes.  Also covers
Sum, Max, Min, Prod ReduceOps at P=4.
"""

import pytest
import torch

import pypto.language as pl
import pypto.language.distributed as pld
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

STAGE_CHUNK = 8192


def _expected_allreduce(inputs: torch.Tensor, op_name: str = "sum") -> torch.Tensor:
    """Replicate the selected element-wise reduction on every rank."""
    if op_name == "sum":
        reduced = inputs.sum(dim=0)
    elif op_name == "max":
        reduced = inputs.max(dim=0).values
    elif op_name == "min":
        reduced = inputs.min(dim=0).values
    elif op_name == "prod":
        reduced = inputs.prod(dim=0)
    else:
        raise ValueError(f"unsupported golden allreduce op: {op_name}")
    return torch.stack([reduced] * inputs.shape[0])


def _make_rank_inputs(
    n_ranks: int,
    size: int,
    *,
    op_name: str = "sum",
    torch_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Distinct per-rank tensors so the golden reduction is non-trivial."""
    if op_name == "prod" or torch_dtype == torch.float16:
        rows = [
            (1.0 + r * 0.125 + torch.arange(size, dtype=torch.float32).remainder(5) * 0.0625).reshape(1, size)
            for r in range(n_ranks)
        ]
    else:
        rows = [
            torch.arange(r * 100.0, r * 100.0 + size, dtype=torch.float32).reshape(1, size)
            for r in range(n_ranks)
        ]
    return torch.stack(rows).to(torch_dtype)


def _build_bidirectional_ring_allreduce_program(
    n_ranks: int,
    size: int,
    *,
    reduce_op: pld.ReduceOp = pld.ReduceOp.Sum,
    dtype=pl.FP32,
    dtype_bytes: int = 4,
):
    """Build an N-rank bidirectional ring allreduce program."""
    nr = n_ranks
    sz = size
    REDUCE_OP = reduce_op
    DTYPE = dtype
    DTYPE_BYTES = dtype_bytes
    total_rounds = 2 * (nr - 1)
    stage_rows = 32 // dtype_bytes if size == 1 else 1
    if size == 1:
        stage_cols = 1
    elif dtype_bytes == 2:
        alignment = 32 // dtype_bytes
        stage_cols = min(STAGE_CHUNK, ((size + alignment - 1) // alignment) * alignment)
    else:
        stage_cols = STAGE_CHUNK

    @pl.program
    class BidirectionalRingAllReduceIntrinsicNRank:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            inp: pl.Tensor[[1, sz], DTYPE],
            out: pl.Out[pl.Tensor[[1, sz], DTYPE]],
            data: pl.InOut[pld.DistributedTensor[[1, sz], DTYPE]],
            signal: pl.InOut[pld.DistributedTensor[[total_rounds, nr], pl.INT32]],
        ) -> pl.Tensor[[1, sz], DTYPE]:
            """One-call bidirectional ring allreduce."""
            # Stage-in: copy local input into HCCL window.
            for col, (data_iter,) in pl.range(0, sz, stage_cols, init_values=(data,)):
                valid = pl.min(stage_cols, sz - col)
                local = pl.load(
                    inp,
                    [0, col],
                    [stage_rows, stage_cols],
                    valid_shapes=[1, valid],
                )
                data_iter = pl.store(local, [0, col], data_iter)
                staged_data = pl.yield_(data_iter)

            data = pld.tensor.allreduce(staged_data, signal, op=REDUCE_OP, mode="bidirectional_ring")

            # Stage-out: copy reduced result to local output.
            for col, (out_iter,) in pl.range(0, sz, stage_cols, init_values=(out,)):
                valid = pl.min(stage_cols, sz - col)
                acc = pl.load(
                    data,
                    [0, col],
                    [stage_rows, stage_cols],
                    valid_shapes=[1, valid],
                )
                out_iter = pl.store(acc, [0, col], out_iter)
                staged_out = pl.yield_(out_iter)
            return staged_out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, sz], DTYPE],
            out: pl.Out[pl.Tensor[[1, sz], DTYPE]],
            data: pl.InOut[pld.DistributedTensor[[1, sz], DTYPE]],
            signal: pl.InOut[pld.DistributedTensor[[total_rounds, nr], pl.INT32]],
        ) -> pl.Tensor[[1, sz], DTYPE]:
            return self.reduce_step(inp, out, data, signal)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[nr, 1, sz], DTYPE],
            outputs: pl.Out[pl.Tensor[[nr, 1, sz], DTYPE]],
        ) -> pl.Tensor[[nr, 1, sz], DTYPE]:
            """Launch one chip orchestration per rank with shared window buffers."""
            data_buf = pld.alloc_window_buffer(sz * DTYPE_BYTES)
            signal_buf = pld.alloc_window_buffer(total_rounds * nr * pl.INT32.get_byte())

            for r in pl.range(nr):
                data = pld.window(data_buf, [1, sz], dtype=DTYPE)
                signal = pld.window(signal_buf, [total_rounds, nr], dtype=pl.INT32)
                self.chip_orch(
                    inputs[r],
                    outputs[r],
                    data,
                    signal,
                    device=r,
                )
            return outputs

    return BidirectionalRingAllReduceIntrinsicNRank


class TestL3TensorBidirectionalRingAllReduceIntrinsic:
    """Sim-only validation of ``pld.tensor.allreduce(mode="bidirectional_ring")``.

    The Pull-based decomposition covers element-wise Sum, Max, Min, and Prod
    via local accumulation, so all four are tested.
    """

    @pytest.mark.parametrize("n_ranks", [2, 4])
    @pytest.mark.parametrize("size", [8, 128, 8192, 65536])
    def test_bidirectional_ring_allreduce_intrinsic(
        self, test_config, device_ids, n_ranks, size
    ):
        if len(device_ids) < n_ranks:
            pytest.skip(
                f"bidirectional ring P={n_ranks} needs {n_ranks} devices, got {device_ids}"
            )
        assert size % (2 * n_ranks) == 0, f"test size {size} not divisible by 2*NR={2 * n_ranks}"

        program = _build_bidirectional_ring_allreduce_program(n_ranks, size)
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )

        inputs = _make_rank_inputs(n_ranks, size)
        outputs = torch.zeros((n_ranks, 1, size), dtype=torch.float32)

        compiled(inputs, outputs)

        expected = _expected_allreduce(inputs)
        assert torch.allclose(outputs, expected), (
            f"bidirectional ring intrinsic P={n_ranks} size={size} mismatch: "
            f"max diff = {(outputs - expected).abs().max().item()}"
        )

    @pytest.mark.parametrize(
        "reduce_op,op_key",
        [
            (pld.ReduceOp.Sum, "sum"),
            (pld.ReduceOp.Max, "max"),
            (pld.ReduceOp.Min, "min"),
            (pld.ReduceOp.Prod, "prod"),
        ],
    )
    def test_bidirectional_ring_reduce_ops(
        self, test_config, device_ids, reduce_op, op_key
    ):
        """Bidirectional ring with every supported ReduceOp.

        The pull-based composite lowering accumulates locally, so all four
        ReduceOps are supported (unlike the simpler push-based kernel which
        is Sum-only due to TPUT AtomicAdd).
        """
        n_ranks = 4
        size = 128
        if len(device_ids) < n_ranks:
            pytest.skip(
                f"bidirectional ring P={n_ranks} needs {n_ranks} devices, got {device_ids}"
            )

        program = _build_bidirectional_ring_allreduce_program(
            n_ranks, size, reduce_op=reduce_op
        )
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:n_ranks],
                num_sub_workers=0,
            ),
        )

        inputs = _make_rank_inputs(n_ranks, size, op_name=op_key)
        outputs = torch.zeros((n_ranks, 1, size), dtype=torch.float32)

        compiled(inputs, outputs)

        expected = _expected_allreduce(inputs, op_key)
        assert torch.allclose(outputs, expected), (
            f"bidirectional ring {op_key} P={n_ranks} size={size} mismatch: "
            f"max diff = {(outputs - expected).abs().max().item()}"
        )
