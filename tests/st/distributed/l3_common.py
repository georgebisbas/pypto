# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared L3 distributed GEMM / allreduce building blocks for ST program factories.

Single source for cube GEMM InCore, 2-rank ``reduce_step``, window byte sizes,
and host dispatch helpers used by ``l3_gemm``, ``l3_allreduce_gemm``,
``l3_hier_split_k_gemm``, and ``test_l3_allreduce``.
"""

from __future__ import annotations

import pypto.language as pl
import pypto.language.distributed as pld

# Default cube-friendly tile sizes (match Phase 2/3 ST).
M0 = 64
K = 64
N = 64


def window_scratch_nbytes() -> int:
    """Dummy 1x1 INT32 window for CollectCommGroups when no collective runs."""
    return 4


def data_window_nbytes(m0: int, n: int) -> int:
    return m0 * n * 4


def signal_window_nbytes() -> int:
    return 4


def make_cube_gemm(*, m0: int, k: int, n: int):
    """Return an InCore cube GEMM method: ``partial = a_shard @ b``."""

    @pl.function(type=pl.FunctionType.InCore)
    def gemm(
        self,
        a_shard: pl.Tensor[[m0, k], pl.FP32],
        b: pl.Tensor[[k, n], pl.FP32],
        partial: pl.Out[pl.Tensor[[m0, n], pl.FP32]],
    ) -> pl.Tensor[[m0, n], pl.FP32]:
        tile_a_l1 = pl.load(a_shard, offsets=[0, 0], shapes=[m0, k], target_memory=pl.MemorySpace.Mat)
        tile_b_l1 = pl.load(b, offsets=[0, 0], shapes=[k, n], target_memory=pl.MemorySpace.Mat)
        tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
        tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
        tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
        return pl.store(tile_c_l0c, offsets=[0, 0], output_tensor=partial)

    return gemm


def make_reduce_step_2rank(*, m0: int, n: int):
    """Return 4-phase peer allreduce InCore (single ``remote_load``; ``nranks=2``)."""

    @pl.function(type=pl.FunctionType.InCore)
    def reduce_step(
        self,
        inp: pl.Tensor[[m0, n], pl.FP32],
        out: pl.Out[pl.Tensor[[m0, n], pl.FP32]],
        data: pl.InOut[pld.DistributedTensor[[m0, n], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
        peer: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[m0, n], pl.FP32]:
        local = pl.load(inp, [0, 0], [m0, n])
        _ = pl.store(local, [0, 0], data)

        pld.system.notify(
            signal,
            peer=peer,
            offsets=[0, 0],
            value=1,
            op=pld.NotifyOp.AtomicAdd,
        )
        pld.system.wait(
            signal=signal,
            offsets=[0, 0],
            expected=1,
            cmp=pld.WaitCmp.Ge,
        )

        acc = pl.load(data, [0, 0], [m0, n])
        recv = pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[m0, n])
        acc = pl.add(acc, recv)
        return pl.store(acc, [0, 0], out)

    return reduce_step


def build_l3_allreduce_only_program(*, nranks: int, m0: int, n: int):
    """Standalone 2-rank allreduce (no GEMM), ring peer ``(r+1) % nranks``."""

    if nranks != 2:
        raise ValueError(f"build_l3_allreduce_only_program requires nranks=2; got {nranks}")

    inp_shape = [nranks, m0, n]

    @pl.program
    class L3AllReduceOnlyProgram:
        reduce_step = make_reduce_step_2rank(m0=m0, n=n)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[m0, n], pl.FP32],
            out: pl.Out[pl.Tensor[[m0, n], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[m0, n], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[m0, n], pl.FP32]:
            return self.reduce_step(inp, out, data, signal, peer)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[inp_shape, pl.FP32],  # type: ignore[valid-type]
            outputs: pl.Out[pl.Tensor[inp_shape, pl.FP32]],  # type: ignore[valid-type]
        ) -> pl.Tensor[inp_shape, pl.FP32]:  # type: ignore[valid-type]
            data_buf = pld.alloc_window_buffer(data_window_nbytes(m0, n))
            signal_buf = pld.alloc_window_buffer(signal_window_nbytes())

            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [m0, n], dtype=pl.FP32)
                signal = pld.window(signal_buf, [1, 1], dtype=pl.INT32)
                self.chip_orch(
                    inputs[r],
                    outputs[r],
                    data,
                    signal,
                    (r + 1) % pld.world_size(),
                    device=r,
                )
            return outputs

    return L3AllReduceOnlyProgram
