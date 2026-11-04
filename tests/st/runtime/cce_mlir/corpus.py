# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Kernel corpus for the PTOAS-versus-cce-mlir performance matrix.

Every kernel here is written on the user-facing ``@pl.jit`` surface and is
constrained to what the cce-mlir bridge can lower today: rank-2, statically
shaped, single-block, no ``while``/``if``, and no padded on-chip buffers. The
shared gate is ``SelectBufferIRFunctions`` (rank-2 static tensors/tiles, one
slot) followed by ``LowerTileToBuffer`` (a Buffer transfer recipe must exist).

The matrix in ``test_cce_mlir_performance.py`` compiles each of these through
both routes and reports the AICore task time, so a kernel that stops lowering
to Buffer IR is a matrix regression, not just a cce-mlir one.

Each kernel exposes ``__expected__`` metadata through the per-case table in the
test module rather than here; the tolerance and the golden multiplier live with
the parametrization so a single kernel can be exercised at different trip
counts.
"""

import pypto.language as pl

__all__ = [
    "MM_KC",
    "MM_K",
    "matmul_accum",
    "matmul_kloop",
    "matmul_kloop_pipelined",
    "matmul_single",
    "mm_then_scale",
    "vector_add",
    "vector_scale_add",
]

# Tiled matmul: 128x128 output, K reduced in chunks of MM_KC.
MM_K = 512
MM_KC = 128

_ACC_TRIPS = 64


@pl.jit
def matmul_single(
    a: pl.Tensor[[128, 128], pl.FP16],
    b: pl.Tensor[[128, 128], pl.FP16],
    c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    """One 128x128x128 MMAD: the smallest cube kernel the route supports."""
    with pl.at(level=pl.Level.CORE_GROUP):
        ta = pl.load(a, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
        tb = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
        la = pl.move(ta, target_memory=pl.MemorySpace.Left)
        lb = pl.move(tb, target_memory=pl.MemorySpace.Right)
        acc = pl.matmul(la, lb, out_dtype=pl.FP32)
        pl.store(acc, [0, 0], c)
    return c


@pl.jit
def matmul_accum(
    a: pl.Tensor[[128, 128], pl.FP16],
    b: pl.Tensor[[128, 128], pl.FP16],
    c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    """One MMAD followed by ``_ACC_TRIPS`` accumulating MMADs.

    The operands are loaded once outside the loop, so the loop is pure cube
    issue: this is the scheduling-sensitive case where the two compilers can
    differ in how they keep the accumulator in L0C.
    """
    with pl.at(level=pl.Level.CORE_GROUP):
        ta = pl.load(a, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
        tb = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
        la = pl.move(ta, target_memory=pl.MemorySpace.Left)
        lb = pl.move(tb, target_memory=pl.MemorySpace.Right)
        acc = pl.matmul(la, lb, out_dtype=pl.FP32)
        for _ in pl.range(_ACC_TRIPS):
            acc = pl.matmul_acc(acc, la, lb)
        pl.store(acc, [0, 0], c)
    return c


@pl.jit
def matmul_kloop(
    a: pl.Tensor[[128, MM_K], pl.FP16],
    b: pl.Tensor[[MM_K, 128], pl.FP16],
    c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    """Tiled K reduction with a windowed load at a loop-variable offset.

    The first product is issued outside the loop so the body is a plain
    ``matmul_acc`` with no first-iteration branch — the form the Buffer IR
    selection gate accepts.
    """
    with pl.at(level=pl.Level.CORE_GROUP):
        a0 = pl.load(a, [0, 0], [128, MM_KC], target_memory=pl.MemorySpace.Mat)
        b0 = pl.load(b, [0, 0], [MM_KC, 128], target_memory=pl.MemorySpace.Mat)
        la0 = pl.move(a0, target_memory=pl.MemorySpace.Left)
        lb0 = pl.move(b0, target_memory=pl.MemorySpace.Right)
        acc = pl.matmul(la0, lb0, out_dtype=pl.FP32)
        for k in pl.range(1, MM_K // MM_KC):
            k0 = k * MM_KC
            at = pl.load(a, [0, k0], [128, MM_KC], target_memory=pl.MemorySpace.Mat)
            bt = pl.load(b, [k0, 0], [MM_KC, 128], target_memory=pl.MemorySpace.Mat)
            la = pl.move(at, target_memory=pl.MemorySpace.Left)
            lb = pl.move(bt, target_memory=pl.MemorySpace.Right)
            acc = pl.matmul_acc(acc, la, lb)
        pl.store(acc, [0, 0], c)
    return c


@pl.jit
def matmul_kloop_pipelined(
    a: pl.Tensor[[128, MM_K], pl.FP16],
    b: pl.Tensor[[MM_K, 128], pl.FP16],
    c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    """``matmul_kloop`` with a two-stage pipeline on the K loop.

    Under the PyPTO memory planner ``LowerPipelineLoops`` replicates the body,
    so the stage buffers are distinct tiles and no ``MemRef`` slot indexing
    reaches the bridge. Both routes are therefore compared on the same
    replicated schedule.
    """
    with pl.at(level=pl.Level.CORE_GROUP):
        a0 = pl.load(a, [0, 0], [128, MM_KC], target_memory=pl.MemorySpace.Mat)
        b0 = pl.load(b, [0, 0], [MM_KC, 128], target_memory=pl.MemorySpace.Mat)
        la0 = pl.move(a0, target_memory=pl.MemorySpace.Left)
        lb0 = pl.move(b0, target_memory=pl.MemorySpace.Right)
        acc = pl.matmul(la0, lb0, out_dtype=pl.FP32)
        for k in pl.pipeline(1, MM_K // MM_KC, stage=2):
            k0 = k * MM_KC
            at = pl.load(a, [0, k0], [128, MM_KC], target_memory=pl.MemorySpace.Mat)
            bt = pl.load(b, [k0, 0], [MM_KC, 128], target_memory=pl.MemorySpace.Mat)
            la = pl.move(at, target_memory=pl.MemorySpace.Left)
            lb = pl.move(bt, target_memory=pl.MemorySpace.Right)
            acc = pl.matmul_acc(acc, la, lb)
        pl.store(acc, [0, 0], c)
    return c


@pl.jit
def vector_add(
    lhs: pl.Tensor[[128, 128], pl.FP32],
    rhs: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    """Reference vector kernel: two loads, one add, one store."""
    with pl.at(level=pl.Level.CORE_GROUP):
        lhs_tile = pl.load(lhs, [0, 0], [128, 128])
        rhs_tile = pl.load(rhs, [0, 0], [128, 128])
        pl.store(pl.add(lhs_tile, rhs_tile), [0, 0], out)
    return out


@pl.jit
def vector_scale_add(
    lhs: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    """Scalar elementwise chain: ``lhs * 1.5 + 1.0`` (``buffer.muls``/``adds``)."""
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(lhs, [0, 0], [128, 128])
        pl.store(pl.add(pl.mul(tile, 1.5), 1.0), [0, 0], out)
    return out


# ── Program-level fusion: one cube kernel followed by one vector kernel ──────


@pl.jit.incore
def _mm_kernel(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    tile_a_l1 = pl.load(a, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
    tile_b_l1 = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
    tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
    tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
    pl.store(pl.matmul(tile_a_l0a, tile_b_l0b, out_dtype=pl.FP32), [0, 0], out)
    return out


@pl.jit.incore
def _scale_kernel(x: pl.Tensor, out: pl.Out[pl.Tensor]):
    tile = pl.load(x, [0, 0], [128, 128])
    pl.store(pl.mul(tile, 0.5), [0, 0], out)
    return out


@pl.jit
def mm_then_scale(
    a: pl.Tensor[[128, 128], pl.FP16],
    b: pl.Tensor[[128, 128], pl.FP16],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    """Two InCore kernels under one orchestration: cube MMAD, then vector scale.

    This exercises both device pipelines in one program without the mixed-kernel
    cross-core pipe protocol, which the bridge does not model.
    """
    tmp = pl.create_tensor([128, 128], dtype=pl.FP32)
    tmp = _mm_kernel(a, b, tmp)
    out = _scale_kernel(tmp, out)
    return out
