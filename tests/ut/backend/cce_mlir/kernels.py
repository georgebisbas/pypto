# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Small kernels used by the cce-mlir front-end tests.

Parameter names deliberately differ from the cube fixture (`c` is `result`,
the vector output is `out`) so a lowering that matches by name fails here.
"""

import pypto.language as pl


@pl.jit
def cube_matmul(
    a: pl.Tensor[[64, 128], pl.FP16],
    b: pl.Tensor[[128, 64], pl.FP16],
    result: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        a_mat = pl.load(a, [0, 0], [64, 128], target_memory=pl.MemorySpace.Mat)
        b_mat = pl.load(b, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
        a_left = pl.move(a_mat, target_memory=pl.MemorySpace.Left)
        b_right = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
        acc = pl.matmul(a_left, b_right, out_dtype=pl.FP32)
        pl.store(acc, [0, 0], result)
    return result


@pl.jit
def cube_matmul_32x64x32(
    a: pl.Tensor[[32, 64], pl.FP16],
    b: pl.Tensor[[64, 32], pl.FP16],
    result: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        a_mat = pl.load(a, [0, 0], [32, 64], target_memory=pl.MemorySpace.Mat)
        b_mat = pl.load(b, [0, 0], [64, 32], target_memory=pl.MemorySpace.Mat)
        a_left = pl.move(a_mat, target_memory=pl.MemorySpace.Left)
        b_right = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
        acc = pl.matmul(a_left, b_right, out_dtype=pl.FP32)
        pl.store(acc, [0, 0], result)
    return result


@pl.jit
def loop_matmul(
    a: pl.Tensor[[128, 128], pl.FP16],
    b: pl.Tensor[[128, 128], pl.FP16],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        a_mat = pl.load(a, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
        b_mat = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
        a_left = pl.move(a_mat, target_memory=pl.MemorySpace.Left)
        b_right = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
        acc = pl.matmul(a_left, b_right, out_dtype=pl.FP32)
        for _ in pl.range(2):
            acc = pl.matmul_acc(acc, a_left, b_right)
        pl.store(acc, [0, 0], out)
    return out


@pl.jit
def vector_add(
    lhs: pl.Tensor[[128, 128], pl.FP32],
    rhs: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        lhs_tile = pl.load(lhs, [0, 0], [128, 128])
        rhs_tile = pl.load(rhs, [0, 0], [128, 128])
        out_tile = pl.add(lhs_tile, rhs_tile)
        pl.store(out_tile, [0, 0], out)
    return out


@pl.jit
def vector_add_scalar(
    lhs: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        lhs_tile = pl.load(lhs, [0, 0], [128, 128])
        out_tile = pl.add(lhs_tile, 1.0)
        pl.store(out_tile, [0, 0], out)
    return out


@pl.jit
def vector_add_window(
    lhs: pl.Tensor[[256, 256], pl.FP32],
    rhs: pl.Tensor[[256, 256], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        lhs_tile = pl.load(lhs, [0, 0], [128, 128])
        rhs_tile = pl.load(rhs, [0, 0], [128, 128])
        out_tile = pl.add(lhs_tile, rhs_tile)
        pl.store(out_tile, [0, 0], out)
    return out
