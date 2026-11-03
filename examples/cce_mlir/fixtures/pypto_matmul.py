"""The first reproducible PyPTO-to-cce-mlir cube kernel."""

import pypto.language as pl


@pl.jit
def matmul_64x128x64(
    a: pl.Tensor[[64, 128], pl.FP16],
    b: pl.Tensor[[128, 64], pl.FP16],
    c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        a_mat = pl.load(a, [0, 0], [64, 128], target_memory=pl.MemorySpace.Mat)
        b_mat = pl.load(b, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
        a_left = pl.move(a_mat, target_memory=pl.MemorySpace.Left)
        b_right = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
        c_acc = pl.matmul(a_left, b_right, out_dtype=pl.FP32)
        pl.store(c_acc, [0, 0], c)
    return c
