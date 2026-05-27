# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 hierarchical distributed split-K GEMM + peer allreduce.

Parallel model (strong scaling on K):
  - Shard ``K`` across ranks: rank ``r`` holds ``A[:, k_r:k_{r+1}]`` and ``B[k_r:k_{r+1}, :]``.
  - **Intra-chip:** ``SPLIT`` CORE_GROUP workers sum partials over ``k_s = K / nranks`` (see
    ``examples/kernels/10_split_k.py``).
  - **Inter-chip:** 4-phase ``reduce_step`` sums rank-local ``M0 x N`` partials.

Golden: ``outputs[r] == sum_s matmul(A[:, s*k_s:(s+1)*k_s], B[s*k_s:(s+1)*k_s, :])`` for all ``r``.

Two ranks only (single peer read in ``reduce_step``).
"""

from __future__ import annotations

import pypto.language as pl
import pypto.language.distributed as pld

from tests.st.distributed.l3_common import (
    M0,
    K,
    N,
    data_window_nbytes,
    make_reduce_step_2rank,
    signal_window_nbytes,
)

# ST defaults: large K for meaningful split-K; SPLIT matches 10_split_k style.
DEFAULT_K = 512
DEFAULT_SPLIT = 4

__all__ = ["M0", "K", "N", "DEFAULT_K", "DEFAULT_SPLIT", "build_l3_hier_split_k_allreduce_gemm_program"]


def make_hier_split_k_gemm(*, m0: int, k_s: int, n: int, split: int):
    """InCore GEMM with CORE_GROUP split-K over local contraction ``k_s``."""

    if k_s % split != 0:
        raise ValueError(f"k_s={k_s} must be divisible by split={split}")
    ks = k_s // split

    @pl.function(type=pl.FunctionType.InCore)
    def gemm_hier_split_k(
        self,
        a_shard: pl.Tensor[[m0, k_s], pl.FP32],
        b_shard: pl.Tensor[[k_s, n], pl.FP32],
        partial: pl.Out[pl.Tensor[[m0, n], pl.FP32]],
    ) -> pl.Tensor[[m0, n], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="zero_init"):
            partial = pl.assemble(partial, pl.full([m0, n], dtype=pl.FP32, value=0.0), [0, 0])
        for ks_idx in pl.parallel(0, split):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="split_k"):
                k0 = ks_idx * ks
                a_k = a_shard[:, k0 : k0 + ks]
                b_k = b_shard[k0 : k0 + ks, :]
                part = pl.matmul(a_k, b_k, out_dtype=pl.FP32)
                partial = pl.assemble(partial, part, [0, 0], atomic=pl.AtomicType.Add)
        return partial

    return gemm_hier_split_k


def build_l3_hier_split_k_allreduce_gemm_program(
    *,
    nranks: int = 2,
    m0: int = M0,
    k: int = DEFAULT_K,
    n: int = N,
    split: int = DEFAULT_SPLIT,
):
    """K-sharded hierarchical split-K GEMM then 2-rank allreduce."""

    if nranks != 2:
        raise ValueError(
            f"build_l3_hier_split_k_allreduce_gemm_program requires nranks=2; got {nranks}"
        )
    if k % nranks != 0:
        raise ValueError(f"k={k} must be divisible by nranks={nranks}")

    k_s = k // nranks
    if k_s % split != 0:
        raise ValueError(f"k_s={k_s} must be divisible by split={split}")

    a_shape = [nranks, m0, k_s]
    b_shape = [nranks, k_s, n]
    partial_shape = [nranks, m0, n]
    out_shape = [nranks, m0, n]

    @pl.program
    class L3HierSplitKAllReduceGemmProgram:
        gemm_hier_split_k = make_hier_split_k_gemm(m0=m0, k_s=k_s, n=n, split=split)
        reduce_step = make_reduce_step_2rank(m0=m0, n=n)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            a_shard: pl.Tensor[[m0, k_s], pl.FP32],
            b_shard: pl.Tensor[[k_s, n], pl.FP32],
            partial: pl.InOut[pl.Tensor[[m0, n], pl.FP32]],
            out: pl.Out[pl.Tensor[[m0, n], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[m0, n], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[m0, n], pl.FP32]:
            partial_out: pl.Tensor[[m0, n], pl.FP32] = self.gemm_hier_split_k(a_shard, b_shard, partial)
            return self.reduce_step(partial_out, out, data, signal, peer)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            a: pl.Tensor[a_shape, pl.FP32],  # type: ignore[valid-type]
            b: pl.Tensor[b_shape, pl.FP32],  # type: ignore[valid-type]
            partials: pl.InOut[pl.Tensor[partial_shape, pl.FP32]],  # type: ignore[valid-type]
            outputs: pl.Out[pl.Tensor[out_shape, pl.FP32]],  # type: ignore[valid-type]
        ) -> pl.Tensor[out_shape, pl.FP32]:  # type: ignore[valid-type]
            data_buf = pld.alloc_window_buffer(data_window_nbytes(m0, n))
            signal_buf = pld.alloc_window_buffer(signal_window_nbytes())

            data0 = pld.window(data_buf, [m0, n], dtype=pl.FP32)
            signal0 = pld.window(signal_buf, [1, 1], dtype=pl.INT32)
            self.chip_orch(a[0], b[0], partials[0], outputs[0], data0, signal0, 1, device=0)

            data1 = pld.window(data_buf, [m0, n], dtype=pl.FP32)
            signal1 = pld.window(signal_buf, [1, 1], dtype=pl.INT32)
            self.chip_orch(a[1], b[1], partials[1], outputs[1], data1, signal1, 0, device=1)

            return outputs

    return L3HierSplitKAllReduceGemmProgram
