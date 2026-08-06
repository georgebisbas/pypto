# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Capacity-gate warning (MemoryReuse) and its AllocateMemoryAddr backstop.

MemoryReuse emits a Warning when even the legacy (no-reuse) packing of a memory
space overflows the platform capacity, and AllocateMemoryAddr then hard-fails
with the identical footprint numbers — both use the shared SpaceFootprint walk
(#1475). These tests pin:

* the warning text: exact footprint/capacity bytes plus the explicit
  "will fail at AllocateMemoryAddr" outcome, so an author does not have to read
  the allocator's CHECK to learn it;
* the boundary: a single tile exactly at capacity fits (no warning, no failure);
* the parity: a pass-33 warning fires exactly when pass-34 fails, with the same
  numbers, across a tile-size sweep.
"""

from __future__ import annotations

import pypto.language as pl
import pytest
from pypto import passes
from pypto.ir import Program


def _gemm_with_right_extract(n_tile: int) -> Program:
    """InCore GEMM whose B operand is extracted into Right at full K_TILE.

    ``b_right`` is ``[256, n_tile]`` FP16 = ``256 * n_tile * 2`` bytes in the
    Right (L0B) space (Ascend910B capacity 65536):

    * ``n_tile=256`` -> 131072 bytes (overflows),
    * ``n_tile=128`` -> exactly 65536 bytes (fits at the limit),
    * ``n_tile=64``  -> 32768 bytes (fits).
    """
    m_tile, k_tile = 128, 256

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[m_tile, k_tile], pl.FP16],
            b: pl.Tensor[[k_tile, n_tile], pl.FP16],
            c: pl.Out[pl.Tensor[[m_tile, n_tile], pl.FP32]],
        ) -> pl.Tensor[[m_tile, n_tile], pl.FP32]:
            a_mat: pl.Tile[[m_tile, k_tile], pl.FP16, pl.Mem.Mat] = pl.load(
                a, [0, 0], [m_tile, k_tile], target_memory=pl.MemorySpace.Mat
            )
            b_mat: pl.Tile[[k_tile, n_tile], pl.FP16, pl.Mem.Mat] = pl.load(
                b, [0, 0], [k_tile, n_tile], target_memory=pl.MemorySpace.Mat
            )
            a_left: pl.Tile[[m_tile, k_tile], pl.FP16, pl.Mem.Left] = pl.tile.extract(
                a_mat, 0, 0, [m_tile, k_tile], target_memory=pl.Mem.Left
            )
            b_right: pl.Tile[[k_tile, n_tile], pl.FP16, pl.Mem.Right] = pl.tile.extract(
                b_mat, 0, 0, [k_tile, n_tile], target_memory=pl.Mem.Right
            )
            acc: pl.Tile[[m_tile, n_tile], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(a_left, b_right)
            result: pl.Tensor[[m_tile, n_tile], pl.FP32] = pl.store(acc, [0, 0], c)
            return result

    return Before


def _run_memory_pipeline(program: Program) -> None:
    """Run the memory-planning tail: strides, init_mem_ref, aliases, reuse, addr."""
    prepared = passes.materialize_tensor_strides()(program)
    initialized = passes.init_mem_ref()(prepared)
    aliased = passes.materialize_semantic_aliases()(initialized)
    reused = passes.memory_reuse()(aliased)
    passes.allocate_memory_addr()(reused)


def test_warning_reports_exact_bytes_and_outcome(ascend_backend, capfd):
    """Deterministic overflow: MemoryReuse names the exact bytes and the outcome."""
    with pytest.raises(ValueError, match=r"Right buffer usage .* exceeds platform limit"):
        _run_memory_pipeline(_gemm_with_right_extract(256))
    captured = capfd.readouterr()
    combined = captured.out + captured.err
    assert "capacity-gated reuse could not fit memory space Right" in combined
    assert "footprint = 131072 bytes > capacity = 65536 bytes" in combined
    assert "will fail at AllocateMemoryAddr" in combined


def test_allocate_memory_addr_backstop_still_fails(ascend_backend):
    """The hard CHECK at AllocateMemoryAddr still fires after the pass-33 warning."""
    with pytest.raises(
        ValueError,
        match=r"Right buffer usage \(131072 bytes\) exceeds platform limit \(65536 bytes\)",
    ):
        _run_memory_pipeline(_gemm_with_right_extract(256))


@pytest.mark.parametrize(
    "n_tile,footprint_bytes",
    [(64, 32768), (128, 65536), (256, 131072)],
)
def test_warning_parity_with_allocator(ascend_backend, capfd, n_tile, footprint_bytes):
    """Pass-33 warning <-> pass-34 failure, with matching byte numbers."""
    overflow = footprint_bytes > 65536
    program = _gemm_with_right_extract(n_tile)
    if overflow:
        with pytest.raises(ValueError, match=r"Right buffer usage .* exceeds platform limit"):
            _run_memory_pipeline(program)
    else:
        _run_memory_pipeline(program)

    captured = capfd.readouterr()
    combined = captured.out + captured.err
    warning_present = "capacity-gated reuse could not fit memory space Right" in combined
    assert warning_present == overflow, (
        f"MemoryReuse warning should fire iff the allocator fails (n_tile={n_tile}):\n{combined}"
    )
    if overflow:
        assert f"footprint = {footprint_bytes} bytes > capacity = 65536 bytes" in combined
