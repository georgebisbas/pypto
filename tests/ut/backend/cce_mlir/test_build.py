# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""MLIR construction tests. Skipped when the cce-mlir bindings are absent."""

import pytest

pytest.importorskip("cce_mlir")

from pypto.backend.cce_mlir import CceTarget, lower_program  # noqa: E402
from pypto.backend.cce_mlir.emit import emit_module  # noqa: E402

from . import kernels  # noqa: E402


def _emit(kernel):
    (lowered,) = lower_program(kernel.lower(), target=CceTarget.A3)
    return lowered, str(emit_module(lowered))


def test_cube_matmul_builds_a_cube_kernel_with_real_layouts():
    _, text = _emit(kernels.cube_matmul)

    assert "npu.module @" in text
    assert "#npu.kernel_target<a3>" in text
    assert "kind = #npu.kernel_kind<cube>" in text
    assert "layout.pack_layout" in text
    assert "layout.repack_layout" in text
    assert "layout.unpack_layout" in text
    assert "pto.load" in text
    assert "pto.mov" in text
    assert "pto.matmul" in text
    assert "pto.store" in text
    assert "tensor<64x128xf16" in text
    assert "tensor<64x64xf32" in text
    assert "#npu.memory_space<l1>" in text
    assert "#npu.memory_space<l0a>" in text
    assert "#npu.memory_space<l0b>" in text
    assert "#npu.memory_space<l0c>" in text


def test_cube_matmul_encodings_come_from_the_buffer_contract():
    _, text = _emit(kernels.cube_matmul)

    # NZ staging layout for the L1 buffers.
    assert "inner_dims_pos = [0, 1], inner_tile_sizes = [16, 16], outer_dims_perm = [1, 0]" in text
    # ZN layout for the right operand.
    assert "inner_dims_pos = [1, 0], inner_tile_sizes = [16, 16], outer_dims_perm = []" in text


def test_changing_shapes_changes_the_emitted_types():
    _, big = _emit(kernels.cube_matmul)
    _, small = _emit(kernels.cube_matmul_32x64x32)

    assert "tensor<64x128xf16" in big and "tensor<32x64xf16" in small
    assert "tensor<64x64xf32" in big and "tensor<32x32xf32" in small
    assert big != small


def test_vector_add_builds_a_vector_kernel_in_ub():
    _, text = _emit(kernels.vector_add)

    assert "kind = #npu.kernel_kind<vector>" in text
    assert "pto.load" in text
    assert "pto.add" in text
    assert "pto.store" in text
    assert "#npu.memory_space<ub>" in text
    assert "layout.pack_layout" not in text


def test_emitted_module_verifies_and_reparses():
    from cce_mlir import ir as cce_ir
    from pypto.backend.cce_mlir._dialects import make_context

    _, text = _emit(kernels.cube_matmul)
    context = make_context()
    with context, cce_ir.Location.unknown():
        reparsed = cce_ir.Module.parse(text)
        reparsed.operation.verify()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
