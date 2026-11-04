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


def test_static_loop_builds_an_scf_for():
    _, text = _emit(kernels.loop_matmul)

    assert "scf.for" in text
    assert "pto.matmul_acc" in text
    assert "scf.yield" in text


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


def test_in_place_update_copies_back_inside_a_loop():
    # PyPTO's Buffer IR reuses the destination tile as a source for in-place
    # ops, and cce-mlir miscomputes dest == src. The emitter computes into a
    # distinct scratch; inside a loop it then copies the result back into the
    # destination buffer, so the value yielded for the loop carry is still that
    # buffer and not a separate allocation.
    _, text = _emit(kernels.vector_chunked_add)

    loop = text.index("scf.for")
    assert "tensor.empty" not in text[loop:]
    add = next(line for line in text.splitlines() if "pto.add " in line)
    first, _, destination = (
        part.strip() for part in add.split("pto.add", 1)[1].split(" : ", 1)[0].split(",")[:3]
    )
    assert destination != first, add
    assert "pto.mov" in text[loop:]
    moved = next(line for line in text.splitlines() if "pto.mov" in line)
    assert moved.split("pto.mov", 1)[1].split(" : ", 1)[0].split(",")[1].strip() == first


def test_scalar_elementwise_builds_with_a_typed_scalar():
    _, text = _emit(kernels.vector_scalar_chain)

    assert "pto.muls" in text
    assert "pto.adds" in text
    assert "arith.constant 1.500000e+00 : f32" in text


def test_full_window_transfers_emit_no_slice():
    # A tile that fills its GM tensor needs no window: the emitted MLIR for the
    # existing simple kernels must stay slice-free.
    _, text = _emit(kernels.cube_matmul)

    assert "tensor.extract_slice" not in text
    assert "tensor.insert_slice" not in text


def test_offset_window_emits_an_extract_slice():
    _, text = _emit(kernels.vector_add_offset_window)

    assert "tensor.extract_slice" in text
    assert "tensor.insert_slice" not in text


def test_windowed_store_emits_an_insert_slice():
    _, text = _emit(kernels.vector_store_window)

    assert "tensor.extract_slice" in text
    assert "tensor.insert_slice" in text


def test_loop_variable_offset_slices_with_the_induction_variable():
    _, text = _emit(kernels.matmul_kloop_windowed)

    assert "scf.for" in text
    assert "tensor.extract_slice" in text
    # The slice offset is computed from the loop induction variable, not a
    # constant: arithmetic feeds the slice, and the pretty form prints the
    # dynamic operand (a `%`-prefixed value) inside the offset list.
    assert "arith.muli" in text
    assert any(
        "tensor.extract_slice" in line and "%" in line.split("tensor.extract_slice", 1)[1]
        for line in text.splitlines()
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
