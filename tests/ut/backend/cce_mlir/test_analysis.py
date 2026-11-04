# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Buffer IR -> typed kernel IR tests (no cce-mlir dependency)."""

import pytest
from pypto.backend.cce_mlir import (
    CceKernelKind,
    CceMlirError,
    CceTarget,
    lower_program,
)
from pypto.backend.cce_mlir.kernel_ir import CceFor, CceLayout, CceMemorySpace
from pypto.pypto_core.ir import ParamDirection

from . import kernels


def _lower(kernel) -> tuple:
    return lower_program(kernel.lower(), target=CceTarget.A3)


def test_cube_matmul_lowers_from_real_buffer_ir():
    (lowered,) = _lower(kernels.cube_matmul)

    assert lowered.function_name.endswith("_incore_0")
    assert lowered.kind is CceKernelKind.CUBE
    assert lowered.target is CceTarget.A3
    assert lowered.device_symbol.endswith("__kernel0")
    assert [parameter.shape for parameter in lowered.parameters] == [
        (64, 128),
        (128, 64),
        (64, 64),
    ]
    assert [parameter.direction for parameter in lowered.parameters] == [
        ParamDirection.In,
        ParamDirection.In,
        ParamDirection.Out,
    ]
    assert all(parameter.space is CceMemorySpace.GM for parameter in lowered.parameters)
    # Directions come from the function, not from the parameter spelling.
    assert lowered.parameters[2].element.mlir == "f32"

    assert [op.op_name for op in lowered.ops] == [
        "buffer.alloc",
        "buffer.alloc",
        "buffer.alloc",
        "buffer.alloc",
        "buffer.alloc",
        "buffer.load",
        "buffer.load",
        "buffer.copy",
        "buffer.copy",
        "buffer.matmul",
        "buffer.store",
    ]


def test_allocation_spaces_and_encodings_are_derived_from_buffer_type():
    (lowered,) = _lower(kernels.cube_matmul)
    buffers = [value for value in lowered.values if not value.is_parameter]

    assert [buffer.space for buffer in buffers] == [
        CceMemorySpace.L1,
        CceMemorySpace.L1,
        CceMemorySpace.L0A,
        CceMemorySpace.L0B,
        CceMemorySpace.L0C,
    ]
    assert buffers[0].encoding.layout == CceLayout((0, 1), (16, 16), (1, 0))
    assert buffers[2].encoding.layout == CceLayout((0, 1), (16, 16), ())
    assert buffers[3].encoding.layout == CceLayout((1, 0), (16, 16), ())
    assert buffers[4].encoding.layout == CceLayout((0, 1), (16, 16), (1, 0))
    assert buffers[4].element == lowered.parameters[2].element


def test_vector_add_lowers_to_the_ub_path():
    (lowered,) = _lower(kernels.vector_add)

    assert lowered.kind is CceKernelKind.VECTOR
    buffers = [value for value in lowered.values if not value.is_parameter]
    assert [buffer.space for buffer in buffers] == [CceMemorySpace.UB, CceMemorySpace.UB]
    assert all(buffer.encoding.layout is None for buffer in buffers)
    assert [op.op_name for op in lowered.ops] == [
        "buffer.alloc",
        "buffer.alloc",
        "buffer.load",
        "buffer.load",
        "buffer.add",
        "buffer.store",
    ]
    assert lowered.parameters[2].direction is ParamDirection.Out


def test_static_loop_becomes_a_structured_statement():
    (lowered,) = _lower(kernels.loop_matmul)

    loops = [statement for statement in lowered.body if isinstance(statement, CceFor)]
    assert len(loops) == 1
    loop = loops[0]
    assert (loop.start, loop.stop, loop.step) == (0, 2, 1)
    assert [op.op_name for op in loop.body] == ["buffer.matmul_acc"]
    # ``ops`` flattens loop bodies in program order for reporting.
    assert [op.op_name for op in lowered.ops].count("buffer.matmul_acc") == 1


def test_shapes_and_dtypes_reach_the_typed_ir():
    (lowered,) = _lower(kernels.vector_add)
    lhs = lowered.parameters[0]

    assert lhs.shape == (128, 128)
    assert lhs.element.mlir == "f32"
    assert lhs.byte_size == 128 * 128 * 4
    assert lhs.element == lowered.parameters[1].element
    assert lowered.parameters[2].element.mlir == "f32"


def test_scalar_elementwise_variant_is_refused_with_a_reason():
    with pytest.raises(CceMlirError) as error:
        _lower(kernels.vector_add_scalar)

    assert error.value.reason == "unsupported-op"
    assert error.value.op_name == "buffer.adds"


def test_partial_window_transfer_is_refused():
    with pytest.raises(CceMlirError) as error:
        _lower(kernels.vector_add_window)

    assert error.value.reason in {"shape-mismatch", "unsupported-window"}


def test_unknown_target_arch_is_refused(monkeypatch):
    from pypto.backend.cce_mlir import analysis

    class _Handler:
        def get_pto_target_arch(self):
            return "a9"

    monkeypatch.setattr(analysis._backend_core, "get_handler", lambda: _Handler())

    with pytest.raises(CceMlirError) as error:
        analysis.resolve_target()
    assert error.value.reason == "unsupported-target"


def test_dtype_sizes_are_used_for_byte_sizes():
    (lowered,) = _lower(kernels.cube_matmul)
    for parameter in lowered.parameters:
        expected = 1
        for extent in parameter.shape:
            expected *= extent
        expected *= parameter.element.byte_size
        assert parameter.byte_size == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
