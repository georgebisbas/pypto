# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Layout and memory-space derivation tests (no cce-mlir dependency)."""

import pytest
from pypto.backend.cce_mlir import CceMlirError
from pypto.backend.cce_mlir.kernel_ir import CceElementType, CceLayout, CceMemorySpace
from pypto.backend.cce_mlir.layout import (
    check_buffer_descriptor,
    element_extent,
    encoding_for,
    memory_space,
)
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import CompactMode, MemorySpace, PadValue, TileLayout


def test_memory_space_mapping_covers_the_tiers_the_route_supports():
    assert memory_space(MemorySpace.DDR) is CceMemorySpace.GM
    assert memory_space(MemorySpace.Vec) is CceMemorySpace.UB
    assert memory_space(MemorySpace.Mat) is CceMemorySpace.L1
    assert memory_space(MemorySpace.Left) is CceMemorySpace.L0A
    assert memory_space(MemorySpace.Right) is CceMemorySpace.L0B
    assert memory_space(MemorySpace.Acc) is CceMemorySpace.L0C


def test_memory_space_refuses_the_bias_and_scale_files():
    for space in (MemorySpace.Bias, MemorySpace.LeftScale, MemorySpace.RightScale):
        with pytest.raises(CceMlirError) as error:
            memory_space(space)
        assert error.value.reason == "unsupported-memory-space"


def test_element_extent_is_256_bits_of_elements():
    assert element_extent(CceElementType.from_data_type(DataType.FP16)) == 16
    assert element_extent(CceElementType.from_data_type(DataType.FP32)) == 8
    assert element_extent(CceElementType.from_data_type(DataType.INT8)) == 32


def test_encodings_follow_cce_mlir_factories():
    fp16 = CceElementType.from_data_type(DataType.FP16)
    fp32 = CceElementType.from_data_type(DataType.FP32)

    assert encoding_for(CceMemorySpace.GM, fp16).layout is None
    assert encoding_for(CceMemorySpace.UB, fp16).layout is None

    assert encoding_for(CceMemorySpace.L1, fp16).layout == CceLayout((0, 1), (16, 16), (1, 0))
    assert encoding_for(CceMemorySpace.L0A, fp16).layout == CceLayout((0, 1), (16, 16), ())
    assert encoding_for(CceMemorySpace.L0B, fp16).layout == CceLayout((1, 0), (16, 16), ())
    assert encoding_for(CceMemorySpace.L0C, fp16).layout == CceLayout((0, 1), (16, 16), (1, 0))

    # f32 shrinks the contraction/staging tile to 8 elements.
    assert encoding_for(CceMemorySpace.L1, fp32).layout == CceLayout((0, 1), (16, 8), (1, 0))
    assert encoding_for(CceMemorySpace.L0A, fp32).layout == CceLayout((0, 1), (16, 8), ())
    # The accumulator block stays 16x16 for every dtype.
    assert encoding_for(CceMemorySpace.L0C, fp32).layout == CceLayout((0, 1), (16, 16), (1, 0))


def test_buffer_descriptor_accepts_the_canonical_combinations():
    fp16 = CceElementType.from_data_type(DataType.FP16)
    fp32 = CceElementType.from_data_type(DataType.FP32)

    check_buffer_descriptor(
        name="mat",
        space=CceMemorySpace.L1,
        element=fp16,
        blayout=TileLayout.col_major,
        slayout=TileLayout.row_major,
        fractal=512,
        pad=PadValue.null,
        compact=CompactMode.null,
    )
    check_buffer_descriptor(
        name="acc",
        space=CceMemorySpace.L0C,
        element=fp32,
        blayout=TileLayout.col_major,
        slayout=TileLayout.row_major,
        fractal=1024,
        pad=PadValue.null,
        compact=CompactMode.null,
    )
    check_buffer_descriptor(
        name="vec",
        space=CceMemorySpace.UB,
        element=fp32,
        blayout=TileLayout.row_major,
        slayout=TileLayout.none_box,
        fractal=512,
        pad=PadValue.null,
        compact=CompactMode.null,
    )


def test_buffer_descriptor_refuses_contradictions():
    fp16 = CceElementType.from_data_type(DataType.FP16)

    with pytest.raises(CceMlirError) as wrong_fractal:
        check_buffer_descriptor(
            name="mat",
            space=CceMemorySpace.L1,
            element=fp16,
            blayout=TileLayout.col_major,
            slayout=TileLayout.row_major,
            fractal=1024,
            pad=PadValue.null,
            compact=CompactMode.null,
        )
    assert wrong_fractal.value.reason == "layout-mismatch"

    with pytest.raises(CceMlirError) as wrong_layout:
        check_buffer_descriptor(
            name="right",
            space=CceMemorySpace.L0B,
            element=fp16,
            blayout=TileLayout.col_major,
            slayout=TileLayout.row_major,
            fractal=512,
            pad=PadValue.null,
            compact=CompactMode.null,
        )
    assert wrong_layout.value.reason == "layout-mismatch"

    with pytest.raises(CceMlirError) as padded:
        check_buffer_descriptor(
            name="mat",
            space=CceMemorySpace.L1,
            element=fp16,
            blayout=TileLayout.col_major,
            slayout=TileLayout.row_major,
            fractal=512,
            pad=PadValue.zero,
            compact=CompactMode.null,
        )
    assert padded.value.reason == "unsupported-padding"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
