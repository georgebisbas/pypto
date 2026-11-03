# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Derivation of cce-mlir memory spaces and fractal encodings.

The rules mirror cce-mlir's own layout factories
(``include/NPU/IR/NPULayout.h``): the 32-byte NZ minor block, the
256-bit contraction tile, and the fixed 16-row fractal. PyPTO's
``BufferType`` is the authority for what a buffer *is*; this module derives what
it must *look like* in cce-mlir and refuses when the two disagree.
"""

from __future__ import annotations

from pypto.pypto_core.ir import CompactMode, MemorySpace, PadValue, TileLayout

from .errors import CceMlirError
from .kernel_ir import (
    CceElementType,
    CceEncoding,
    CceLayout,
    CceMemorySpace,
)

__all__ = [
    "CONTRACTION_BITS",
    "C0_BYTES",
    "FRACTAL_ROWS",
    "check_buffer_descriptor",
    "element_extent",
    "encoding_for",
    "memory_space",
]

# cce-mlir NPULayout.h: the M0/N0 fractal block is always 16 rows; the staging
# minor block is 32 bytes of elements; the contraction tile is 256 bits.
FRACTAL_ROWS = 16
C0_BYTES = 32
CONTRACTION_BITS = 256

_MEMORY_SPACES: dict[MemorySpace, CceMemorySpace] = {
    MemorySpace.DDR: CceMemorySpace.GM,
    MemorySpace.Vec: CceMemorySpace.UB,
    MemorySpace.Mat: CceMemorySpace.L1,
    MemorySpace.Left: CceMemorySpace.L0A,
    MemorySpace.Right: CceMemorySpace.L0B,
    MemorySpace.Acc: CceMemorySpace.L0C,
}


def memory_space(space: MemorySpace) -> CceMemorySpace:
    """Map a PyPTO memory space onto cce-mlir's named tier.

    Args:
        space: The PyPTO ``MemorySpace`` of a buffer or tensor parameter.

    Returns:
        The cce-mlir memory space.

    Raises:
        CceMlirError: For spaces the route does not model (bias and the A5
            scale files).
    """
    mapped = _MEMORY_SPACES.get(space)
    if mapped is None:
        supported = ", ".join(sorted(item.name for item in _MEMORY_SPACES))
        raise CceMlirError(
            "unsupported-memory-space",
            f"memory space {space} is not supported by the cce-mlir route (supported: {supported})",
        )
    return mapped


def element_extent(element: CceElementType) -> int:
    """Elements in one 256-bit contraction tile (cce-mlir ``k0Of``/``c0Of``).

    Args:
        element: The tensor element type.

    Returns:
        ``256 // bitwidth``, e.g. 16 for f16 and 8 for f32.
    """
    return CONTRACTION_BITS // element.bits


def _nz_layout(element: CceElementType) -> CceLayout:
    """GM/L1-facing NZ staging layout."""
    return CceLayout(
        inner_dims_pos=(0, 1),
        inner_tile_sizes=(FRACTAL_ROWS, element_extent(element)),
        outer_dims_perm=(1, 0),
    )


def _nz_accum_layout() -> CceLayout:
    """L0C accumulator layout: fixed 16x16 regardless of dtype."""
    return CceLayout(
        inner_dims_pos=(0, 1),
        inner_tile_sizes=(FRACTAL_ROWS, FRACTAL_ROWS),
        outer_dims_perm=(1, 0),
    )


def _zz_layout(element: CceElementType) -> CceLayout:
    """L0A left-matrix layout."""
    return CceLayout(
        inner_dims_pos=(0, 1),
        inner_tile_sizes=(FRACTAL_ROWS, element_extent(element)),
        outer_dims_perm=(),
    )


def _zn_layout(element: CceElementType) -> CceLayout:
    """L0B right-matrix layout."""
    return CceLayout(
        inner_dims_pos=(1, 0),
        inner_tile_sizes=(FRACTAL_ROWS, element_extent(element)),
        outer_dims_perm=(),
    )


def encoding_for(space: CceMemorySpace, element: CceElementType) -> CceEncoding:
    """Derive the cce-mlir encoding for a memory tier and element type.

    Args:
        space: The cce-mlir memory space.
        element: The tensor element type.

    Returns:
        ``CceEncoding(layout=None)`` for dense GM/UB tensors, otherwise the
        canonical fractal packing for the tier.
    """
    if space in (CceMemorySpace.GM, CceMemorySpace.UB):
        return CceEncoding()
    if space is CceMemorySpace.L1:
        return CceEncoding(_nz_layout(element))
    if space is CceMemorySpace.L0A:
        return CceEncoding(_zz_layout(element))
    if space is CceMemorySpace.L0B:
        return CceEncoding(_zn_layout(element))
    if space is CceMemorySpace.L0C:
        return CceEncoding(_nz_accum_layout())
    raise CceMlirError(
        "unsupported-memory-space",
        f"no cce-mlir encoding is defined for memory space {space}",
    )


def _expected_fractal_bytes(space: CceMemorySpace, element: CceElementType) -> int | None:
    """Fractal box size in bytes for a tier, or ``None`` for dense tiers."""
    if space in (CceMemorySpace.GM, CceMemorySpace.UB):
        return None
    rows, extent = (
        (FRACTAL_ROWS, FRACTAL_ROWS)
        if space is CceMemorySpace.L0C
        else (FRACTAL_ROWS, element_extent(element))
    )
    return rows * extent * element.byte_size


def _expected_block_layouts(space: CceMemorySpace) -> tuple[TileLayout, TileLayout] | None:
    """The ``(blayout, slayout)`` PyPTO must report for a tier, if constrained."""
    if space is CceMemorySpace.GM:
        return None
    if space is CceMemorySpace.UB:
        return (TileLayout.row_major, TileLayout.none_box)
    if space is CceMemorySpace.L0B:
        return (TileLayout.row_major, TileLayout.col_major)
    return (TileLayout.col_major, TileLayout.row_major)


def check_buffer_descriptor(
    *,
    name: str,
    space: CceMemorySpace,
    element: CceElementType,
    blayout: TileLayout,
    slayout: TileLayout,
    fractal: int,
    pad: PadValue,
    compact: CompactMode,
) -> None:
    """Cross-check PyPTO's ``BufferType`` against the derived cce-mlir encoding.

    A disagreement means one of the two backends has a different understanding
    of the buffer; refusing here keeps the mismatch from surfacing as a wrong
    program much later in the pipeline.

    Args:
        name: Buffer variable name, for diagnostics.
        space: Derived cce-mlir memory space.
        element: Element type.
        blayout: ``BufferType.blayout``.
        slayout: ``BufferType.slayout``.
        fractal: ``BufferType.fractal`` in bytes.
        pad: ``BufferType.pad``.
        compact: ``BufferType.compact``.

    Raises:
        CceMlirError: On a contradictory descriptor.
    """
    if pad is not PadValue.null:
        raise CceMlirError(
            "unsupported-padding",
            f"buffer {name!r} uses pad mode {pad.name}; the cce-mlir route does not model pad-fill semantics",
        )
    if compact is not CompactMode.null:
        raise CceMlirError(
            "unsupported-compaction",
            f"buffer {name!r} uses compact mode {compact.name}; the cce-mlir "
            "route only models dense and canonical-fractal buffers",
        )

    expected_layouts = _expected_block_layouts(space)
    if expected_layouts is not None:
        expected_blayout, expected_slayout = expected_layouts
        if blayout is not expected_blayout or slayout is not expected_slayout:
            raise CceMlirError(
                "layout-mismatch",
                f"buffer {name!r} in {space.value} reports layouts "
                f"(blayout={blayout.name}, slayout={slayout.name}) but cce-mlir's "
                f"{space.value} encoding requires "
                f"(blayout={expected_blayout.name}, slayout={expected_slayout.name})",
            )

    expected_fractal = _expected_fractal_bytes(space, element)
    if expected_fractal is not None and fractal != expected_fractal:
        raise CceMlirError(
            "layout-mismatch",
            f"buffer {name!r} in {space.value} reports fractal={fractal} bytes but "
            f"the {element.mlir} {space.value} encoding requires {expected_fractal}",
        )
