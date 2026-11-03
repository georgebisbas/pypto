# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Typed, backend-independent representation of one cce-mlir kernel.

This layer has no dependency on the external ``cce_mlir`` Python package: it is
built from PyPTO Buffer IR and PyPTO's own backend metadata, and every unit test
of the lowering runs without the cce-mlir toolchain.

Values are immutable and carry a unique ``value_id``; the lowering threads a
new value through every write, mirroring cce-mlir's destination-passing SSA.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from math import prod

from pypto.pypto_core import DataType
from pypto.pypto_core.ir import ParamDirection

from .errors import CceMlirError

__all__ = [
    "CceElementType",
    "CceEncoding",
    "CceKernel",
    "CceKernelKind",
    "CceLayout",
    "CceMemorySpace",
    "CceOp",
    "CceScalar",
    "CceTarget",
    "CceTensor",
]

# MLIR spelling and C spelling of every data type the route accepts. The table
# is the single source of truth: analysis refuses anything absent from it.
_ELEMENT_TYPES: dict[str, tuple[str, str]] = {
    "fp16": ("f16", "half"),
    "bfloat16": ("bf16", "bfloat16_t"),
    "fp32": ("f32", "float"),
    "int8": ("i8", "int8_t"),
    "int32": ("i32", "int32_t"),
}


class CceMemorySpace(enum.Enum):
    """Memory tiers named by cce-mlir's ``#npu.memory_space`` attribute."""

    GM = "gm"
    UB = "ub"
    L1 = "l1"
    L0A = "l0a"
    L0B = "l0b"
    L0C = "l0c"


class CceKernelKind(enum.Enum):
    """cce-mlir ``#npu.kernel_kind``."""

    CUBE = "cube"
    VECTOR = "vector"


class CceTarget(enum.Enum):
    """cce-mlir ``#npu.kernel_target``."""

    A3 = "a3"
    A5 = "a5"


@dataclass(frozen=True)
class CceElementType:
    """Element type as spelled by MLIR and the device C ABI."""

    mlir: str
    c_type: str
    bits: int

    @property
    def byte_size(self) -> int:
        """Whole bytes occupied by one element."""
        return (self.bits + 7) // 8

    @classmethod
    def from_data_type(cls, dtype: DataType) -> CceElementType:
        """Map a PyPTO ``DataType`` onto the cce-mlir element vocabulary.

        Args:
            dtype: The PyPTO element type.

        Returns:
            The matching :class:`CceElementType`.

        Raises:
            CceMlirError: When the dtype has no cce-mlir spelling yet.
        """
        name = dtype.to_string()
        spelled = _ELEMENT_TYPES.get(name)
        if spelled is None:
            supported = ", ".join(sorted(_ELEMENT_TYPES))
            raise CceMlirError(
                "unsupported-dtype",
                f"dtype {name!r} is not supported by the cce-mlir route (supported: {supported})",
            )
        mlir, c_type = spelled
        return cls(mlir=mlir, c_type=c_type, bits=dtype.get_bit())


@dataclass(frozen=True)
class CceLayout:
    """One canonical cce-mlir ``#npu.layout`` packing."""

    inner_dims_pos: tuple[int, ...]
    inner_tile_sizes: tuple[int, ...]
    outer_dims_perm: tuple[int, ...]

    def __str__(self) -> str:
        return (
            "#npu.layout<inner_dims_pos = "
            f"{list(self.inner_dims_pos)}, inner_tile_sizes = "
            f"{list(self.inner_tile_sizes)}, outer_dims_perm = "
            f"{list(self.outer_dims_perm)}>"
        )


@dataclass(frozen=True)
class CceEncoding:
    """Tensor encoding: bare memory space, or memory space plus layout."""

    layout: CceLayout | None = None

    @property
    def is_dense(self) -> bool:
        """Whether the encoding is the identity (ND) layout."""
        return self.layout is None

    def __str__(self) -> str:
        return "dense" if self.layout is None else str(self.layout)


@dataclass(frozen=True)
class CceTensor:
    """One physical tensor value in the kernel.

    ``value_id`` distinguishes successive writes to the same buffer; ``name`` is
    the originating PyPTO variable name and is used only for diagnostics.
    """

    value_id: int
    name: str
    shape: tuple[int, ...]
    element: CceElementType
    space: CceMemorySpace
    encoding: CceEncoding = field(default_factory=CceEncoding)
    direction: ParamDirection | None = None

    @property
    def is_parameter(self) -> bool:
        """Whether this value is a kernel parameter."""
        return self.direction is not None

    @property
    def byte_size(self) -> int:
        """Dense byte size of the tensor: shape product times element size."""
        return prod(self.shape) * self.element.byte_size

    @property
    def rank(self) -> int:
        """Number of dimensions."""
        return len(self.shape)


@dataclass(frozen=True)
class CceScalar:
    """A compile-time scalar operand (elementwise scalar variants)."""

    value: int | float
    element: CceElementType


@dataclass(frozen=True)
class CceIntTuple:
    """A ``(Int, Int, ...)`` operand such as offsets or valid extents."""

    values: tuple[int, ...]


@dataclass(frozen=True)
class CceOp:
    """One operation, keyed by its registered PyPTO buffer-op name.

    Operand *roles* are captured from ``ir.get_op_buffer_arg_effect`` during
    lowering, so emitters never depend on operand positions.

    ``result`` is set only for ``buffer.alloc``; every other buffer op mutates
    the operand at ``write_index``.
    """

    op_name: str
    operands: tuple[CceTensor | CceScalar | CceIntTuple, ...]
    write_index: int | None = None
    read_indices: tuple[int, ...] = ()
    result: CceTensor | None = None
    span: object | None = None

    @property
    def tensors(self) -> tuple[CceTensor, ...]:
        """Tensor operands in operand order."""
        return tuple(operand for operand in self.operands if isinstance(operand, CceTensor))

    @property
    def write_tensor(self) -> CceTensor | None:
        """The tensor operand carrying write access, or ``None``."""
        if self.write_index is None:
            return None
        operand = self.operands[self.write_index]
        assert isinstance(operand, CceTensor)
        return operand

    @property
    def read_tensors(self) -> tuple[CceTensor, ...]:
        """Tensor operands carrying read access, in operand order."""
        return tuple(self.operands[index] for index in self.read_indices)  # type: ignore[return-value]


@dataclass(frozen=True)
class CceKernel:
    """One lowered InCore kernel."""

    function_name: str
    module_name: str
    symbol_name: str
    device_symbol: str
    target: CceTarget
    kind: CceKernelKind
    parameters: tuple[CceTensor, ...]
    values: tuple[CceTensor, ...]
    ops: tuple[CceOp, ...]
    returns: tuple[CceTensor, ...]
