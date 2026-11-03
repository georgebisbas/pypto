# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Built-in cce-mlir emitters for the supported PyPTO buffer operations.

Emitter selection is by registered PyPTO op name; operand roles come from the
typed op (`write_tensor` / `read_tensors`) and memory spaces are used to
identify matmul operands, never positions.
"""

from __future__ import annotations

from typing import Any

from cce_mlir.dialects import pto

from .kernel_ir import CceMemorySpace, CceOp
from .registry import EMITTERS

__all__: list[str] = []


@EMITTERS.register("buffer.alloc")
def _emit_alloc(op: CceOp, builder: Any) -> None:
    tensor = op.result
    assert tensor is not None, "buffer.alloc without a result tensor"
    builder.bind(tensor, builder.empty_like(tensor))


@EMITTERS.register("buffer.load")
def _emit_load(op: CceOp, builder: Any) -> None:
    destination = op.write_tensor
    assert destination is not None
    source = op.read_tensors[0]
    value = pto.LoadOp(
        builder.tensor_type(destination),
        builder.coerce(source, destination),
        builder.value(destination),
    ).result
    builder.bind(destination, value)


@EMITTERS.register("buffer.store")
def _emit_store(op: CceOp, builder: Any) -> None:
    destination = op.write_tensor
    assert destination is not None
    source = op.read_tensors[0]
    value = pto.StoreOp(
        builder.tensor_type(destination),
        builder.coerce(source, destination),
        builder.value(destination),
    ).result
    builder.bind(destination, value)


@EMITTERS.register("buffer.copy")
def _emit_copy(op: CceOp, builder: Any) -> None:
    destination = op.write_tensor
    assert destination is not None
    source = op.read_tensors[0]
    value = pto.MovOp(
        builder.tensor_type(destination),
        builder.coerce(source, destination),
        builder.value(destination),
    ).result
    builder.bind(destination, value)


@EMITTERS.register("buffer.matmul")
def _emit_matmul(op: CceOp, builder: Any) -> None:
    destination = op.write_tensor
    assert destination is not None
    left = next(tensor for tensor in op.read_tensors if tensor.space is CceMemorySpace.L0A)
    right = next(tensor for tensor in op.read_tensors if tensor.space is CceMemorySpace.L0B)
    value = pto.MatmulOp(
        builder.tensor_type(destination),
        builder.value(left),
        builder.value(right),
        builder.value(destination),
    ).result
    builder.bind(destination, value)


@EMITTERS.register("buffer.matmul_acc")
def _emit_matmul_acc(op: CceOp, builder: Any) -> None:
    destination = op.write_tensor
    assert destination is not None
    left = next(tensor for tensor in op.read_tensors if tensor.space is CceMemorySpace.L0A)
    right = next(tensor for tensor in op.read_tensors if tensor.space is CceMemorySpace.L0B)
    value = pto.MatmulAccOp(
        builder.tensor_type(destination),
        builder.value(left),
        builder.value(right),
        builder.value(destination),
    ).result
    builder.bind(destination, value)


_BINARY_ELEMENTWISE = {
    "buffer.add": pto.AddOp,
    "buffer.sub": pto.SubOp,
    "buffer.mul": pto.MulOp,
    "buffer.div": pto.DivOp,
    "buffer.maximum": pto.MaxOp,
    "buffer.minimum": pto.MinOp,
}

_UNARY_ELEMENTWISE = {
    "buffer.abs": pto.AbsOp,
    "buffer.exp": pto.ExpOp,
    "buffer.sqrt": pto.SqrtOp,
    "buffer.neg": pto.NegOp,
    "buffer.relu": pto.ReluOp,
    "buffer.log": pto.LogOp,
    "buffer.recip": pto.RecipOp,
}


def _elementwise_emitter(constructor: Any, arity: int) -> Any:
    """Build an emitter that forwards ``arity`` sources plus the destination."""

    def emit(op: CceOp, builder: Any) -> None:
        destination = op.write_tensor
        assert destination is not None
        sources = [builder.value(tensor) for tensor in op.read_tensors[:arity]]
        # PyPTO's Buffer IR reuses the destination tile as the first source for
        # in-place elementwise ops. cce-mlir's destination-passing lowering does
        # not preserve that aliasing, so give the op a distinct destination and
        # let the typed value binding forward the result.
        aliased = any(tensor.value_id == destination.value_id for tensor in op.read_tensors[:arity])
        destination_value = builder.empty_like(destination) if aliased else builder.value(destination)
        value = constructor(builder.tensor_type(destination), *sources, destination_value).result
        builder.bind(destination, value)

    return emit


for _name, _constructor in _BINARY_ELEMENTWISE.items():
    EMITTERS.register(_name)(_elementwise_emitter(_constructor, 2))
for _name, _constructor in _UNARY_ELEMENTWISE.items():
    EMITTERS.register(_name)(_elementwise_emitter(_constructor, 1))
