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

from .kernel_ir import CceMemorySpace, CceOp, CceScalar, CceTensor
from .registry import EMITTERS

__all__: list[str] = []


def _full_window(op: CceOp, gm_tensor: CceTensor) -> bool:
    """Whether the transfer covers the whole GM tensor at zero offsets.

    The extents are validated to equal the on-chip buffer's shape, so a
    full-window transfer additionally requires them to equal the GM tensor's
    shape; anything narrower needs a slice even at zero offsets.
    """
    if op.offsets is None or op.extents is None:
        return True
    if tuple(op.extents.values) != gm_tensor.shape:
        return False
    for offset in op.offsets.values:
        if isinstance(offset, int):
            if offset != 0:
                return False
        elif not (offset.is_constant and offset.constant == 0):
            return False
    return True


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
    if _full_window(op, source):
        value = builder.value(source)
    else:
        # The window is carried by the source slice; ``coerce`` then expresses
        # the layout change at the transfer, which folds both into one DMA.
        value = builder.window(source, op.offsets, destination.shape)
    packed = builder.coerce(source, destination, value=value)
    loaded = pto.LoadOp(builder.tensor_type(destination), packed, builder.value(destination)).result
    builder.bind(destination, loaded)


@EMITTERS.register("buffer.store")
def _emit_store(op: CceOp, builder: Any) -> None:
    destination = op.write_tensor
    assert destination is not None
    source = op.read_tensors[0]
    if _full_window(op, destination):
        value = pto.StoreOp(
            builder.tensor_type(destination),
            builder.coerce(source, destination),
            builder.value(destination),
        ).result
        builder.bind(destination, value)
        return
    # A partial store writes a window of the GM tensor. The value produced by
    # ``pto.store`` is that window, so it must be spliced back into the parent
    # before any later use of the parameter observes the write.
    window = builder.window(destination, op.offsets, source.shape)
    window_type = builder.type_for(
        shape=source.shape,
        element=destination.element,
        space=destination.space,
        encoding=destination.encoding,
    )
    stored = pto.StoreOp(
        window_type,
        builder.coerce(source, destination, shape=source.shape),
        window,
    ).result
    builder.bind(destination, builder.insert_window(destination, stored, op.offsets, source.shape))


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
        # An in-place update (PyPTO's Buffer IR reuses the destination as a
        # source). cce-mlir's destination-passing lowering does not preserve
        # dest == src (it miscomputes), so compute into a distinct scratch and
        # copy the result into the destination buffer. Writing the destination
        # buffer keeps a loop-carried value equivalent to its iter argument.
        aliased = any(tensor.value_id == destination.value_id for tensor in op.read_tensors[:arity])
        if not aliased:
            value = constructor(builder.tensor_type(destination), *sources, builder.value(destination)).result
        elif builder.in_loop:
            # Allocating inside the loop body is rejected, and a loop carry must
            # yield the destination buffer itself: compute into a scratch
            # allocated at entry, then copy the result back into the buffer.
            computed = constructor(builder.tensor_type(destination), *sources, builder.scratch(op)).result
            value = pto.MovOp(builder.tensor_type(destination), computed, builder.value(destination)).result
        else:
            value = constructor(
                builder.tensor_type(destination), *sources, builder.empty_like(destination)
            ).result
        builder.bind(destination, value)

    return emit


for _name, _constructor in _BINARY_ELEMENTWISE.items():
    EMITTERS.register(_name)(_elementwise_emitter(_constructor, 2))
for _name, _constructor in _UNARY_ELEMENTWISE.items():
    EMITTERS.register(_name)(_elementwise_emitter(_constructor, 1))


_SCALAR_ELEMENTWISE = {
    "buffer.adds": pto.AddSOp,
    "buffer.subs": pto.SubSOp,
    "buffer.muls": pto.MulSOp,
    "buffer.divs": pto.DivSOp,
    "buffer.maximums": pto.MaxSOp,
    "buffer.minimums": pto.MinSOp,
    "buffer.lrelu": pto.LreluOp,
}


def _scalar_emitter(constructor: Any) -> Any:
    """Build an emitter for a one-tile ``op(src, scalar, dst)`` variant."""

    def emit(op: CceOp, builder: Any) -> None:
        destination = op.write_tensor
        assert destination is not None
        source = op.read_tensors[0]
        (scalar,) = [operand for operand in op.operands if isinstance(operand, CceScalar)]
        # In-place update: compute into a scratch, and inside a loop copy the
        # result into the destination buffer (see the elementwise emitter above).
        if source.value_id == destination.value_id and builder.in_loop:
            computed = constructor(
                builder.tensor_type(destination),
                builder.coerce(source, destination),
                builder.scalar_value(scalar),
                builder.scratch(op),
            ).result
            value = pto.MovOp(builder.tensor_type(destination), computed, builder.value(destination)).result
        elif source.value_id == destination.value_id:
            value = constructor(
                builder.tensor_type(destination),
                builder.coerce(source, destination),
                builder.scalar_value(scalar),
                builder.empty_like(destination),
            ).result
        else:
            value = constructor(
                builder.tensor_type(destination),
                builder.coerce(source, destination),
                builder.scalar_value(scalar),
                builder.value(destination),
            ).result
        builder.bind(destination, value)

    return emit


for _name, _constructor in _SCALAR_ELEMENTWISE.items():
    EMITTERS.register(_name)(_scalar_emitter(_constructor))
