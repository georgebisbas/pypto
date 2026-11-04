# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Build an in-memory cce-mlir module from the typed kernel IR.

Every operation is constructed through cce-mlir's own ODS-generated builders;
there is no MLIR text template. Encodings, layouts, element types and symbols are
all supplied by the typed IR.
"""

from __future__ import annotations

from typing import Any

from pypto.pypto_core.ir import ParamDirection

from .errors import CceMlirError
from .kernel_ir import (
    CceElementType,
    CceEncoding,
    CceFor,
    CceIntTuple,
    CceKernel,
    CceKernelKind,
    CceMemorySpace,
    CceScalar,
    CceTarget,
    CceTensor,
    IndexValue,
)
from .registry import EMITTERS

__all__ = ["MlirBuilder", "emit_module"]

_SPACE_NAMES: dict[CceMemorySpace, str] = {
    CceMemorySpace.GM: "GM",
    CceMemorySpace.UB: "UB",
    CceMemorySpace.L1: "L1",
    CceMemorySpace.L0A: "L0A",
    CceMemorySpace.L0B: "L0B",
    CceMemorySpace.L0C: "L0C",
}

_TARGET_NAMES: dict[CceTarget, str] = {
    CceTarget.A3: "a3",
    CceTarget.A5: "a5",
}

_KIND_NAMES: dict[CceKernelKind, str] = {
    CceKernelKind.CUBE: "cube",
    CceKernelKind.VECTOR: "vector",
}


def _element_type(cir: Any, element: CceElementType) -> Any:
    """Map a typed element to an MLIR element type."""
    if element.mlir == "f16":
        return cir.F16Type.get()
    if element.mlir == "bf16":
        return cir.BF16Type.get()
    if element.mlir == "f32":
        return cir.F32Type.get()
    if element.mlir == "i8":
        return cir.IntegerType.get_signless(8)
    if element.mlir == "i32":
        return cir.IntegerType.get_signless(32)
    raise CceMlirError(
        "unsupported-dtype",
        f"element type {element.mlir!r} has no MLIR spelling in the emitter",
    )


class MlirBuilder:
    """Value/type bookkeeping and typed construction helpers for one kernel."""

    def __init__(self, cir: Any) -> None:
        self._cir = cir
        self._values: dict[int, Any] = {}
        # Enclosing loop induction variables, by PyPTO name, for address
        # normalization. A nested loop may shadow an outer name, so callers
        # restore the previous binding when the body is done.
        self._loop_indices: dict[str, Any] = {}

    # -- types ------------------------------------------------------------

    def tensor_type(self, tensor: CceTensor) -> Any:
        """MLIR ``RankedTensorType`` for a typed tensor."""
        return self.type_for(
            shape=tensor.shape,
            element=tensor.element,
            space=tensor.space,
            encoding=tensor.encoding,
        )

    def type_for(
        self,
        *,
        shape: tuple[int, ...],
        element: CceElementType,
        space: CceMemorySpace,
        encoding: CceEncoding,
    ) -> Any:
        """MLIR ranked tensor type from explicit components."""
        return self._cir.RankedTensorType.get(
            list(shape),
            _element_type(self._cir, element),
            self.encoding_attr(space, encoding),
        )

    def encoding_attr(self, space: CceMemorySpace, encoding: CceEncoding) -> Any:
        """cce-mlir tensor encoding attribute for a tier and layout."""
        from cce_mlir.dialects import npu

        npu_space = getattr(npu.MemorySpace, _SPACE_NAMES[space])
        layout = encoding.layout
        if layout is None:
            return npu.encoding(npu_space)
        layout_attr = npu.PackedLayoutAttr.get(
            list(layout.inner_dims_pos),
            list(layout.inner_tile_sizes),
            list(layout.outer_dims_perm),
        )
        return npu.encoding(npu_space, layout_attr)

    # -- values -----------------------------------------------------------

    def value(self, tensor: CceTensor) -> Any:
        """Current MLIR value for a typed tensor."""
        value = self._values.get(tensor.value_id)
        if value is None:
            raise CceMlirError(
                "unbound-value",
                f"tensor {tensor.name!r} has no MLIR value; the emitter ran out "
                "of order or skipped its definition",
            )
        return value

    def bind(self, tensor: CceTensor, value: Any) -> None:
        """Bind (or rebind) the MLIR value of a typed tensor."""
        self._values[tensor.value_id] = value

    # -- construction helpers --------------------------------------------

    def empty_like(self, tensor: CceTensor) -> Any:
        """Allocate a typed ``tensor.empty`` and return its value."""
        from cce_mlir.dialects import pto

        return pto.empty_like(self.tensor_type(tensor))

    def coerce(
        self,
        source: CceTensor,
        target: CceTensor,
        *,
        value: Any | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> Any:
        """Align a value's encoding with the target's encoding.

        Emits ``layout.pack_layout``, ``layout.repack_layout`` or
        ``layout.unpack_layout`` only when the encodings differ. The
        intermediate keeps the *source* memory space: a layout change is not a
        memory-space move, and the following transfer op performs the move.

        ``value`` overrides the source's bound value when the operation has
        already rewritten it (a windowed slice of the source), while the type
        and encoding still come from ``source``. ``shape`` overrides the
        result shape for a windowed transfer into or out of a larger tensor.
        """
        from cce_mlir.dialects import layout

        value = self.value(source) if value is None else value
        if source.encoding == target.encoding:
            return value
        effective_shape = target.shape if shape is None else shape
        intermediate_type = self.type_for(
            shape=effective_shape,
            element=target.element,
            space=source.space,
            encoding=target.encoding,
        )
        if source.encoding.is_dense:
            return layout.PackLayoutOp(intermediate_type, value).result
        if target.encoding.is_dense:
            return layout.UnpackLayoutOp(intermediate_type, value).result
        return layout.RepackLayoutOp(intermediate_type, value).result

    # -- address and window construction ------------------------------------

    def bind_loop_index(self, name: str, value: Any) -> Any | None:
        """Bind an enclosing loop's induction variable; return the previous value."""
        previous = self._loop_indices.get(name)
        self._loop_indices[name] = value
        return previous

    def restore_loop_index(self, name: str, previous: Any | None) -> None:
        """Undo :meth:`bind_loop_index` for one name."""
        if previous is None:
            self._loop_indices.pop(name, None)
        else:
            self._loop_indices[name] = previous

    def index_value(self, value: IndexValue) -> Any:
        """Materialize an ``index`` scalar for a constant or affine expression."""
        from cce_mlir.dialects import arith

        index = self._cir.IndexType.get()
        if isinstance(value, int):
            return arith.constant(index, value)
        if value.is_constant:
            return arith.constant(index, value.constant)
        result = arith.constant(index, value.constant)
        for name, multiplier in value.terms:
            induction = self._loop_indices.get(name)
            if induction is None:
                raise CceMlirError(
                    "unbound-loop-variable",
                    f"offset expression references {name!r}, which is not an "
                    "enclosing loop variable at the point of use",
                )
            term = induction
            if multiplier != 1:
                term = arith.muli(induction, arith.constant(index, multiplier))
            result = arith.addi(result, term)
        return result

    def scalar_value(self, scalar: CceScalar) -> Any:
        """Materialize a compile-time scalar with its own element type."""
        from cce_mlir.dialects import arith

        return arith.constant(_element_type(self._cir, scalar.element), scalar.value)

    def _slice_arguments(self, offsets: CceIntTuple, sizes: tuple[int, ...]) -> tuple[list[Any], list[int]]:
        """Split a window into dynamic offset operands and static offsets."""
        dynamic_size = self._cir.ShapedType.get_dynamic_size()
        dynamic_offsets: list[Any] = []
        static_offsets: list[int] = []
        for offset in offsets.values:
            if isinstance(offset, int):
                static_offsets.append(offset)
                continue
            if offset.is_constant:
                static_offsets.append(offset.constant)
                continue
            dynamic_offsets.append(self.index_value(offset))
            static_offsets.append(dynamic_size)
        return dynamic_offsets, static_offsets

    def window(
        self,
        tensor: CceTensor,
        offsets: CceIntTuple,
        sizes: tuple[int, ...],
    ) -> Any:
        """Slice ``tensor``'s current value to ``sizes`` at ``offsets``.

        ``tensor.extract_slice`` carries the window; cce-mlir's PTO front end
        folds the slice into the transfer that consumes it, so no copy is
        materialized.
        """
        from cce_mlir.dialects import tensor as tensor_dialect

        dynamic_offsets, static_offsets = self._slice_arguments(offsets, sizes)
        result_type = self.type_for(
            shape=sizes,
            element=tensor.element,
            space=tensor.space,
            encoding=tensor.encoding,
        )
        return tensor_dialect.extract_slice(
            result_type,
            self.value(tensor),
            dynamic_offsets,
            [],
            [],
            static_offsets,
            list(sizes),
            [1] * len(sizes),
        )

    def insert_window(
        self,
        tensor: CceTensor,
        window_value: Any,
        offsets: CceIntTuple,
        sizes: tuple[int, ...],
    ) -> Any:
        """Write ``window_value`` back into ``tensor``'s value at ``offsets``.

        A store produces a new window value; the parent GM tensor is an SSA
        value, so it must be rebuilt with ``tensor.insert_slice`` before later
        uses observe the write. Bufferization folds both slice ops into a
        strided subview of the GM allocation.
        """
        from cce_mlir.dialects import tensor as tensor_dialect

        dynamic_offsets, static_offsets = self._slice_arguments(offsets, sizes)
        return tensor_dialect.insert_slice(
            window_value,
            self.value(tensor),
            dynamic_offsets,
            [],
            [],
            static_offsets,
            list(sizes),
            [1] * len(sizes),
        )


def emit_module(
    kernel: CceKernel,
    *,
    with_host: bool = True,
    host_device: int = 0,
) -> Any:
    """Build and verify the in-memory cce-mlir module for one kernel.

    Args:
        kernel: Typed kernel IR from :func:`pypto.backend.cce_mlir.lower_program`.
        with_host: Append the standalone host driver ``@run``. cce-mlir's full
            PTO-to-LLVM pipeline needs host code referencing the kernel; the
            PyPTO-runtime product only extracts the device object, so the host
            driver is optional there.
        host_device: Logical device index used by the standalone host driver.
            The queue re-indexes granted cards from zero, so the driver passes
            ``TASK_DEVICE`` (default ``0``); the production path does not use
            this host function at all.

    Returns:
        A verified ``cce_mlir.ir.Module``.

    Raises:
        CceMlirError: When an operation has no registered emitter.
        ImportError: When the external ``cce_mlir`` package is unavailable.
    """
    from cce_mlir import ir as cir
    from cce_mlir.dialects import npu

    # Importing registers every built-in emitter.
    from . import emitters as _emitters  # noqa: F401, PLC0415
    from ._dialects import make_context

    context = make_context()
    with context, cir.Location.unknown():
        module = cir.Module.create()
        builder = MlirBuilder(cir)
        with cir.InsertionPoint(module.body):
            npu_module = npu.NPUModuleOp(
                kernel.module_name,
                target=getattr(npu.KernelTarget, _TARGET_NAMES[kernel.target]),
            )
            with cir.InsertionPoint(npu_module.body):
                input_types = [builder.tensor_type(tensor) for tensor in kernel.parameters]
                result_types = [builder.tensor_type(tensor) for tensor in kernel.returns]
                function = npu.NPUFuncOp(
                    kernel.symbol_name,
                    (input_types, result_types),
                    kernel=True,
                    kind=getattr(npu.KernelKind, _KIND_NAMES[kernel.kind]),
                    visibility="private",
                )
                with cir.InsertionPoint(function.add_entry_block()):
                    arguments = list(function.body.blocks[0].arguments)
                    for tensor, value in zip(kernel.parameters, arguments, strict=True):
                        builder.bind(tensor, value)
                    _emit_statements(kernel.body, builder, cir)
                    npu.ReturnOp([builder.value(tensor) for tensor in kernel.returns])
            if with_host:
                _emit_host_driver(cir, kernel, builder, module, host_device)
        module.operation.verify()
        return module


def _written_tensors(statements: Any) -> list[CceTensor]:
    """Tensors written anywhere in a statement block, in first-write order."""
    written: dict[int, CceTensor] = {}

    def visit(block: Any) -> None:
        for statement in block:
            if isinstance(statement, CceFor):
                visit(statement.body)
                continue
            tensor = statement.write_tensor
            if tensor is not None:
                written.setdefault(tensor.value_id, tensor)

    visit(statements)
    return list(written.values())


def _emit_statements(statements: Any, builder: MlirBuilder, cir: Any) -> None:
    """Emit a statement block, lowering loops to ``scf.for``.

    PyPTO's Buffer IR keeps loop state in buffers, so every tensor written in a
    loop body is threaded as an ``scf.for`` iter-arg: its pre-loop value enters,
    the block argument is bound for the body, and the loop result is bound after
    it. Without this the post-loop store would reference a value defined inside
    the loop region and fail dominance.
    """
    from cce_mlir.dialects import arith, scf

    for statement in statements:
        if isinstance(statement, CceFor):
            carried = _written_tensors(statement.body)
            initial = [builder.value(tensor) for tensor in carried]
            index = cir.IndexType.get()
            loop = scf.ForOp(
                arith.constant(index, statement.start),
                arith.constant(index, statement.stop),
                arith.constant(index, statement.step),
                initial,
            )
            with cir.InsertionPoint(loop.body):
                # Block arguments are the induction variable followed by the
                # iter-args, one per carried tensor. Offsets inside the body may
                # be affine in the induction variable, so bind it for the body.
                body_arguments = list(loop.body.arguments)
                induction = body_arguments[0]
                previous_index = builder.bind_loop_index(statement.loop_var, induction)
                try:
                    for tensor, argument in zip(carried, body_arguments[1:], strict=True):
                        builder.bind(tensor, argument)
                    _emit_statements(statement.body, builder, cir)
                    scf.YieldOp([builder.value(tensor) for tensor in carried])
                finally:
                    builder.restore_loop_index(statement.loop_var, previous_index)
            for tensor, result in zip(carried, loop.results, strict=True):
                builder.bind(tensor, result)
            continue
        emitter = EMITTERS.get(statement.op_name)
        if emitter is None:
            raise CceMlirError(
                "unsupported-op",
                f"no cce-mlir emitter is registered for {statement.op_name!r}",
                op_name=statement.op_name,
                span=statement.span,
            )
        emitter(statement, builder)


def _npu_memory_space(tensor: CceTensor) -> Any:
    """The ``#npu.memory_space`` attribute for a typed tensor."""
    from cce_mlir.dialects import npu

    return npu.MemorySpaceAttr.get(int(getattr(npu.MemorySpace, _SPACE_NAMES[tensor.space])))


def _emit_host_driver(
    cir: Any,
    kernel: CceKernel,
    builder: MlirBuilder,
    module: Any,
    device_index: int,
) -> None:
    """Append the standalone ``@run`` host driver.

    The driver stages every input on the device, launches the kernel over all
    parameters, copies outputs back and tears the stream down. Sizes come from
    the typed tensors; the block count is one because the route refuses SPMD
    block queries.
    """
    from cce_mlir.dialects import arith, func, npu

    i32 = cir.IntegerType.get_signless(32)
    i64 = cir.IntegerType.get_signless(64)
    host_types = [
        cir.MemRefType.get(list(tensor.shape), _element_type(cir, tensor.element))
        for tensor in kernel.parameters
    ]
    with cir.InsertionPoint(module.body):
        driver = func.FuncOp("run", func.FunctionType.get(host_types, []))
        driver.attributes["llvm.emit_c_interface"] = cir.UnitAttr.get()
        with cir.InsertionPoint(driver.add_entry_block()):
            npu.InitOp()
            device = npu.SetDeviceOp(npu.DeviceType.get(), arith.constant(i32, device_index)).result
            stream = npu.CreateStreamOp(npu.StreamType.get()).result
            device_buffers = []
            for tensor, argument in zip(kernel.parameters, driver.arguments, strict=True):
                device_type = cir.MemRefType.get(
                    list(tensor.shape),
                    _element_type(cir, tensor.element),
                    memory_space=_npu_memory_space(tensor),
                )
                buffer = npu.AllocOp(device_type, arith.constant(i64, tensor.byte_size)).result
                if tensor.direction in (ParamDirection.In, ParamDirection.InOut):
                    npu.MemcpyOp(None, buffer, argument, stream)
                device_buffers.append(buffer)
            npu.LaunchFuncOp(
                cir.SymbolRefAttr.get([kernel.module_name, kernel.symbol_name]),
                arith.constant(i32, 1),
                stream,
                device_buffers,
            )
            npu.SynchronizeStreamOp(stream)
            for tensor, argument, buffer in zip(
                kernel.parameters, driver.arguments, device_buffers, strict=True
            ):
                if tensor.direction in (ParamDirection.Out, ParamDirection.InOut):
                    npu.MemcpyOp(None, argument, buffer, stream)
            npu.SynchronizeStreamOp(stream)
            for buffer in device_buffers:
                npu.DeallocOp(buffer)
            npu.DestroyStreamOp(stream)
            npu.ResetDeviceOp(device)
            npu.FinalizeOp()
            func.ReturnOp([])


def render(kernel: CceKernel) -> str:
    """Build the module and return its canonical text."""
    return str(emit_module(kernel))
