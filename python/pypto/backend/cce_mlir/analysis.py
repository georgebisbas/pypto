# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Lower PyPTO Buffer-stage IR into the typed cce-mlir kernel IR.

The lowering is deliberately structural rather than positional:

* operand *roles* come from the operator registry
  (``ir.get_op_buffer_arg_effect``), not from an index table;
* memory tiers come from ``BufferType.memory_space``;
* layouts come from :mod:`pypto.backend.cce_mlir.layout`, cross-checked against
  the ``BufferType`` descriptor;
* parameter directions come from ``Function.param_directions``, never names.

Anything outside the documented contract raises :class:`CceMlirError` with a
stable reason, so an unsupported kernel fails here instead of emitting a module
that is subtly wrong.
"""

from __future__ import annotations

from collections.abc import Iterable

import pypto.ir as _pypto_ir
from pypto.pypto_core import backend as _backend_core
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import (
    AssignStmt,
    BufferAccess,
    BufferType,
    ConstFloat,
    ConstInt,
    EvalStmt,
    FunctionType,
    MakeTuple,
    ParamDirection,
    ReturnStmt,
    SeqStmts,
    TensorType,
    Var,
)

from .errors import CceMlirError
from .kernel_ir import (
    CceElementType,
    CceIntTuple,
    CceKernel,
    CceKernelKind,
    CceMemorySpace,
    CceOp,
    CceScalar,
    CceTarget,
    CceTensor,
)
from .layout import check_buffer_descriptor, encoding_for, memory_space

__all__ = ["lower_function", "lower_program", "resolve_target"]

_TARGETS: dict[str, CceTarget] = {"a2a3": CceTarget.A3, "a5": CceTarget.A5}

# Buffer-stage ops this milestone understands. Membership is checked against the
# registry stage at lowering time; emission adds the semantic mapping.
_SUPPORTED_OPS = frozenset(
    {
        "buffer.alloc",
        "buffer.load",
        "buffer.store",
        "buffer.copy",
        "buffer.matmul",
        "buffer.matmul_acc",
        "buffer.add",
        "buffer.sub",
        "buffer.mul",
        "buffer.div",
        "buffer.maximum",
        "buffer.minimum",
        "buffer.abs",
        "buffer.exp",
        "buffer.sqrt",
        "buffer.neg",
        "buffer.relu",
        "buffer.log",
        "buffer.recip",
    }
)

_CUBE_SPACES = frozenset({CceMemorySpace.L1, CceMemorySpace.L0A, CceMemorySpace.L0B, CceMemorySpace.L0C})


def resolve_target() -> CceTarget:
    """Resolve the cce-mlir target from the configured PyPTO backend handler.

    Returns:
        ``CceTarget.A3`` for the ``a2a3`` handler, ``CceTarget.A5`` for ``a5``.

    Raises:
        CceMlirError: When no backend is configured or the arch is unknown.
    """
    try:
        arch = _backend_core.get_handler().get_pto_target_arch()
    except Exception as exc:  # nanobind raises RuntimeError when unconfigured
        raise CceMlirError(
            "unconfigured-backend",
            "no PyPTO backend is configured; select a BackendType before lowering",
        ) from exc
    target = _TARGETS.get(arch)
    if target is None:
        supported = ", ".join(sorted(_TARGETS))
        raise CceMlirError(
            "unsupported-target",
            f"backend PTO target arch {arch!r} has no cce-mlir target (supported: {supported})",
        )
    return target


def lower_program(
    program: _ir_core.Program,
    *,
    target: CceTarget | None = None,
) -> tuple[CceKernel, ...]:
    """Lower every Buffer-stage InCore function in a program.

    Args:
        program: The post-pass program produced by ``JITFunction.lower()`` or
            ``ir.compile``'s pipeline.
        target: cce-mlir target. ``None`` resolves it from the configured
            PyPTO backend.

    Returns:
        One :class:`CceKernel` per Buffer-stage InCore function.

    Raises:
        CceMlirError: When the program has no eligible function, or any
            function violates the supported contract.
    """
    effective_target = target if target is not None else resolve_target()
    functions = [
        function
        for function in program.functions.values()
        if function.ir_stage == _ir_core.FunctionIRStage.Buffer
        and _pypto_ir.is_incore_type(function.func_type)
    ]
    if not functions:
        raise CceMlirError(
            "no-buffer-incore-function",
            "program contains no Buffer-stage InCore function; lower with "
            "enable_buffer_ir=True before selecting the cce-mlir route",
        )
    return tuple(lower_function(function, target=effective_target) for function in functions)


def lower_function(
    function: _ir_core.Function,
    *,
    target: CceTarget,
) -> CceKernel:
    """Lower one Buffer-stage InCore function.

    Args:
        function: A function with ``ir_stage == FunctionIRStage.Buffer`` that
            ``ir.is_incore_type`` accepts.
        target: cce-mlir target.

    Returns:
        The typed kernel IR.

    Raises:
        CceMlirError: On any contract violation.
    """
    if function.ir_stage != _ir_core.FunctionIRStage.Buffer:
        raise CceMlirError(
            "wrong-ir-stage",
            f"function {function.name!r} is at {function.ir_stage.name}, expected Buffer",
        )
    if not _pypto_ir.is_incore_type(function.func_type):
        raise CceMlirError(
            "wrong-function-type",
            f"function {function.name!r} has type {function.func_type.name}, "
            "expected an InCore/AIC/AIV kernel",
        )
    return _KernelLowering(function, target).build()


def _static_extents(shape: Iterable[object], *, where: str) -> tuple[int, ...]:
    """Convert a sequence of ``ConstInt`` extents into positive Python ints."""
    extents: list[int] = []
    for index, extent in enumerate(shape):
        # ``TensorType`` shapes are ``ConstInt`` expressions; ``BufferType``
        # shapes are already plain Python ints.
        if isinstance(extent, ConstInt):
            value = extent.value
        elif isinstance(extent, int):
            value = extent
        else:
            raise CceMlirError(
                "dynamic-shape",
                f"{where} dimension {index} is {type(extent).__name__}; the "
                "cce-mlir route requires static shapes",
            )
        if value <= 0:
            raise CceMlirError(
                "invalid-shape",
                f"{where} dimension {index} must be positive, got {value}",
            )
        extents.append(value)
    return tuple(extents)


def _int_tuple(expr: object, *, where: str) -> tuple[int, ...]:
    """Read a ``MakeTuple`` of integer constants."""
    if not isinstance(expr, MakeTuple):
        raise CceMlirError(
            "malformed-operand",
            f"{where} expects a tuple of integer constants",
        )
    values: list[int] = []
    for element in expr.elements:
        if not isinstance(element, ConstInt):
            raise CceMlirError(
                "malformed-operand",
                f"{where} expects integer constants, got {type(element).__name__}",
            )
        values.append(element.value)
    return tuple(values)


def _direction_name(direction: ParamDirection) -> str:
    return direction.name


class _KernelLowering:
    """Stateful walk over one Buffer-stage function."""

    def __init__(self, function: _ir_core.Function, target: CceTarget) -> None:
        self._function = function
        self._target = target
        self._next_value_id = 0
        self._tensors: dict[str, CceTensor] = {}
        self._parameters: list[CceTensor] = []
        self._values: list[CceTensor] = []
        self._ops: list[CceOp] = []
        self._stored_parameters: dict[str, int] = {}
        self._written: set[str] = set()
        self._return_names: tuple[str, ...] = ()

    # -- construction -----------------------------------------------------

    def build(self) -> CceKernel:
        """Walk the function and assemble the kernel IR."""
        self._bind_parameters()
        self._visit(self._function.body)
        self._validate_returns()
        kind = self._derive_kind()
        symbol = _symbol_name(self._function.name)
        return CceKernel(
            function_name=self._function.name,
            module_name=f"{symbol}_module",
            symbol_name=symbol,
            device_symbol=f"{symbol}__kernel0",
            target=self._target,
            kind=kind,
            parameters=tuple(self._parameters),
            values=tuple(self._values),
            ops=tuple(self._ops),
            returns=tuple(
                self._tensors[name]
                for name, direction in self._param_directions()
                if direction is ParamDirection.Out
            ),
        )

    def _bind_parameters(self) -> None:
        for parameter, direction in zip(self._function.params, self._function.param_directions, strict=True):
            tensor_type = parameter.type
            if not isinstance(tensor_type, TensorType):
                raise CceMlirError(
                    "unsupported-parameter",
                    f"parameter {parameter.name_hint!r} has type "
                    f"{type(tensor_type).__name__}; only tensor parameters are "
                    "supported in this milestone",
                )
            shape = _static_extents(tensor_type.shape, where=f"parameter {parameter.name_hint!r}")
            if tensor_type.memory_space is not _ir_core.MemorySpace.DDR:
                raise CceMlirError(
                    "unsupported-parameter",
                    f"parameter {parameter.name_hint!r} lives in "
                    f"{tensor_type.memory_space.name}; kernel parameters must be DDR",
                )
            element = CceElementType.from_data_type(tensor_type.dtype)
            tensor = self._new_value(
                name=parameter.name_hint,
                shape=shape,
                element=element,
                space=CceMemorySpace.GM,
                direction=direction,
            )
            self._parameters.append(tensor)

    def _new_value(
        self,
        *,
        name: str,
        shape: tuple[int, ...],
        element: CceElementType,
        space: CceMemorySpace,
        direction: ParamDirection | None = None,
        check_descriptor: tuple[object, object, object, int, object] | None = None,
    ) -> CceTensor:
        if check_descriptor is not None:
            blayout, slayout, pad, fractal, compact = check_descriptor
            check_buffer_descriptor(
                name=name,
                space=space,
                element=element,
                blayout=blayout,
                slayout=slayout,
                fractal=fractal,
                pad=pad,
                compact=compact,
            )
        tensor = CceTensor(
            value_id=self._next_value_id,
            name=name,
            shape=shape,
            element=element,
            space=space,
            encoding=encoding_for(space, element),
            direction=direction,
        )
        self._next_value_id += 1
        self._values.append(tensor)
        self._tensors[name] = tensor
        return tensor

    # -- traversal --------------------------------------------------------

    def _visit(self, stmt: object) -> None:
        visitor = _StatementWalker(self)
        visitor.visit_stmt(stmt)

    def visit_assign(self, stmt: object) -> None:
        """Handle ``var = <expr>``."""
        value = stmt.value
        if isinstance(value, _ir_core.Call):
            if value.op.name == "buffer.alloc":
                self._bind_allocation(stmt, value)
                return
            raise CceMlirError(
                "unsupported-op",
                f"assignment of {value.op.name!r} is not supported; only "
                "buffer.alloc may define a buffer handle",
                op_name=value.op.name,
                span=getattr(value, "span", None),
            )
        # Scalar assignments (constants, block indices) carry no tensor state.
        if isinstance(getattr(stmt.var, "type", None), BufferType):
            raise CceMlirError(
                "unsupported-op",
                f"buffer handle {stmt.var.name_hint!r} is not defined by buffer.alloc",
                span=getattr(stmt, "span", None),
            )

    def visit_eval(self, stmt: object) -> None:
        """Handle ``<call>`` evaluated for its effects."""
        call = stmt.expr
        if not isinstance(call, _ir_core.Call):
            raise CceMlirError(
                "unsupported-statement",
                f"evaluated expression {type(call).__name__} is not a call",
                span=getattr(stmt, "span", None),
            )
        self._lower_call(call)

    def visit_return(self, stmt: object) -> None:
        """Record the returned values for contract validation."""
        values = tuple(stmt.value)
        names: list[str] = []
        for value in values:
            if not isinstance(value, Var):
                raise CceMlirError(
                    "invalid-return",
                    f"kernel returns {type(value).__name__}; only tensor parameters may be returned",
                    span=getattr(stmt, "span", None),
                )
            names.append(value.name_hint)
        self._return_names = tuple(names)

    # -- allocation -------------------------------------------------------

    def _bind_allocation(self, stmt: object, call: _ir_core.Call) -> None:
        buffer_type = stmt.var.type
        if not isinstance(buffer_type, BufferType):
            raise CceMlirError(
                "malformed-op",
                f"buffer.alloc result {stmt.var.name_hint!r} is not a BufferType",
                op_name="buffer.alloc",
                span=getattr(call, "span", None),
            )
        shape = _static_extents(buffer_type.shape, where=f"buffer {stmt.var.name_hint!r}")
        valid_shape = tuple(buffer_type.valid_shape) or shape
        if valid_shape != shape:
            raise CceMlirError(
                "unsupported-valid-shape",
                f"buffer {stmt.var.name_hint!r} has valid_shape {valid_shape} "
                f"differing from its physical shape {shape}; partial tiles are "
                "not supported by the cce-mlir route",
                op_name="buffer.alloc",
                span=getattr(call, "span", None),
            )
        space = memory_space(buffer_type.memory_space)
        element = CceElementType.from_data_type(buffer_type.dtype)
        if space is CceMemorySpace.GM:
            raise CceMlirError(
                "unsupported-alloc",
                f"buffer {stmt.var.name_hint!r} allocates DDR directly; on-chip buffers only",
                op_name="buffer.alloc",
                span=getattr(call, "span", None),
            )
        # The optional address argument is PyPTO's static offset inside the
        # tier. cce-mlir owns allocation and offset assignment later, so the
        # value is intentionally not propagated; the buffer identity is what
        # matters here.
        self._new_value(
            name=stmt.var.name_hint,
            shape=shape,
            element=element,
            space=space,
            check_descriptor=(
                buffer_type.blayout,
                buffer_type.slayout,
                buffer_type.pad,
                buffer_type.fractal,
                buffer_type.compact,
            ),
        )
        self._ops.append(
            CceOp(
                op_name="buffer.alloc",
                operands=(),
                result=self._tensors[stmt.var.name_hint],
                span=getattr(call, "span", None),
            )
        )

    # -- calls ------------------------------------------------------------

    def _lower_call(self, call: _ir_core.Call) -> None:
        op_name = call.op.name
        stage = self._registered_stage(op_name, call)
        if stage is not _ir_core.OpIRStage.Buffer:
            raise CceMlirError(
                "unsupported-op",
                f"operator {op_name!r} is registered at {stage.name}, not Buffer",
                op_name=op_name,
                span=getattr(call, "span", None),
            )
        if op_name not in _SUPPORTED_OPS:
            supported = ", ".join(sorted(_SUPPORTED_OPS))
            raise CceMlirError(
                "unsupported-op",
                f"operator {op_name!r} is not supported by the cce-mlir route (supported: {supported})",
                op_name=op_name,
                span=getattr(call, "span", None),
            )
        operands = self._convert_operands(call)
        effects = self._arg_effects(op_name, len(call.args), call)
        self._validate_operand_effects(op_name, operands, effects, call)
        self._validate_op(op_name, operands, effects, call)
        write_index = self._write_index(op_name, operands, effects, call)
        self._written.add(operands[write_index].name)  # type: ignore[union-attr]
        self._ops.append(
            CceOp(
                op_name=op_name,
                operands=operands,
                write_index=write_index,
                read_indices=tuple(
                    index
                    for index, effect in enumerate(effects)
                    if isinstance(operands[index], CceTensor)
                    and effect in (BufferAccess.Read, BufferAccess.ReadWrite)
                ),
                span=getattr(call, "span", None),
            )
        )

    def _registered_stage(self, op_name: str, call: _ir_core.Call) -> _ir_core.OpIRStage:
        try:
            return _ir_core.get_op_ir_stage(op_name)
        except Exception as exc:
            raise CceMlirError(
                "unregistered-op",
                f"operator {op_name!r} is not registered",
                op_name=op_name,
                span=getattr(call, "span", None),
            ) from exc

    def _convert_operands(self, call: _ir_core.Call) -> tuple[CceTensor | CceScalar | CceIntTuple, ...]:
        operands: list[CceTensor | CceScalar | CceIntTuple] = []
        for index, argument in enumerate(call.args):
            if isinstance(argument, Var) and argument.name_hint in self._tensors:
                operands.append(self._tensors[argument.name_hint])
            elif isinstance(argument, MakeTuple):
                operands.append(CceIntTuple(_int_tuple(argument, where=f"{call.op.name} argument {index}")))
            elif isinstance(argument, ConstInt):
                operands.append(
                    CceScalar(
                        value=argument.value,
                        element=CceElementType.from_data_type(argument.dtype),
                    )
                )
            elif isinstance(argument, ConstFloat):
                operands.append(
                    CceScalar(
                        value=argument.value,
                        element=CceElementType.from_data_type(argument.dtype),
                    )
                )
            else:
                raise CceMlirError(
                    "unsupported-operand",
                    f"{call.op.name} argument {index} is "
                    f"{type(argument).__name__}, which the route cannot represent",
                    op_name=call.op.name,
                    span=getattr(call, "span", None),
                )
        return tuple(operands)

    def _arg_effects(self, op_name: str, arity: int, call: _ir_core.Call) -> tuple[BufferAccess, ...]:
        try:
            return tuple(_ir_core.get_op_buffer_arg_effect(op_name, index).data for index in range(arity))
        except Exception as exc:
            raise CceMlirError(
                "malformed-op",
                f"operator {op_name!r} does not declare buffer effects for {arity} arguments",
                op_name=op_name,
                span=getattr(call, "span", None),
            ) from exc

    def _validate_operand_effects(
        self,
        op_name: str,
        operands: tuple[CceTensor | CceScalar | CceIntTuple, ...],
        effects: tuple[BufferAccess, ...],
        call: _ir_core.Call,
    ) -> None:
        for index, operand in enumerate(operands):
            effect = effects[index]
            if isinstance(operand, CceTensor):
                if effect is BufferAccess.None_:
                    raise CceMlirError(
                        "malformed-op",
                        f"{op_name} argument {index} is a tensor but the registry "
                        "declares no buffer access for it",
                        op_name=op_name,
                        span=getattr(call, "span", None),
                    )
            elif effect is not BufferAccess.None_:
                raise CceMlirError(
                    "malformed-op",
                    f"{op_name} argument {index} is {type(operand).__name__} but the "
                    f"registry declares {effect.name} buffer access for it",
                    op_name=op_name,
                    span=getattr(call, "span", None),
                )

    # -- semantic validation ---------------------------------------------

    def _write_index(
        self,
        op_name: str,
        operands: tuple[CceTensor | CceScalar | CceIntTuple, ...],
        effects: tuple[BufferAccess, ...],
        call: _ir_core.Call,
    ) -> int:
        writers = [
            index
            for index, (operand, effect) in enumerate(zip(operands, effects, strict=True))
            if isinstance(operand, CceTensor) and effect in (BufferAccess.Write, BufferAccess.ReadWrite)
        ]
        if len(writers) != 1:
            raise CceMlirError(
                "malformed-op",
                f"{op_name} must have exactly one written tensor operand, found {len(writers)}",
                op_name=op_name,
                span=getattr(call, "span", None),
            )
        return writers[0]

    def _write_tensor(
        self,
        op_name: str,
        operands: tuple[CceTensor | CceScalar | CceIntTuple, ...],
        effects: tuple[BufferAccess, ...],
        call: _ir_core.Call,
    ) -> CceTensor:
        operand = operands[self._write_index(op_name, operands, effects, call)]
        assert isinstance(operand, CceTensor)
        return operand

    def _read_tensors(
        self,
        op_name: str,
        operands: tuple[CceTensor | CceScalar | CceIntTuple, ...],
        effects: tuple[BufferAccess, ...],
        call: _ir_core.Call,
    ) -> tuple[CceTensor, ...]:
        return tuple(
            operand
            for operand, effect in zip(operands, effects, strict=True)
            if isinstance(operand, CceTensor) and effect in (BufferAccess.Read, BufferAccess.ReadWrite)
        )

    def _validate_op(
        self,
        op_name: str,
        operands: tuple[CceTensor | CceScalar | CceIntTuple, ...],
        effects: tuple[BufferAccess, ...],
        call: _ir_core.Call,
    ) -> None:
        span = getattr(call, "span", None)
        if op_name in ("buffer.load", "buffer.store"):
            self._validate_transfer(op_name, operands, effects, call)
            return
        if op_name == "buffer.copy":
            self._validate_copy(op_name, operands, effects, call)
            return
        if op_name in ("buffer.matmul", "buffer.matmul_acc"):
            self._validate_matmul(op_name, operands, effects, call)
            return
        # Elementwise family: reads in UB, exactly one UB write, one shape.
        write = self._write_tensor(op_name, operands, effects, call)
        reads = self._read_tensors(op_name, operands, effects, call)
        if write.space is not CceMemorySpace.UB or any(
            tensor.space is not CceMemorySpace.UB for tensor in reads
        ):
            raise CceMlirError(
                "unsupported-memory-space",
                f"{op_name} is emitted on the vector engine and requires UB "
                f"operands; got write={write.space.value} "
                f"reads={[tensor.space.value for tensor in reads]}",
                op_name=op_name,
                span=span,
            )
        self._require_same_shape(op_name, (write, *reads), call)
        if any(tensor.element != write.element for tensor in reads):
            raise CceMlirError(
                "dtype-mismatch",
                f"{op_name} operands must share one element type; got "
                f"{write.element.mlir} and "
                f"{[tensor.element.mlir for tensor in reads]}",
                op_name=op_name,
                span=span,
            )

    def _validate_transfer(
        self,
        op_name: str,
        operands: tuple[CceTensor | CceScalar | CceIntTuple, ...],
        effects: tuple[BufferAccess, ...],
        call: _ir_core.Call,
    ) -> None:
        span = getattr(call, "span", None)
        tensors = [operand for operand in operands if isinstance(operand, CceTensor)]
        tuples = [operand for operand in operands if isinstance(operand, CceIntTuple)]
        if len(tensors) != 2 or len(tuples) != 2:
            raise CceMlirError(
                "malformed-op",
                f"{op_name} expects two tensor and two integer-tuple operands, got "
                f"{len(tensors)} tensor(s) and {len(tuples)} tuple(s)",
                op_name=op_name,
                span=span,
            )
        write = self._write_tensor(op_name, operands, effects, call)
        read = next(tensor for tensor in tensors if tensor is not write)
        offsets = min(tuples, key=lambda item: sum(value != 0 for value in item.values))
        extents = next(item for item in tuples if item is not offsets)
        if any(value != 0 for value in offsets.values):
            raise CceMlirError(
                "unsupported-window",
                f"{op_name} uses non-zero offsets {offsets.values}; windowed "
                "transfers are not supported in this milestone",
                op_name=op_name,
                span=span,
            )
        if len(extents.values) != len(write.shape) or extents.values != write.shape:
            raise CceMlirError(
                "unsupported-window",
                f"{op_name} moves extents {extents.values} but the destination "
                f"buffer is {write.shape}; partial transfers are not supported",
                op_name=op_name,
                span=span,
            )
        if read.shape != write.shape:
            raise CceMlirError(
                "shape-mismatch",
                f"{op_name} transfers {read.shape} into {write.shape}; shapes must match",
                op_name=op_name,
                span=span,
            )
        if read.element != write.element:
            raise CceMlirError(
                "dtype-mismatch",
                f"{op_name} transfers {read.element.mlir} into {write.element.mlir}; dtypes must match",
                op_name=op_name,
                span=span,
            )
        if op_name == "buffer.load":
            if read.space is not CceMemorySpace.GM or write.space is CceMemorySpace.GM:
                raise CceMlirError(
                    "unsupported-memory-space",
                    f"buffer.load must move GM to an on-chip tier; got "
                    f"{read.space.value} -> {write.space.value}",
                    op_name=op_name,
                    span=span,
                )
        else:
            if write.space is not CceMemorySpace.GM or read.space is CceMemorySpace.GM:
                raise CceMlirError(
                    "unsupported-memory-space",
                    f"buffer.store must move an on-chip tier to GM; got "
                    f"{read.space.value} -> {write.space.value}",
                    op_name=op_name,
                    span=span,
                )
            if write.name in self._stored_parameters:
                self._stored_parameters[write.name] += 1
            else:
                self._stored_parameters[write.name] = 1

    def _validate_copy(
        self,
        op_name: str,
        operands: tuple[CceTensor | CceScalar | CceIntTuple, ...],
        effects: tuple[BufferAccess, ...],
        call: _ir_core.Call,
    ) -> None:
        span = getattr(call, "span", None)
        write = self._write_tensor(op_name, operands, effects, call)
        read = next(
            tensor for tensor in self._read_tensors(op_name, operands, effects, call) if tensor is not write
        )
        allowed = {
            (CceMemorySpace.L1, CceMemorySpace.L0A),
            (CceMemorySpace.L1, CceMemorySpace.L0B),
            (CceMemorySpace.UB, CceMemorySpace.UB),
        }
        if (read.space, write.space) not in allowed:
            rendered = ", ".join(f"{source.value}->{target.value}" for source, target in sorted(allowed))
            raise CceMlirError(
                "unsupported-memory-space",
                f"buffer.copy from {read.space.value} to {write.space.value} is not "
                f"supported (allowed: {rendered})",
                op_name=op_name,
                span=span,
            )
        if read.shape != write.shape or read.element != write.element:
            raise CceMlirError(
                "shape-mismatch",
                f"buffer.copy operands differ: {read.shape}/{read.element.mlir} vs "
                f"{write.shape}/{write.element.mlir}",
                op_name=op_name,
                span=span,
            )

    def _validate_matmul(
        self,
        op_name: str,
        operands: tuple[CceTensor | CceScalar | CceIntTuple, ...],
        effects: tuple[BufferAccess, ...],
        call: _ir_core.Call,
    ) -> None:
        span = getattr(call, "span", None)
        write = self._write_tensor(op_name, operands, effects, call)
        reads = [tensor for tensor in self._read_tensors(op_name, operands, effects, call)]
        if op_name == "buffer.matmul_acc":
            reads = [tensor for tensor in reads if tensor is not write]
        if len(reads) != 2:
            raise CceMlirError(
                "malformed-op",
                f"{op_name} expects two read operands, got {len(reads)}",
                op_name=op_name,
                span=span,
            )
        left, right = reads
        if left.space is not CceMemorySpace.L0A or right.space is not CceMemorySpace.L0B:
            raise CceMlirError(
                "unsupported-memory-space",
                f"{op_name} requires Left (l0a) and Right (l0b) operands, got "
                f"{left.space.value} and {right.space.value}",
                op_name=op_name,
                span=span,
            )
        if write.space is not CceMemorySpace.L0C:
            raise CceMlirError(
                "unsupported-memory-space",
                f"{op_name} requires an Acc (l0c) destination, got {write.space.value}",
                op_name=op_name,
                span=span,
            )
        if len(left.shape) != 2 or len(right.shape) != 2 or len(write.shape) != 2:
            raise CceMlirError(
                "unsupported-shape",
                f"{op_name} requires rank-2 operands, got {left.shape}, {right.shape}, {write.shape}",
                op_name=op_name,
                span=span,
            )
        m, k = left.shape
        k2, n = right.shape
        if k != k2 or write.shape != (m, n):
            raise CceMlirError(
                "shape-mismatch",
                f"{op_name} operand shapes {left.shape} x {right.shape} do not produce {write.shape}",
                op_name=op_name,
                span=span,
            )
        if left.element != right.element:
            raise CceMlirError(
                "dtype-mismatch",
                f"{op_name} matrix operand dtypes differ: {left.element.mlir} vs {right.element.mlir}",
                op_name=op_name,
                span=span,
            )
        if op_name == "buffer.matmul_acc" and write.name not in self._written:
            raise CceMlirError(
                "uninitialized-accumulator",
                f"buffer.matmul_acc accumulates into {write.name!r}, which has no "
                "preceding write; use buffer.matmul for the first product",
                op_name=op_name,
                span=span,
            )

    def _require_same_shape(self, op_name: str, tensors: tuple[CceTensor, ...], call: _ir_core.Call) -> None:
        shapes = {tensor.shape for tensor in tensors}
        if len(shapes) != 1:
            raise CceMlirError(
                "shape-mismatch",
                f"{op_name} operands have different shapes: {[tensor.shape for tensor in tensors]}",
                op_name=op_name,
                span=getattr(call, "span", None),
            )

    # -- post conditions --------------------------------------------------

    def _param_directions(self) -> tuple[tuple[str, ParamDirection], ...]:
        return tuple(
            (parameter.name_hint, direction)
            for parameter, direction in zip(
                self._function.params, self._function.param_directions, strict=True
            )
        )

    def _validate_returns(self) -> None:
        out_names = {name for name, direction in self._param_directions() if direction is ParamDirection.Out}
        in_names = {
            name for name, direction in self._param_directions() if direction is not ParamDirection.Out
        }
        for name in out_names:
            if name not in self._stored_parameters:
                raise CceMlirError(
                    "missing-output-store",
                    f"output parameter {name!r} is never written by a buffer.store",
                    span=getattr(self._function, "span", None),
                )
        for name in self._stored_parameters:
            if name in in_names:
                raise CceMlirError(
                    "write-to-input",
                    f"input parameter {name!r} is written by buffer.store",
                    span=getattr(self._function, "span", None),
                )
        if self._return_names and set(self._return_names) != out_names:
            raise CceMlirError(
                "invalid-return",
                f"kernel returns {sorted(self._return_names)} but its Out parameters are {sorted(out_names)}",
                span=getattr(self._function, "span", None),
            )

    def _derive_kind(self) -> CceKernelKind:
        spaces = {value.space for value in self._values}
        uses_cube = any(space in _CUBE_SPACES for space in spaces)
        declared = {
            FunctionType.AIC: CceKernelKind.CUBE,
            FunctionType.AIV: CceKernelKind.VECTOR,
        }.get(self._function.func_type)
        if declared is None:
            return CceKernelKind.CUBE if uses_cube else CceKernelKind.VECTOR
        if declared is CceKernelKind.VECTOR and uses_cube:
            raise CceMlirError(
                "mixed-kernel",
                f"function {self._function.name!r} is declared {self._function.func_type.name} "
                "but allocates matrix buffers; mixed cube/vector kernels are not supported",
                span=getattr(self._function, "span", None),
            )
        return declared


class _StatementWalker(_pypto_ir.IRVisitor):
    """Dispatch statements to the lowering, refusing control flow."""

    def __init__(self, lowering: _KernelLowering) -> None:
        super().__init__()
        self._lowering = lowering

    def visit_stmt(self, stmt: object) -> None:
        if isinstance(stmt, (SeqStmts, AssignStmt, EvalStmt, ReturnStmt)):
            super().visit_stmt(stmt)
            return
        self._refuse(stmt)

    def visit_seq_stmts(self, op: object) -> None:
        for stmt in op.stmts:
            self.visit_stmt(stmt)

    def visit_assign_stmt(self, op: object) -> None:
        self._lowering.visit_assign(op)

    def visit_eval_stmt(self, op: object) -> None:
        self._lowering.visit_eval(op)

    def visit_return_stmt(self, op: object) -> None:
        self._lowering.visit_return(op)

    def _refuse(self, op: object) -> None:
        raise CceMlirError(
            "unsupported-control-flow",
            f"{type(op).__name__} is not supported by the cce-mlir route in this "
            "milestone; kernels must be straight-line blocks",
            span=getattr(op, "span", None),
        )

    visit_for_stmt = _refuse
    visit_while_stmt = _refuse
    visit_if_stmt = _refuse
    visit_break_stmt = _refuse
    visit_continue_stmt = _refuse
    visit_yield_stmt = _refuse
    visit_in_core_scope_stmt = _refuse
    visit_cluster_scope_stmt = _refuse
    visit_graph_scope_stmt = _refuse
    visit_hierarchy_scope_stmt = _refuse
    visit_spmd_scope_stmt = _refuse
    visit_split_aiv_scope_stmt = _refuse
    visit_runtime_scope_stmt = _refuse
    visit_comm_domain_scope_stmt = _refuse
    visit_inline_stmt = _refuse


def _symbol_name(name: str) -> str:
    """Return an MLIR-symbol-safe spelling of a PyPTO function name."""
    sanitized = "".join(character if character.isalnum() or character in "_$." else "_" for character in name)
    if not sanitized or sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized
