"""PyPTO IR to the small structured cce-mlir contract."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class CceMlirError(ValueError):
    """The experimental bridge cannot represent the requested kernel."""


class CceTarget(str, Enum):
    A3 = "a3"
    A5 = "a5"


@dataclass(frozen=True)
class Argument:
    name: str
    shape: tuple[int, ...]
    dtype: str
    memory_space: str = "gm"
    direction: str = "input"


@dataclass(frozen=True)
class Operation:
    name: str
    inputs: tuple[str, ...]
    output: str


@dataclass(frozen=True)
class KernelSpec:
    name: str
    target: CceTarget
    kind: str
    arguments: tuple[Argument, ...]
    operations: tuple[Operation, ...]


_EXPECTED = (
    Operation("load", ("a",), "mat"),
    Operation("load", ("b",), "mat"),
    Operation("move", ("a",), "left"),
    Operation("move", ("b",), "right"),
    Operation("matmul", ("left", "right"), "acc"),
    Operation("store", ("acc",), "c"),
)


def _load_kernel(path: Path):
    spec = importlib.util.spec_from_file_location("pypto_cce_mlir_fixture", path.resolve())
    if spec is None or spec.loader is None:
        raise CceMlirError(f"cannot load fixture: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernels = [value for value in vars(module).values() if value.__class__.__name__ == "JITFunction"]
    if len(kernels) != 1:
        raise CceMlirError(f"expected one PyPTO JITFunction, found {len(kernels)}")
    return kernels[0]


def _parameter_name(value: object, parameters: dict[str, object]) -> str | None:
    hint = getattr(value, "name_hint", "")
    return next((name for name in parameters if hint == name or hint.startswith(f"{name}__ssa")), None)


def _memory_space(value: object) -> str:
    return str(value).rsplit(".", 1)[-1].lower()


def _shape(value: object) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    dimensions = tuple(getattr(dimension, "value", None) for dimension in (shape or ()))
    if len(dimensions) != 2 or any(not isinstance(dimension, int) or dimension <= 0 for dimension in dimensions):
        raise CceMlirError("the experimental cube bridge requires static rank-2 shapes")
    if any(dimension % 16 for dimension in dimensions):
        raise CceMlirError(f"cube dimensions must be multiples of 16: {dimensions}")
    return dimensions


def _dtype(value: object) -> str:
    name = str(value).rsplit(".", 1)[-1].lower()
    return {"fp16": "f16", "fp32": "f32"}.get(name, name)


def lower_pypto_matmul(path: str | Path, *, target: CceTarget) -> KernelSpec:
    from pypto import ir as pypto_ir

    kernel = _load_kernel(Path(path))
    try:
        program = kernel.lower()
    except Exception as error:  # pragma: no cover - environment diagnostic
        raise CceMlirError(f"PyPTO lowering failed: {error}") from error
    functions = [
        function
        for function in program.functions.values()
        if function.ir_stage == pypto_ir.FunctionIRStage.Buffer and pypto_ir.is_incore_type(function.func_type)
    ]
    if len(functions) != 1:
        raise CceMlirError(f"expected one lowered Buffer InCore function, found {len(functions)}")
    function = functions[0]
    parameters = {}
    arguments = []
    for parameter in function.params:
        name = _parameter_name(parameter, {"a": None, "b": None, "c": None})
        if name is None:
            raise CceMlirError(f"unexpected PyPTO parameter: {parameter.name_hint}")
        parameters[name] = parameter
        arguments.append(Argument(name, _shape(parameter.type), _dtype(getattr(parameter.type, "dtype", None)), direction="output" if name == "c" else "input"))
    expected = ((64, 128, "f16"), (128, 64, "f16"), (64, 64, "f32"))
    if tuple((argument.shape[0], argument.shape[1], argument.dtype) for argument in arguments) != expected:
        raise CceMlirError(f"unexpected PyPTO operand contract: {arguments}")

    buffers: dict[str, str] = {}
    operations: list[Operation] = []

    class Collector(pypto_ir.IRVisitor):
        def visit_assign_stmt(self, statement):  # type: ignore[override]
            value = statement.value
            if getattr(getattr(value, "op", None), "name", None) == "buffer.alloc":
                buffers[statement.var.name_hint] = _memory_space(statement.var.type.memory_space)
            super().visit_assign_stmt(statement)

        def visit_eval_stmt(self, statement):  # type: ignore[override]
            expression = statement.expr
            operation = getattr(getattr(expression, "op", None), "name", None)
            args = list(getattr(expression, "args", ()))
            if operation == "buffer.load":
                source = _parameter_name(args[0], parameters)
                destination = getattr(args[3], "name_hint", "")
                if source not in ("a", "b") or buffers.get(destination) != "mat":
                    raise CceMlirError("buffer.load must map A/B to Mat")
                buffers[destination] = source
                operations.append(Operation("load", (source,), "mat"))
            elif operation == "buffer.copy":
                source = buffers.get(getattr(args[0], "name_hint", ""))
                destination = getattr(args[1], "name_hint", "")
                memory = buffers.get(destination)
                if source not in ("a", "b") or memory not in ("left", "right"):
                    raise CceMlirError("buffer.copy must map Mat to Left/Right")
                buffers[destination] = memory
                operations.append(Operation("move", (source,), memory))
            elif operation == "buffer.matmul":
                lhs = buffers.get(getattr(args[0], "name_hint", ""))
                rhs = buffers.get(getattr(args[1], "name_hint", ""))
                destination = getattr(args[2], "name_hint", "")
                if (lhs, rhs) != ("left", "right") or buffers.get(destination) != "acc":
                    raise CceMlirError("buffer.matmul must consume Left/Right and produce Acc")
                buffers[destination] = "acc"
                operations.append(Operation("matmul", ("left", "right"), "acc"))
            elif operation == "buffer.store":
                source = buffers.get(getattr(args[0], "name_hint", ""))
                destination = _parameter_name(args[3], parameters)
                if source != "acc" or destination != "c":
                    raise CceMlirError("buffer.store must map Acc to C")
                operations.append(Operation("store", ("acc",), "c"))
            elif operation != "buffer.alloc":
                raise CceMlirError(f"unsupported PyPTO operation: {operation}")
            super().visit_eval_stmt(statement)

    Collector().visit_stmt(function.body)
    if tuple(operations) != _EXPECTED:
        raise CceMlirError(f"unexpected lowered operation sequence: {operations}")
    return KernelSpec(kernel.__name__, target, "cube", tuple(arguments), tuple(operations))


def emit_cce_mlir(spec: KernelSpec) -> str:
    if spec.kind != "cube" or tuple(spec.operations) != _EXPECTED or spec.target is not CceTarget.A3:
        raise CceMlirError("the first emitter supports only the validated A3 cube contract")
    return _CUBE_MODULE


_CUBE_MODULE = r'''module {
  npu.module @pypto_matmul_mod attributes {target = #npu.kernel_target<a3>} {
    npu.func private @matmul_64x128x64 (%arg0: tensor<64x128xf16, #npu.memory_space<gm>>, %arg1: tensor<128x64xf16, #npu.memory_space<gm>>, %arg2: tensor<64x64xf32, #npu.memory_space<gm>>) kernel attributes {kind = #npu.kernel_kind<cube>} -> tensor<64x64xf32, #npu.memory_space<gm>> {
      %a_gm = layout.pack_layout %arg0 : tensor<64x128xf16, #npu.memory_space<gm>> -> tensor<64x128xf16, {npu.layout = #npu.layout<inner_dims_pos = [0, 1], inner_tile_sizes = [16, 16], outer_dims_perm = [1, 0]>, npu.memory_space = #npu.memory_space<gm>}>
      %a_l1_dst = tensor.empty() : tensor<64x128xf16, {npu.layout = #npu.layout<inner_dims_pos = [0, 1], inner_tile_sizes = [16, 16], outer_dims_perm = [1, 0]>, npu.memory_space = #npu.memory_space<l1>}>
      %a_l1 = pto.load %a_gm, %a_l1_dst : tensor<64x128xf16, {npu.layout = #npu.layout<inner_dims_pos = [0, 1], inner_tile_sizes = [16, 16], outer_dims_perm = [1, 0]>, npu.memory_space = #npu.memory_space<gm>}>, tensor<64x128xf16, {npu.layout = #npu.layout<inner_dims_pos = [0, 1], inner_tile_sizes = [16, 16], outer_dims_perm = [1, 0]>, npu.memory_space = #npu.memory_space<l1>}> -> tensor<64x128xf16, {npu.layout = #npu.layout<inner_dims_pos = [0, 1], inner_tile_sizes = [16, 16], outer_dims_perm = [1, 0]>, npu.memory_space = #npu.memory_space<l1>}>
      %a_l1_repacked = layout.repack_layout %a_l1 : tensor<64x128xf16, {npu.layout = #npu.layout<inner_dims_pos = [0, 1], inner_tile_sizes = [16, 16], outer_dims_perm = [1, 0]>, npu.memory_space = #npu.memory_space<l1>}> -> tensor<64x128xf16, {npu.layout = #npu.layout<inner_dims_pos = [0, 1], inner_tile_sizes = [16, 16], outer_dims_perm = []>, npu.memory_space = #npu.memory_space<l1>}>
      %a_l0a_dst = tensor.empty() : tensor<64x128xf16, {npu.layout = #npu.layout<inner_dims_pos = [0, 1], inner_tile_sizes = [16, 16], outer_dims_perm = []>, npu.memory_space = #npu.memory_space<l0a>}>
      %a_l0a = pto.mov %a_l1_repacked, %a_l0a_dst : tensor<64x128xf16, {npu.layout = #npu.layout<inner_dims_pos = [0, 1], inner_tile_sizes = [16, 16], outer_dims_perm = []>, npu.memory_space = #npu.memory_space<l1>}>, tensor<64x128xf16, {npu.layout = #npu.layout<inner_dims_pos = [0, 1], inner_tile_sizes = [16, 16], outer_dims_perm = []>, npu.memory_space = #npu.memory_space<l0a>}> -> tensor<64x128xf16, {npu.layout = #npu.layout<inner_dims_pos = [0, 1], inner_tile_sizes = [16, 16], outer_dims_perm = []>, npu.memory_space = #npu.memory_space<l0a>}>
      %b_gm = layout.pack_layout %arg1 : tensor<128x64xf16, #npu.memory_space<gm>> -> tensor<128x64xf16, {npu.layout = #npu.layout<inner_dims_pos = [0, 1], inner_tile_sizes = [16, 16], outer_dims_perm = [1, 0]>, npu.memory_space = #npu.memory_space<gm>}>
      %b_l1_dst = tensor.empty() : tensor<128x64xf16, {npu.layout = #npu.layout<inner_dims_pos = [0, 1], inner_tile_sizes = [16, 16], outer_dims_perm = [1, 0]>, npu.memory_space = #npu.memory_space<l1>}>
      %b_l1 = pto.load %b_gm, %b_l1_dst : tensor<128x64xf16, {npu.layout = #npu.layout<inner_dims_pos = [0, 1], inner_tile_sizes = [16, 16], outer_dims_perm = [1, 0]>, npu.memory_space = #npu.memory_space<gm>}>, tensor<128x64xf16, {npu.layout = #npu.layout<inner_dims_pos = [0, 1], inner_tile_sizes = [16, 16], outer_dims_perm = [1, 0]>, npu.memory_space = #npu.memory_space<l1>}> -> tensor<128x64xf16, {npu.layout = #npu.layout<inner_dims_pos = [0, 1], inner_tile_sizes = [16, 16], outer_dims_perm = [1, 0]>, npu.memory_space = #npu.memory_space<l1>}>
      %b_l1_repacked = layout.repack_layout %b_l1 : tensor<128x64xf16, {npu.layout = #npu.layout<inner_dims_pos = [0, 1], inner_tile_sizes = [16, 16], outer_dims_perm = [1, 0]>, npu.memory_space = #npu.memory_space<l1>}> -> tensor<128x64xf16, {npu.layout = #npu.layout<inner_dims_pos = [1, 0], inner_tile_sizes = [16, 16], outer_dims_perm = []>, npu.memory_space = #npu.memory_space<l1>}>
      %b_l0b_dst = tensor.empty() : tensor<128x64xf16, {npu.layout = #npu.layout<inner_dims_pos = [1, 0], inner_tile_sizes = [16, 16], outer_dims_perm = []>, npu.memory_space = #npu.memory_space<l0b>}>
      %b_l0b = pto.mov %b_l1_repacked, %b_l0b_dst : tensor<128x64xf16, {npu.layout = #npu.layout<inner_dims_pos = [1, 0], inner_tile_sizes = [16, 16], outer_dims_perm = []>, npu.memory_space = #npu.memory_space<l1>}>, tensor<128x64xf16, {npu.layout = #npu.layout<inner_dims_pos = [1, 0], inner_tile_sizes = [16, 16], outer_dims_perm = []>, npu.memory_space = #npu.memory_space<l0b>}> -> tensor<128x64xf16, {npu.layout = #npu.layout<inner_dims_pos = [1, 0], inner_tile_sizes = [16, 16], outer_dims_perm = []>, npu.memory_space = #npu.memory_space<l0b>}>
      %c_l0c_dst = tensor.empty() : tensor<64x64xf32, {npu.layout = #npu.layout<inner_dims_pos = [0, 1], inner_tile_sizes = [16, 16], outer_dims_perm = [1, 0]>, npu.memory_space = #npu.memory_space<l0c>}>
      %c_l0c = pto.matmul %a_l0a, %b_l0b, %c_l0c_dst : tensor<64x128xf16, {npu.layout = #npu.layout<inner_dims_pos = [0, 1], inner_tile_sizes = [16, 16], outer_dims_perm = []>, npu.memory_space = #npu.memory_space<l0a>}>, tensor<128x64xf16, {npu.layout = #npu.layout<inner_dims_pos = [1, 0], inner_tile_sizes = [16, 16], outer_dims_perm = []>, npu.memory_space = #npu.memory_space<l0b>}>, tensor<64x64xf32, {npu.layout = #npu.layout<inner_dims_pos = [0, 1], inner_tile_sizes = [16, 16], outer_dims_perm = [1, 0]>, npu.memory_space = #npu.memory_space<l0c>}> -> tensor<64x64xf32, {npu.layout = #npu.layout<inner_dims_pos = [0, 1], inner_tile_sizes = [16, 16], outer_dims_perm = [1, 0]>, npu.memory_space = #npu.memory_space<l0c>}>
      %c_nat = layout.unpack_layout %c_l0c : tensor<64x64xf32, {npu.layout = #npu.layout<inner_dims_pos = [0, 1], inner_tile_sizes = [16, 16], outer_dims_perm = [1, 0]>, npu.memory_space = #npu.memory_space<l0c>}> -> tensor<64x64xf32, #npu.memory_space<l0c>>
      %c_gm = pto.store %c_nat, %arg2 : tensor<64x64xf32, #npu.memory_space<l0c>>, tensor<64x64xf32, #npu.memory_space<gm>> -> tensor<64x64xf32, #npu.memory_space<gm>>
      npu.return %c_gm : tensor<64x64xf32, #npu.memory_space<gm>>
    }
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func @run(%a: memref<64x128xf16>, %b: memref<128x64xf16>, %c: memref<64x64xf32>) attributes {llvm.emit_c_interface} {
    npu.init
    %zero = arith.constant 0 : i32
    %device = npu.set_device %zero : !npu.device
    %stream = npu.create_stream : !npu.stream
    %bytes = arith.constant 16384 : i64
    %a_gm = npu.alloc %bytes : memref<64x128xf16, #npu.memory_space<gm>>
    %b_gm = npu.alloc %bytes : memref<128x64xf16, #npu.memory_space<gm>>
    %c_gm = npu.alloc %bytes : memref<64x64xf32, #npu.memory_space<gm>>
    npu.memcpy %a_gm, %a, %stream : memref<64x128xf16, #npu.memory_space<gm>>, memref<64x128xf16>, !npu.stream
    npu.memcpy %b_gm, %b, %stream : memref<128x64xf16, #npu.memory_space<gm>>, memref<128x64xf16>, !npu.stream
    %blocks = arith.constant 1 : i32
    npu.launch_func @pypto_matmul_mod::@matmul_64x128x64 blocks(%blocks : i32) on(%stream : !npu.stream) args(%a_gm, %b_gm, %c_gm : memref<64x128xf16, #npu.memory_space<gm>>, memref<128x64xf16, #npu.memory_space<gm>>, memref<64x64xf32, #npu.memory_space<gm>>)
    npu.sync_stream %stream : !npu.stream
    npu.memcpy %c, %c_gm, %stream : memref<64x64xf32>, memref<64x64xf32, #npu.memory_space<gm>>, !npu.stream
    npu.sync_stream %stream : !npu.stream
    npu.dealloc %a_gm : memref<64x128xf16, #npu.memory_space<gm>>
    npu.dealloc %b_gm : memref<128x64xf16, #npu.memory_space<gm>>
    npu.dealloc %c_gm : memref<64x64xf32, #npu.memory_space<gm>>
    npu.destroy_stream %stream : !npu.stream
    npu.reset_device %device : !npu.device
    npu.finalize
    return
  }
  func.func @main() -> i32 {
    %a = memref.alloc() : memref<64x128xf16>
    %b = memref.alloc() : memref<128x64xf16>
    %c = memref.alloc() : memref<64x64xf32>
    call @run(%a, %b, %c) : (memref<64x128xf16>, memref<128x64xf16>, memref<64x64xf32>) -> ()
    %result = arith.constant 0 : i32
    return %result : i32
  }
}
'''
