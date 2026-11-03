# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Artifact generation for the cce-mlir kernel-compiler route.

Produces the same runtime layout as the PTOAS route — adapter source, device
object, orchestration and ``kernel_config.py`` — so ``CompiledProgram`` and the
device runner can dispatch the kernel without route-specific code. The device
object is linked under ``kernel_entry`` by ``KernelCompiler.compile_incore``.
"""

from __future__ import annotations

import json
from typing import Any

from pypto.pypto_core import codegen as _codegen_core
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core import passes as _passes

from ..kernel_config import generate_kernel_config
from . import CceMlirError, CceTarget, lower_program
from .adapter import generate_adapter_source
from .elf import symbol_names
from .emit import emit_module
from .manifest import build_manifest
from .pipeline import cann_toolkit, extract_device_object, lower_to_llvm

__all__ = ["CceMlirBackend", "generate"]

_NAME = "cce_mlir"


class CceMlirBackend:
    """Generate a PyPTO runtime artifact using cce-mlir for device code."""

    name = _NAME

    def generate(
        self,
        program: _ir_core.Program,
        output_dir: str,
        skip_ptoas: bool = False,
        *,
        memory_planner: _passes.MemoryPlanner | None = None,
        emit_source_loc: bool | None = None,
        dump_ptoas_passes: bool = False,
        runtime: _passes.RuntimeKind | None = None,
        target: CceTarget | None = None,
    ) -> dict[str, str | bytes]:
        """Lower every InCore kernel and assemble the runtime artifact.

        Args:
            program: Post-pass program with one Orchestration entry and its
                Buffer-stage InCore kernels.
            output_dir: Artifact root, used only for diagnostics.
            skip_ptoas: Ignored; the route never invokes PTOAS.
            memory_planner: Accepted for signature compatibility. cce-mlir owns
                device allocation, so the planner only affects ignored offsets.
            emit_source_loc: Unused; MLIR locations come from the typed IR.
            dump_ptoas_passes: Ignored; there is no PTOAS pass dump.
            runtime: Simpler runtime ABI recorded in ``kernel_config.py``.
            target: cce-mlir target; resolved from the backend handler when None.

        Returns:
            Relative path -> text or bytes.

        Raises:
            CceMlirError: On an unsupported program, missing toolchain, or a
                kernel whose generated source does not match the orchestration.
        """
        # ``skip_ptoas`` / ``dump_ptoas_passes`` are PTOAS-only options. The JIT
        # path sets skip_ptoas from the environment, so this route ignores both
        # instead of rejecting a call the user never made explicitly.
        effective_runtime = runtime or _passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER
        kernels = lower_program(program, target=target)
        toolkit = cann_toolkit()

        orchestration = _orchestration_function(program)
        orchestration_result = _codegen_core.generate_orchestration(program, orchestration)
        core_types = dict(orchestration_result.func_name_to_core_type)

        files: dict[str, str | bytes] = {}
        link_objects: dict[str, list[str]] = {}
        for kernel in kernels:
            core_type = core_types.get(kernel.function_name)
            if core_type is None:
                raise CceMlirError(
                    "orchestration-mismatch",
                    f"orchestration {orchestration.name!r} does not reference kernel "
                    f"{kernel.function_name!r}",
                )
            core_dir = "aiv" if core_type == _ir_core.CoreType.VECTOR else "aic"
            module = emit_module(kernel)
            emitted_text = str(module)
            lowered_text = lower_to_llvm(module, toolkit=toolkit)
            device_object = extract_device_object(module)
            try:
                symbols = symbol_names(device_object)
            except Exception as exc:  # ElfError is the expected case
                raise CceMlirError(
                    "invalid-device-object",
                    f"kernel {kernel.function_name!r} produced an unreadable device object: {exc}",
                ) from exc
            if kernel.device_symbol not in symbols:
                raise CceMlirError(
                    "missing-device-symbol",
                    f"device object for {kernel.function_name!r} does not define {kernel.device_symbol!r}",
                )

            source_path = f"kernels/{core_dir}/{kernel.function_name}.cpp"
            object_path = f"kernels/{core_dir}/{kernel.function_name}.cce.o"
            files[source_path] = generate_adapter_source(kernel)
            files[object_path] = device_object
            files[f"cce_mlir/{kernel.function_name}.mlir"] = emitted_text
            files[f"cce_mlir/{kernel.function_name}.lowered.mlir"] = lowered_text
            manifest = build_manifest(
                kernel,
                status="lowered",
                object_bytes=device_object,
                object_symbols=symbols,
            )
            files[f"cce_mlir/{kernel.function_name}.manifest.json"] = json.dumps(manifest, indent=2) + "\n"
            link_objects[kernel.function_name] = [object_path]

        files[f"orchestration/{orchestration.name}.cpp"] = (
            f"// Orchestration Function: {orchestration.name}\n"
            f"// Generated by PyPTO IR Compiler\n\n"
            f"{orchestration_result.code}"
        )
        files["kernel_config.py"] = generate_kernel_config(
            orchestration.name,
            orchestration_result.func_name_to_id,
            orchestration_result.func_name_to_core_type,
            orchestration_result.func_name_to_signature,
            orchestration_result.orchestration_signature,
            func_name_to_link_objects=link_objects,
            runtime=effective_runtime,
        )
        return files


def _orchestration_function(program: _ir_core.Program) -> _ir_core.Function:
    """Return the program's single Orchestration entry."""
    orchestrations = [
        function
        for function in program.functions.values()
        if function.func_type == _ir_core.FunctionType.Orchestration
    ]
    if len(orchestrations) != 1:
        raise CceMlirError(
            "unsupported-program",
            f"the cce_mlir route requires exactly one Orchestration function, found {len(orchestrations)}",
        )
    return orchestrations[0]


def generate(
    program: _ir_core.Program,
    output_dir: str,
    skip_ptoas: bool = False,
    **options: Any,
) -> dict[str, str | bytes]:
    """Module-level entry with the same shape as ``pto_backend.generate``."""
    return CceMlirBackend().generate(program, output_dir, skip_ptoas, **options)
