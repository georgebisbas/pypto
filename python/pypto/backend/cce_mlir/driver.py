# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Standalone compile driver for the cce-mlir route.

This is what ``python -m examples.cce_mlir compile`` calls. It lowers one or
more Buffer-stage kernels, emits cce-mlir MLIR through the dialect bindings,
runs cce-mlir's pipeline in process, and writes the device object, manifest and
report. Missing toolchains are reported as explicit blocked statuses rather than
raised, so the command stays usable in containers without CANN.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .elf import ElfError, symbol_names
from .errors import CceMlirError
from .kernel_ir import CceKernel
from .manifest import build_manifest, write_manifest
from .pipeline import cann_toolkit, extract_device_object, lower_to_llvm

__all__ = ["CompileResult", "CompileStatus", "KernelArtifact", "compile_program"]


class CompileStatus(str, Enum):
    """Outcome of a standalone compile."""

    LOWERED = "lowered"
    UNSUPPORTED_CONTRACT = "unsupported_contract"
    COMPILER_FAILED = "compiler_failed"
    BLOCKED_MISSING_CCE_MLIR = "blocked_missing_cce_mlir"
    BLOCKED_MISSING_CANN = "blocked_missing_cann"


@dataclass(frozen=True)
class KernelArtifact:
    """Files produced for one kernel."""

    kernel: CceKernel
    mlir_path: Path
    lowered_path: Path
    object_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class CompileResult:
    """Result of :func:`compile_program`."""

    status: CompileStatus
    artifacts: tuple[KernelArtifact, ...] = ()
    diagnostics: str = ""

    @property
    def lowered(self) -> bool:
        """Whether every kernel produced a device object."""
        return self.status is CompileStatus.LOWERED


def compile_program(
    program: Any,
    output_dir: str | Path,
    *,
    target: Any | None = None,
    host_device: int = 0,
) -> CompileResult:
    """Compile every Buffer-stage InCore function in a program.

    Args:
        program: Post-pass PyPTO ``Program``.
        output_dir: Directory for emitted MLIR, device objects and manifests.
        target: cce-mlir target; resolved from the configured backend when None.
        host_device: Logical device index embedded in the standalone host
            driver. Callers pass the queue's ``TASK_DEVICE`` (default ``0``);
            the production runtime path does not use this driver.

    Returns:
        The compile result. Blocked and refused outcomes are returned, not
        raised, so the caller can report them.
    """
    # Imported here so the typed front end stays usable without cce-mlir.
    from .analysis import lower_program
    from .emit import emit_module

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "report.json"

    try:
        kernels = lower_program(program, target=target)
    except CceMlirError as error:
        _write_report(report_path, CompileStatus.UNSUPPORTED_CONTRACT, str(error))
        return CompileResult(CompileStatus.UNSUPPORTED_CONTRACT, diagnostics=str(error))

    try:
        toolkit = cann_toolkit()
        modules = [(kernel, emit_module(kernel, host_device=host_device)) for kernel in kernels]
    except ImportError as error:
        _write_report(report_path, CompileStatus.BLOCKED_MISSING_CCE_MLIR, str(error))
        return CompileResult(CompileStatus.BLOCKED_MISSING_CCE_MLIR, diagnostics=str(error))
    except CceMlirError as error:
        status = (
            CompileStatus.BLOCKED_MISSING_CANN
            if error.reason == "missing-cann"
            else CompileStatus.COMPILER_FAILED
        )
        _write_report(report_path, status, str(error))
        return CompileResult(status, diagnostics=str(error))

    revisions = None
    artifacts: list[KernelArtifact] = []
    for kernel, module in modules:
        mlir_path = destination / f"{kernel.function_name}.mlir"
        lowered_path = destination / f"{kernel.function_name}.lowered.mlir"
        object_path = destination / f"{kernel.function_name}.o"
        manifest_path = destination / f"{kernel.function_name}.manifest.json"
        mlir_path.write_text(str(module), encoding="utf-8")
        try:
            lowered_text = lower_to_llvm(module, toolkit=toolkit)
            object_bytes = extract_device_object(module)
        except CceMlirError as error:
            _write_report(report_path, CompileStatus.COMPILER_FAILED, str(error))
            return CompileResult(CompileStatus.COMPILER_FAILED, tuple(artifacts), diagnostics=str(error))
        lowered_path.write_text(lowered_text, encoding="utf-8")
        object_path.write_bytes(object_bytes)
        try:
            symbols = symbol_names(object_bytes)
        except ElfError as error:
            _write_report(report_path, CompileStatus.COMPILER_FAILED, str(error))
            return CompileResult(CompileStatus.COMPILER_FAILED, tuple(artifacts), diagnostics=str(error))
        manifest = build_manifest(
            kernel,
            status=CompileStatus.LOWERED.value,
            object_bytes=object_bytes,
            object_symbols=symbols,
            revisions=revisions,
        )
        write_manifest(manifest, manifest_path)
        artifacts.append(KernelArtifact(kernel, mlir_path, lowered_path, object_path, manifest_path))

    _write_report(report_path, CompileStatus.LOWERED, "")
    return CompileResult(CompileStatus.LOWERED, tuple(artifacts))


def _write_report(path: Path, status: CompileStatus, diagnostics: str) -> None:
    path.write_text(
        json.dumps({"status": status.value, "diagnostics": diagnostics}, indent=2) + "\n",
        encoding="utf-8",
    )
