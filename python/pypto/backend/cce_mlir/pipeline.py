# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""In-process cce-mlir pipelines.

Passes run through cce-mlir's own ``PassManager`` over the pipelines registered
by ``_ccePasses``. PyPTO never drives cce-opt as a subprocess; the device
compiler that cce-mlir's own pass invokes (bisheng/ld.lld) remains cce-mlir's
business.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .elf import ElfError, is_relocatable, symbol_names
from .errors import CceMlirError
from .kernel_ir import CceKernel

__all__ = [
    "cann_toolkit",
    "compile_device_function",
    "compile_device_object",
    "extract_device_object",
    "lower_to_llvm",
]

# cce-mlir spells the bisheng arch with a "-cube" suffix for both core kinds;
# the per-function target-cpu selects cube vs vector.
_DEVICE_ARCH: dict[str, str] = {"a3": "dav-c220-cube", "a5": "dav-c310-cube"}

_ELF_MAGIC = b"\x7fELF"


class CceToolkit:
    """CANN paths and architecture used by the cce-mlir pipelines."""

    def __init__(self, root: str, arch: str) -> None:
        self.root = root
        self.arch = arch

    def pipeline_options(self) -> str:
        """Pipeline option block for ``npu-lower-pto-to-llvm-pipeline``."""
        return f"cann-root={self.root} cann-arch={self.arch}"


def cann_toolkit() -> CceToolkit:
    """Discover the CANN installation through cce-mlir.

    Returns:
        The discovered toolkit.

    Raises:
        CceMlirError: When cce-mlir cannot locate CANN.
    """
    try:
        from cce_mlir import cann
    except ImportError as exc:
        raise ImportError(
            "the cce_mlir Python package is required for the cce-mlir route; "
            "use docker/pypto_cce_mlir.Dockerfile or install cce-mlir's bindings"
        ) from exc
    try:
        toolkit = cann.discover()
    except Exception as exc:
        raise CceMlirError(
            "missing-cann",
            "cce-mlir could not discover a CANN installation",
        ) from exc
    return CceToolkit(root=toolkit.root, arch=toolkit.arch)


def lower_to_llvm(module: Any, *, toolkit: CceToolkit | None = None) -> str:
    """Lower a built module through ``npu-lower-pto-to-llvm-pipeline`` in place.

    Args:
        module: A ``cce_mlir.ir.Module`` produced by
            :func:`pypto.backend.cce_mlir.emit.emit_module`.
        toolkit: CANN toolkit; discovered when ``None``.

    Returns:
        The lowered module text.
    """
    from cce_mlir.passmanager import PassManager

    effective = toolkit or cann_toolkit()
    with module.context:
        pass_manager = PassManager("builtin.module")
        pass_manager.add("pto-propagate-layout")
        pass_manager.add("npu-lower-pto-to-llvm-pipeline{" + effective.pipeline_options() + "}")
        pass_manager.run(module.operation)
    return str(module)


def extract_device_object(module: Any) -> bytes:
    """Return the device object embedded by the cce-mlir pipeline.

    Args:
        module: A module that has already been lowered to LLVM.

    Returns:
        Raw ``ET_REL`` ELF bytes.

    Raises:
        CceMlirError: When the module embeds no device object, or it is not a
            relocatable ELF.
    """

    for operation in module.body.operations:
        if operation.name != "llvm.mlir.global":
            continue
        attribute = dict(operation.attributes).get("value")
        if attribute is None:
            continue
        try:
            data = attribute.value_bytes
        except Exception:  # non-string attributes (e.g. dense initializers)
            continue
        if not isinstance(data, bytes) or data[:4] != _ELF_MAGIC:
            continue
        try:
            if not is_relocatable(data):
                raise CceMlirError(
                    "non-relocatable-object",
                    "cce-mlir embedded an ELF that is not relocatable; the route "
                    "needs an ET_REL object to link under kernel_entry",
                )
        except ElfError as exc:
            raise CceMlirError("invalid-device-object", str(exc)) from exc
        return data
    raise CceMlirError(
        "missing-device-object",
        "the lowered module embeds no ELF device object",
    )


def compile_device_object(
    module: Any,
    *,
    toolkit: CceToolkit | None = None,
    expected_symbol: str | None = None,
) -> bytes:
    """Lower and extract the device object in one step.

    Args:
        module: A built (not yet lowered) module.
        toolkit: CANN toolkit; discovered when ``None``.
        expected_symbol: Device entry symbol that must be defined.

    Returns:
        Raw ``ET_REL`` ELF bytes.

    Raises:
        CceMlirError: When the object is missing, not relocatable, or does not
            define ``expected_symbol``.
    """
    lower_to_llvm(module, toolkit=toolkit)
    data = extract_device_object(module)
    if expected_symbol is not None:
        present = expected_symbol in symbol_names(data)
        if not present:
            raise CceMlirError(
                "missing-device-symbol",
                f"device object does not define {expected_symbol!r}",
            )
    return data


def _tool(binary: str, fallback: Path | None = None) -> str:
    """Resolve a driver binary, preferring ``PATH``."""
    found = shutil.which(binary)
    if found is not None:
        return found
    if fallback is not None and fallback.is_file():
        return str(fallback)
    raise CceMlirError(
        "missing-toolchain",
        f"{binary!r} was not found on PATH or at {fallback}",
    )


def _run_tool(command: list[str], *, label: str) -> str:
    """Run a compiler driver, raising its stderr on failure."""
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise CceMlirError(
            "driver-failed",
            f"{label} failed with exit code {completed.returncode}:\n{completed.stderr.strip()}",
        )
    return completed.stdout


def _as_standalone_function(text: str, device_symbol: str, symbol: str) -> str:
    """Turn cce-mlir's translated kernel entry into a plain device function.

    cce-mlir's translator only applies AI-core target attributes to functions
    that carry the kernel marker, and it attaches kernel annotations and renames
    the symbol in the same step. For the PyPTO runtime the body must be a
    callable function: keep the target attributes, drop the kernel annotations
    and use the bare symbol. Upstream should grow a native mode for this.
    """
    text = text.replace(device_symbol, symbol)
    return "\n".join(
        line
        for line in text.splitlines()
        if not line.startswith("!hivm.annotations") and not (line.startswith("!") and '"kernel"' in line)
    )


def compile_device_function(
    module: Any,
    kernel: CceKernel,
    *,
    toolkit: CceToolkit | None = None,
) -> bytes:
    """Compile the kernel body as a standalone, linkable device function.

    Unlike :func:`compile_device_object`, which produces cce-mlir's kernel entry
    (`<name>__kernel0`), this product defines the bare `<name>` symbol with a
    pointer-only ABI and no kernel metadata, so a PyPTO ``kernel_entry`` adapter
    can call it.

    The kernel marker is kept through cce-mlir's lowering and translation
    because that is the only path that stamps the per-function AI-core
    ``target-cpu``/``target-features``; the translated LLVM IR then has its
    kernel symbol and annotations rewritten.

    Args:
        module: A built module from
            :func:`pypto.backend.cce_mlir.emit.emit_module`.
        kernel: Typed kernel IR for the symbols and target.
        toolkit: CANN toolkit; discovered when ``None``.

    Returns:
        Raw ``ET_REL`` ELF bytes defining ``kernel.symbol_name``.

    Raises:
        CceMlirError: On a pipeline, translation, or codegen failure, or when
            the object does not define the standalone symbol.
    """
    from cce_mlir.passmanager import PassManager

    effective = toolkit or cann_toolkit()
    with module.context:
        pass_manager = PassManager("builtin.module")
        # Fold/cancel relayout chains before they are materialized into CCE
        # memory operations. Today's kernels have no cancelling pair, so this is
        # a no-op; it is where vector chains of pack/pointwise/unpack win.
        pass_manager.add("pto-propagate-layout")
        pass_manager.add("npu-lower-pto-to-cce-pipeline")
        pass_manager.run(module.operation)
        pass_manager = PassManager("builtin.module")
        pass_manager.add("npu-lower-cce-to-llvm-device-only-pipeline")
        pass_manager.run(module.operation)

    translate = _tool("cce-translate", Path("/opt/cce-mlir/build/bin/cce-translate"))
    bisheng = _tool("bisheng", Path(effective.root) / "bin" / "bisheng")

    with tempfile.TemporaryDirectory(prefix="pypto-cce-function-") as directory:
        mlir_path = Path(directory) / "device.mlir"
        mlir_path.write_text(str(module), encoding="utf-8")
        llvm_ir = _run_tool(
            [translate, str(mlir_path), "--cce-to-llvmir"],
            label="cce-translate",
        )
        llvm_ir = _as_standalone_function(llvm_ir, kernel.device_symbol, kernel.symbol_name)
        llvm_path = Path(directory) / "device.ll"
        llvm_path.write_text(llvm_ir + "\n", encoding="utf-8")
        object_path = Path(directory) / "device.o"
        _run_tool(
            [
                bisheng,
                "-x",
                "ir",
                "-c",
                "-O3",
                "--cce-aicore-only",
                "--target=hiipu64-hisilicon-cce",
                f"--cce-aicore-arch={_DEVICE_ARCH[kernel.target.value]}",
                "-mllvm",
                "-cce-aicore-stack-size=0x8000",
                "-mllvm",
                "-cce-aicore-function-stack-size=0x8000",
                "-mllvm",
                "-cce-aicore-addr-transform",
                "-o",
                str(object_path),
                str(llvm_path),
            ],
            label="bisheng",
        )
        data = object_path.read_bytes()

    if kernel.symbol_name not in symbol_names(data):
        raise CceMlirError(
            "missing-device-symbol",
            f"standalone device function does not define {kernel.symbol_name!r}",
        )
    return data
