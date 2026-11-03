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

from typing import Any

from .elf import ElfError, is_relocatable, symbol_names
from .errors import CceMlirError

__all__ = [
    "cann_toolkit",
    "compile_device_object",
    "extract_device_object",
    "lower_to_llvm",
]

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
