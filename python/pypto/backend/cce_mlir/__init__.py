# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PyPTO to cce-mlir compiler route.

The typed front end (:mod:`analysis`, :mod:`layout`, :mod:`kernel_ir`) has no
dependency on the external ``cce_mlir`` package. Components that build MLIR or
run cce-mlir pipelines import it lazily and report a clear diagnostic when the
toolchain is absent.
"""

from __future__ import annotations

import importlib.util

from .analysis import lower_function, lower_program, resolve_target
from .errors import CceMlirError
from .kernel_ir import (
    CceEncoding,
    CceKernel,
    CceKernelKind,
    CceLayout,
    CceMemorySpace,
    CceOp,
    CceScalar,
    CceTarget,
    CceTensor,
)

__all__ = [
    "CceEncoding",
    "CceKernel",
    "CceKernelKind",
    "CceLayout",
    "CceMemorySpace",
    "CceMlirError",
    "CceOp",
    "CceScalar",
    "CceTarget",
    "CceTensor",
    "cce_mlir_available",
    "lower_function",
    "lower_program",
    "resolve_target",
]


def cce_mlir_available() -> bool:
    """Whether the external ``cce_mlir`` Python package is importable.

    Returns:
        ``True`` when cce-mlir's bindings are installed (as they are inside
        ``docker/pypto_cce_mlir.Dockerfile``), ``False`` otherwise.
    """
    return importlib.util.find_spec("cce_mlir") is not None
