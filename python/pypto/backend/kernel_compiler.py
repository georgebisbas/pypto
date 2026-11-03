# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Kernel-compiler route selection shared by the compile front ends.

``PYPTO_KERNEL_COMPILER`` provides the default when a caller does not pass
``kernel_compiler`` explicitly, so an unmodified example can be run with::

    PYPTO_KERNEL_COMPILER=cce_mlir python examples/beginner/05_matmul.py

An explicit argument always wins over the environment.
"""

from __future__ import annotations

import os

__all__ = ["ENV_VAR", "KERNEL_COMPILERS", "default_kernel_compiler"]

ENV_VAR = "PYPTO_KERNEL_COMPILER"
KERNEL_COMPILERS = ("ptoas", "cce_mlir")


def default_kernel_compiler() -> str:
    """Return the kernel compiler selected for callers that omit the option.

    Returns:
        ``"ptoas"`` unless ``PYPTO_KERNEL_COMPILER`` names another route.

    Raises:
        ValueError: When the environment names an unknown route.
    """
    value = os.environ.get(ENV_VAR, "").strip().lower()
    if not value:
        return "ptoas"
    if value not in KERNEL_COMPILERS:
        supported = ", ".join(KERNEL_COMPILERS)
        raise ValueError(f"{ENV_VAR}={value!r} is not a kernel compiler; expected one of {supported}")
    return value
