# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Diagnostics for the PyPTO to cce-mlir compiler route.

Every refusal carries a stable ``reason`` category so tests and callers can
assert *why* a kernel is unsupported instead of matching message text.
"""

from __future__ import annotations

from typing import Any


class CceMlirError(RuntimeError):
    """A kernel contract the cce-mlir route refuses to lower.

    Attributes:
        reason: Stable machine-readable category, e.g. ``"unsupported-op"``.
        op_name: Registered PyPTO operator involved, when the failure is
            attributable to one operator.
        span: The IR source span, when a node carrying one is in scope.
    """

    def __init__(
        self,
        reason: str,
        message: str,
        *,
        op_name: str | None = None,
        span: Any | None = None,
    ) -> None:
        self.reason = reason
        self.op_name = op_name
        self.span = span
        context = f" [{op_name}]" if op_name else ""
        location = f" at {span}" if span is not None else ""
        super().__init__(f"{message}{context}{location}")
