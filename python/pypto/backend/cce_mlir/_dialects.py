# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Construction of a strict cce-mlir MLIR context.

Unlike cce-mlir's test runner, this context deliberately does **not** allow
unregistered dialects: an unknown operation must fail verification instead of
being accepted as an opaque generic op.
"""

from __future__ import annotations

__all__ = ["make_context"]


def make_context() -> object:
    """Create an MLIR context with the dialects the emitter uses registered.

    Returns:
        A ``cce_mlir.ir.Context``.

    Raises:
        ImportError: When the external ``cce_mlir`` package is unavailable.
    """
    # Imported lazily: this module must be importable in environments without
    # the cce-mlir toolchain.
    from cce_mlir import ir as cir
    from cce_mlir._mlir_libs import _ccePasses
    from cce_mlir.dialects import cce as cce_dialect
    from cce_mlir.dialects import layout as layout_dialect
    from cce_mlir.dialects import npu as npu_dialect
    from cce_mlir.dialects import pto as pto_dialect

    context = cir.Context()
    _ccePasses.register_upstream_dialects(context)
    cce_dialect.register_dialect(context)
    npu_dialect.register_dialect(context)
    pto_dialect.register_dialect(context)
    layout_dialect.register_dialect(context)
    return context
