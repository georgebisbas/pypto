# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Emitter registry for the cce-mlir builder.

Adding an operation is one :func:`register` call, never an edit to a dispatch
chain. Registration is keyed by the registered PyPTO buffer-op name.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .kernel_ir import CceOp

__all__ = ["EMITTERS", "EmitterRegistry", "EmitterFn"]


class EmitterFn(Protocol):
    """Emit one :class:`CceOp` into the MLIR module under construction."""

    def __call__(self, op: CceOp, builder: object) -> object: ...


class EmitterRegistry:
    """Name -> emitter mapping with duplicate-registration protection."""

    def __init__(self) -> None:
        self._emitters: dict[str, EmitterFn] = {}

    def register(self, op_name: str) -> Callable[[EmitterFn], EmitterFn]:
        """Register an emitter for a registered PyPTO buffer-op name.

        Args:
            op_name: Registered operator name, e.g. ``"buffer.load"``.

        Returns:
            A decorator that registers and returns the emitter.

        Raises:
            ValueError: When the name is registered twice.
        """

        def decorator(emitter: EmitterFn) -> EmitterFn:
            if op_name in self._emitters:
                raise ValueError(f"an emitter for {op_name!r} is already registered")
            self._emitters[op_name] = emitter
            return emitter

        return decorator

    def get(self, op_name: str) -> EmitterFn | None:
        """Return the emitter for an op, or ``None`` when unsupported."""
        return self._emitters.get(op_name)

    def supported(self) -> frozenset[str]:
        """All registered op names."""
        return frozenset(self._emitters)


EMITTERS = EmitterRegistry()
