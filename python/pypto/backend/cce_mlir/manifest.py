# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reproducibility manifest for a compiled cce-mlir kernel.

The manifest records the derived contract (parameters, ops, target, symbols),
toolchain revisions, the device-object digest and (when the run came from the
NPU queue) the ``task-submit`` task id, so an artifact can be traced back to the
exact inputs that produced it.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .kernel_ir import CceKernel

__all__ = ["build_manifest", "default_revisions", "object_digest", "write_manifest"]


def _revision(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def default_revisions() -> dict[str, str]:
    """Record the PyPTO, cce-mlir and CANN revisions visible to this process."""
    return {
        "pypto": _revision(Path(__file__).resolve().parents[3]),
        "cce_mlir": os.environ.get("CCE_MLIR_COMMIT", "unknown"),
        "cann": os.environ.get("CANN_VERSION", "unknown"),
    }


def object_digest(data: bytes) -> str:
    """SHA-256 digest of the device object."""
    return hashlib.sha256(data).hexdigest()


def _queue_task_id() -> str:
    """The ``task-submit`` job id, when this process runs under the queue.

    The queue exports ``ASCEND_PROCESS_LOG_PATH`` as
    ``.../logs/<task-id>.ascend``; older/other queue versions may export
    ``TASKQUEUE_TASK_ID`` directly.
    """
    explicit = os.environ.get("TASKQUEUE_TASK_ID", "")
    if explicit:
        return explicit
    log_path = os.environ.get("ASCEND_PROCESS_LOG_PATH", "")
    name = Path(log_path).name
    if name.startswith("task_") and name.endswith(".ascend"):
        return name[: -len(".ascend")]
    return ""


def build_manifest(
    kernel: CceKernel,
    *,
    status: str,
    object_bytes: bytes | None = None,
    object_symbols: Iterable[str] = (),
    revisions: dict[str, str] | None = None,
    diagnostics: str = "",
) -> dict[str, Any]:
    """Assemble the manifest dictionary for one kernel.

    Args:
        kernel: The typed kernel IR.
        status: Compilation status (``lowered`` or a refusal category).
        object_bytes: Device object bytes when compilation succeeded.
        object_symbols: Symbol names present in the device object.
        revisions: Toolchain revisions; discovered when ``None``.
        diagnostics: Human-readable refusal detail.

    Returns:
        A JSON-serialisable manifest.
    """
    return {
        "kernel": kernel.function_name,
        "module": kernel.module_name,
        "symbol": kernel.symbol_name,
        "device_symbol": kernel.device_symbol,
        "target": kernel.target.value,
        "kernel_kind": kernel.kind.value,
        "parameters": [
            {
                "name": parameter.name,
                "shape": list(parameter.shape),
                "dtype": parameter.element.mlir,
                "memory_space": parameter.space.value,
                "direction": parameter.direction.name if parameter.direction else None,
                "bytes": parameter.byte_size,
            }
            for parameter in kernel.parameters
        ],
        "operations": [
            {"name": op.op_name, "write": op.write_tensor.name if op.write_tensor else None}
            for op in kernel.ops
        ],
        "revisions": revisions if revisions is not None else default_revisions(),
        "queue_task_id": _queue_task_id(),
        "object_sha256": object_digest(object_bytes) if object_bytes is not None else None,
        "object_symbols": sorted(set(object_symbols)),
        "status": status,
        "diagnostics": diagnostics,
    }


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    """Write the manifest as pretty JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
