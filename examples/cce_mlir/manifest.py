"""Reproducibility and ABI manifest for experimental artifacts."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .abi import audit
from .lowering import KernelSpec


def _revision(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def default_revisions() -> dict[str, str]:
    package_root = Path(__file__).resolve().parents[2]
    return {
        "pypto": _revision(package_root),
        "cce_mlir": os.environ.get("CCE_MLIR_COMMIT", "unknown"),
        "cann": os.environ.get("CANN_VERSION", "unknown"),
    }


def build_manifest(
    spec: KernelSpec,
    artifact: str | Path | None,
    *,
    revisions: dict[str, str] | None = None,
    compiler_command: tuple[str, ...] = (),
    status: str | None = None,
) -> dict[str, Any]:
    artifact_path = Path(artifact) if artifact else None
    abi = (
        audit(artifact_path)
        if artifact_path and artifact_path.is_file()
        else {
            "artifact": str(artifact_path) if artifact_path else None,
            "cce_device_symbols": [],
            "cce_acl_launcher_apis": [],
            "pypto_required_entry": "kernel_entry(__gm__ int64_t*)",
            "pypto_entry_present": False,
            "compatible_with_pypto_external_source": False,
            "status": "not_lowered",
        }
    )
    return {
        "kernel": spec.name,
        "target": spec.target.value,
        "kernel_kind": spec.kind,
        "arguments": [asdict(argument) for argument in spec.arguments],
        "operations": [asdict(operation) for operation in spec.operations],
        "revisions": revisions or default_revisions(),
        "compiler_command": list(compiler_command),
        "abi": abi,
        "status": status or str(abi["status"]),
    }


def write_manifest(
    spec: KernelSpec,
    artifact: str | Path | None,
    path: str | Path,
    *,
    revisions: dict[str, str] | None = None,
    compiler_command: tuple[str, ...] = (),
    status: str | None = None,
) -> Path:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        spec,
        artifact,
        revisions=revisions,
        compiler_command=compiler_command,
        status=status,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path
