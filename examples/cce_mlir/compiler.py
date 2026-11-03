"""cce-opt driver for the standalone experimental bridge."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .lowering import KernelSpec, emit_cce_mlir
from .manifest import write_manifest


@dataclass(frozen=True)
class CompileResult:
    status: str
    input_path: Path
    lowered_path: Path | None
    manifest_path: Path
    command: tuple[str, ...] = ()
    returncode: int | None = None
    stderr: str = ""


class CceMlirCompiler:
    def __init__(
        self,
        *,
        cce_opt: str | None = None,
        cann_root: str | None = None,
        cann_arch: str | None = None,
    ):
        self.cce_opt = cce_opt
        self.cann_root = cann_root
        self.cann_arch = cann_arch

    def _find_cce_opt(self) -> str | None:
        if not self.cce_opt:
            return shutil.which("cce-opt")
        candidate = Path(self.cce_opt)
        if candidate.parent != Path("."):
            return self.cce_opt if candidate.is_file() and os.access(candidate, os.X_OK) else None
        return shutil.which(self.cce_opt)

    def command(self, input_path: Path) -> tuple[str, ...]:
        compiler = self._find_cce_opt() or self.cce_opt or "cce-opt"
        cann_root = self.cann_root or os.environ.get("CANN_ROOT")
        cann_arch = self.cann_arch or os.environ.get("CANN_ARCH")
        if not cann_root or not cann_arch:
            raise ValueError("CANN_ROOT and CANN_ARCH are required for the LLVM pipeline")
        pipeline = f"--npu-lower-pto-to-llvm-pipeline=cann-root={cann_root} cann-arch={cann_arch}"
        return (compiler, str(input_path), pipeline)

    def compile(self, spec: KernelSpec, output_dir: Path) -> CompileResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        input_path = output_dir / f"{spec.name}.mlir"
        lowered_path = output_dir / f"{spec.name}.lowered.mlir"
        manifest_path = output_dir / f"{spec.name}.manifest.json"
        report_path = output_dir / f"{spec.name}.report.json"
        input_path.write_text(emit_cce_mlir(spec))
        compiler = self._find_cce_opt()
        if compiler is None:
            result = CompileResult("blocked_missing_cce_opt", input_path, None, manifest_path)
            write_manifest(spec, None, manifest_path, status=result.status)
            report_path.write_text(
                json.dumps(
                    {
                        "status": result.status,
                        "reason": "cce-opt is not installed or executable",
                        "input": str(input_path),
                    },
                    indent=2,
                )
                + "\n"
            )
            return result
        try:
            command = self.command(input_path)
        except ValueError:
            result = CompileResult("blocked_missing_cann_configuration", input_path, None, manifest_path)
            write_manifest(spec, None, manifest_path, status=result.status)
            report_path.write_text(
                json.dumps(
                    {
                        "status": result.status,
                        "reason": "CANN_ROOT and CANN_ARCH are required for the LLVM pipeline",
                        "input": str(input_path),
                    },
                    indent=2,
                )
                + "\n"
            )
            return result
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        lowered_path.write_text(completed.stdout)
        status = "lowered" if completed.returncode == 0 else "compiler_failed"
        write_manifest(spec, lowered_path, manifest_path, compiler_command=command, status=status)
        report_path.write_text(
            json.dumps(
                {
                    "status": status,
                    "returncode": completed.returncode,
                    "stderr": completed.stderr,
                    "command": list(command),
                    "input": str(input_path),
                    "lowered": str(lowered_path),
                },
                indent=2,
            )
            + "\n"
        )
        return CompileResult(
            status,
            input_path,
            lowered_path,
            manifest_path,
            command,
            completed.returncode,
            completed.stderr,
        )
