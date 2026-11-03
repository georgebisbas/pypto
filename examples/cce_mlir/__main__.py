# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CLI for the PyPTO-to-cce-mlir compiler experiment.

``compile`` lowers the fixture through ``pypto.backend.cce_mlir`` and writes the
emitted MLIR, the device object and a manifest. ``run`` additionally executes
the generated host driver on an NPU. Inside the queue, the device is the granted
logical card from ``TASK_DEVICE`` (never a hand-picked physical id), and
``ASCEND_RT_VISIBLE_DEVICES`` is left to ``task-submit``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
from pypto.backend.cce_mlir import CceTarget
from pypto.backend.cce_mlir._dialects import make_context
from pypto.backend.cce_mlir.driver import CompileResult, CompileStatus, compile_program
from pypto.jit import JITFunction

_DEFAULT_FIXTURE = Path(__file__).parent / "fixtures" / "pypto_matmul.py"

# The fixture is a cube matmul; these are the NumPy carriers for its element
# types. bf16 is carried as its raw 16-bit bit pattern.
_NUMPY_DTYPES = {
    "f16": np.float16,
    "bf16": np.uint16,
    "f32": np.float32,
    "i8": np.int8,
    "i32": np.int32,
}


def _load_program(path: Path):
    spec = importlib.util.spec_from_file_location("pypto_cce_mlir_fixture", path.resolve())
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load fixture {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernels = [value for value in vars(module).values() if isinstance(value, JITFunction)]
    if len(kernels) != 1:
        raise SystemExit(f"expected exactly one JITFunction in {path}, found {len(kernels)}")
    return kernels[0].lower()


def _arrays(manifest: dict) -> list[np.ndarray]:
    arrays = []
    for parameter in manifest["parameters"]:
        dtype = _NUMPY_DTYPES[parameter["dtype"]]
        shape = tuple(parameter["shape"])
        fill = np.zeros if parameter["direction"] == "Out" else np.ones
        arrays.append(np.ascontiguousarray(fill(shape, dtype=dtype)))
    return arrays


def _run(result: CompileResult) -> int:
    from cce_mlir import ir as cce_ir
    from cce_mlir.npu_runner import cann_pipeline, run_host

    artifact = result.artifacts[0]
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    arrays = _arrays(manifest)

    context = make_context()
    with context, cce_ir.Location.unknown():
        module = cce_ir.Module.parse(artifact.mlir_path.read_text(encoding="utf-8"))
        run_host(
            module,
            "run",
            *arrays,
            pipeline=cann_pipeline("npu-lower-pto-to-llvm-pipeline"),
        )

    expected = arrays[0].astype(np.float32) @ arrays[1].astype(np.float32)
    np.testing.assert_allclose(arrays[-1], expected, rtol=1e-3, atol=1e-3)
    print("NPU smoke test passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("compile", "run"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cce_mlir"))
    parser.add_argument("--fixture", type=Path, default=_DEFAULT_FIXTURE)
    parser.add_argument("--target", choices=("a3", "a5"), default="a3")
    parser.add_argument(
        "--device",
        default=None,
        help="logical device id supplied by the queue (task-submit appends it); "
        "defaults to TASK_DEVICE, then 0. Never a physical card id.",
    )
    args = parser.parse_args()

    program = _load_program(args.fixture)
    host_device = int(args.device or os.environ.get("TASK_DEVICE", "0"))
    result = compile_program(
        program,
        args.output_dir,
        target=CceTarget(args.target),
        host_device=host_device,
    )
    print(result.status.value)
    if result.status is not CompileStatus.LOWERED:
        print(result.diagnostics)
        return 2 if args.command == "run" else 0

    if args.command == "compile":
        return 0
    return _run(result)


if __name__ == "__main__":
    raise SystemExit(main())
