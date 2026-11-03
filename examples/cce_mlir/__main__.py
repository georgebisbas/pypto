"""CLI for the standalone PyPTO-to-cce-mlir experiment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .compiler import CceMlirCompiler
from .lowering import CceTarget, lower_pypto_matmul


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("compile", "run"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cce_mlir"))
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path(__file__).parent / "fixtures" / "pypto_matmul.py",
    )
    parser.add_argument("--cce-opt")
    parser.add_argument("--cann-root")
    parser.add_argument("--cann-arch")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    spec = lower_pypto_matmul(args.fixture, target=CceTarget.A3)
    result = CceMlirCompiler(
        cce_opt=args.cce_opt,
        cann_root=args.cann_root,
        cann_arch=args.cann_arch,
    ).compile(spec, args.output_dir)
    print(result.status)
    if args.command == "run" and result.status != "lowered":
        print("NPU run not attempted because compilation did not produce a lowered artifact")
        return 2
    if args.command == "run":
        import numpy as np
        from cce_mlir.cpu_runner import setup_context
        from cce_mlir.ir import Context, Module
        from cce_mlir.npu_runner import cann_pipeline, run_host

        os.environ.setdefault("ASCEND_VISIBLE_DEVICES", args.device)
        a = np.ones((64, 128), dtype=np.float16)
        b = np.ones((128, 64), dtype=np.float16)
        c = np.zeros((64, 64), dtype=np.float32)
        with Context() as context:
            setup_context(context)
            run_host(
                Module.parse(result.lowered_path.read_text()),
                "run",
                a,
                b,
                c,
                pipeline=cann_pipeline("npu-lower-pto-to-llvm-pipeline"),
            )
        np.testing.assert_allclose(c, a.astype(np.float32) @ b.astype(np.float32), rtol=1e-3, atol=1e-3)
        print("NPU smoke test passed")
    return (
        0
        if result.status in {"lowered", "blocked_missing_cce_opt", "blocked_missing_cann_configuration"}
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
