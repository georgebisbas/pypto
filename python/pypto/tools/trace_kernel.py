# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Trace one ``@pl.jit`` kernel from Python source down to device code.

The tool compiles the kernel once with every PyPTO pass dumped and every
backend artifact saved, then prints a map from each lowering stage to the file
that holds it. For the cce-mlir route it also runs cce-mlir's own pass manager
when ``cce-opt`` is available, so the PTO -> CCE -> LLVM descent can be read
without composing the commands by hand.

Examples::

    python -m pypto.tools.trace_kernel --file examples/cce_mlir/fixtures/pypto_matmul.py
    python -m pypto.tools.trace_kernel --file tests/st/runtime/cce_mlir/corpus.py \\
        --function matmul_kloop
    python -m pypto.tools.trace_kernel --file examples/beginner/05_matmul.py \\
        --samples f32:64x64,f32:64x64,f32:64x64

A kernel whose ``@pl.jit`` parameters carry bare ``pl.Tensor`` annotations has
no static shapes to specialize on; pass one ``--samples`` entry per parameter
(``dtype:rowsxcols``) to specialize from sample tensors instead.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# Stage label -> glob under ``passes_dump/``. Pass numbering shifts as the
# pipeline evolves, so match on the pass name.
_STAGES: tuple[tuple[str, str], ...] = (
    ("your Python, as PyPTO IR", "00_frontend.py"),
    ("SSA form", "*after_ConvertToSSA.py"),
    ("tensor ops become tile ops", "*after_ConvertTensorToTileOps.py"),
    ("L0 tiling", "*after_AutoTileMatmulL0.py"),
    ("tile memory spaces", "*after_InferTileMemorySpace.py"),
    ("tile layouts", "*after_ResolveBackendOpLayouts.py"),
    ("software pipelining", "*after_LowerPipelineLoops.py"),
    ("Buffer-IR gate", "*after_SelectBufferIRFunctions.py"),
    ("address assignment", "*after_AllocateMemoryAddr.py"),
    ("Buffer IR (backend handoff)", "*after_LowerTileToBuffer.py"),
)

_DTYPE_NAMES = ("f16", "f32", "bf16", "i8", "i32")


def parse_samples(spec: str) -> list[Any]:
    """Build sample tensors from ``dtype:rowsxcols`` entries.

    Args:
        spec: Comma-separated ``dtype:shape`` entries, one per kernel parameter,
            e.g. ``f16:128x512,f16:512x128,f32:128x128``.

    Returns:
        One zero-filled tensor per entry, in parameter order.

    Raises:
        ValueError: When an entry is malformed or names an unsupported dtype.
    """
    # Lazy: the compiler-only path (static annotations) needs no torch.
    import torch  # noqa: PLC0415

    dtypes = {
        "f16": torch.float16,
        "f32": torch.float32,
        "bf16": torch.bfloat16,
        "i8": torch.int8,
        "i32": torch.int32,
    }
    tensors = []
    for entry in spec.split(","):
        fields = entry.strip().split(":")
        if len(fields) != 2 or fields[0] not in dtypes:
            raise ValueError(
                f"invalid sample {entry!r}; expected dtype:shape with dtype one of {', '.join(_DTYPE_NAMES)}"
            )
        try:
            shape = [int(dimension) for dimension in fields[1].lower().split("x")]
        except ValueError as error:
            raise ValueError(f"invalid sample shape in {entry!r}") from error
        if not shape or any(dimension <= 0 for dimension in shape):
            raise ValueError(f"sample shape in {entry!r} must be positive")
        tensors.append(torch.zeros(shape, dtype=dtypes[fields[0]]))
    return tensors


def load_kernel(path: Path, function: str | None) -> Any:
    """Load a ``@pl.jit`` function from a Python file.

    Args:
        path: File defining the kernel (imported for its definitions only).
        function: Name of the kernel; required when the file defines several.

    Returns:
        The ``JITFunction``.

    Raises:
        SystemExit: When the file defines no matching kernel.
    """
    from pypto.jit import JITFunction  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location("pypto_traced_kernel", path.resolve())
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernels = {name: value for name, value in vars(module).items() if isinstance(value, JITFunction)}
    if function is not None:
        if function not in kernels:
            raise SystemExit(f"{path} defines no @pl.jit function named {function!r}")
        return kernels[function]
    if len(kernels) != 1:
        names = ", ".join(sorted(kernels)) or "none"
        raise SystemExit(f"{path} defines several @pl.jit functions ({names}); pass --function")
    return next(iter(kernels.values()))


def _find_cce_opt() -> str | None:
    """Locate ``cce-opt``, preferring ``PATH`` and the Docker install tree."""
    found = shutil.which("cce-opt")
    if found is not None:
        return found
    fallback = Path("/opt/cce-mlir/build/bin/cce-opt")
    return str(fallback) if fallback.is_file() else None


def _cce_passes(emitted: Path, output: Path) -> tuple[str, str] | None:
    """Run cce-mlir's PTO-to-CCE pipeline over ``emitted``.

    ``--mlir-print-ir-tree-dir`` makes cce-opt write one file per pass under
    ``<output>/cce_passes/``; the second run materializes the final CCE module.

    Returns the ``(pass directory, CCE-level module)`` paths, or ``None`` when
    ``cce-opt`` is unavailable.
    """
    cce_opt = _find_cce_opt()
    if cce_opt is None:
        return None
    passes = output / "cce_passes"
    passes.mkdir(parents=True, exist_ok=True)
    cce_level = output / "cce_level.mlir"
    command = [cce_opt, "--npu-lower-pto-to-cce-pipeline"]
    subprocess.run(
        [*command, "--mlir-print-ir-after-all", f"--mlir-print-ir-tree-dir={passes}", str(emitted)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    subprocess.run([*command, str(emitted), "-o", str(cce_level)], check=False)
    return str(passes), str(cce_level)


def trace(
    source: Path,
    *,
    function: str | None,
    samples: str | None,
    output: Path,
    compiler: str,
    planner: str | None,
) -> Path:
    """Compile one kernel and write its full descent under ``output``.

    Args:
        source: File defining the kernel.
        function: Kernel name when the file defines more than one.
        samples: ``dtype:shape`` sample spec for bare ``pl.Tensor`` parameters.
        output: Artifact directory.
        compiler: ``cce_mlir`` or ``ptoas``.
        planner: ``PYPTO`` or ``PTOAS`` memory planner; ``None`` keeps the
            ambient default (and is required while an outer ``PassContext``
            already owns the planner).

    Returns:
        ``output``.
    """
    from pypto.backend import BackendType  # noqa: PLC0415
    from pypto.ir import PassDumpLevel  # noqa: PLC0415
    from pypto.pypto_core.passes import MemoryPlanner  # noqa: PLC0415
    from pypto.runtime import ExecutionMode, RunConfig  # noqa: PLC0415

    kernel = load_kernel(source, function)
    output.mkdir(parents=True, exist_ok=True)
    kernel.compile(
        *parse_samples(samples) if samples else [],
        config=RunConfig(
            kernel_compiler=compiler,
            memory_planner=getattr(MemoryPlanner, planner) if planner else None,
            backend_type=BackendType.Ascend910,
            execution_mode=ExecutionMode.ONBOARD,
            codegen_only=True,
            dump_passes=PassDumpLevel.EXPLICIT,
            save_kernels=True,
            save_kernels_dir=str(output),
        ),
    )
    return output


def _report(output: Path, compiler: str) -> None:
    """Print the stage -> file map for one traced kernel."""
    print(f"\nartifacts: {output}")
    dumps = output / "passes_dump"
    if dumps.is_dir():
        print("\nPyPTO passes (passes_dump/):")
        for label, pattern in _STAGES:
            matches = sorted(dumps.glob(pattern))
            if matches:
                print(f"  {label:32s} {matches[0].name}")
        others = sorted(p.name for p in dumps.glob("*.py"))
        print(f"  ({len(others)} files total; 'ls {dumps}' for the rest)")

    if compiler != "cce_mlir":
        # With PTOAS_ROOT set, ``ptoas`` rewrites each kernel under ``ptoas/``;
        # without it PyPTO stops after codegen and the raw ``.pto`` lands under
        # ``kernels/<core>/``.
        pto_files = sorted(output.rglob("*.pto"))
        cpp_files = sorted(output.glob("ptoas/*.cpp"))
        print("\nPTOAS route:")
        for path in pto_files:
            print(f"  PyPTO codegen (.pto)             {path}")
        for path in cpp_files:
            print(f"  ptoas output (pto-isa C++)       {path}")
        if pto_files and not cpp_files:
            print("  (ptoas did not run: set PTOAS_ROOT to an installed ptoas to get its generated C++)")
        return

    artifacts = output / "cce_mlir"
    if not artifacts.is_dir():
        print("\nno cce_mlir/ artifacts: the kernel was not lowered by this route")
        return
    emitted = sorted(p for p in artifacts.glob("*.mlir") if not p.name.endswith(".lowered.mlir"))
    print("\ncce-mlir route:")
    for path in emitted:
        print(f"  PTO MLIR emitted by PyPTO        {path}")
        passes = _cce_passes(path, output)
        if passes is None:
            print("  (cce-opt not found; install cce-mlir to dump its passes)")
        else:
            files = sorted(Path(passes[0]).rglob("*.mlir"))
            print(f"  cce-mlir per-pass IR ({len(files):2d} files)   {passes[0]}/")
            for dump in files:
                print(f"    {dump.name}")
            print(f"  CCE-level module                 {passes[1]}")
    for path in sorted(artifacts.glob("*.lowered.mlir")):
        print(f"  device LLVM                      {path}")
    for path in sorted(output.glob("kernels/*/*.cpp")):
        print(f"  kernel_entry adapter             {path}")
    for path in sorted(output.glob("kernels/*/*.cce.o")):
        print(f"  device object (ET_REL)           {path}")

    print(
        "\nread in this order:\n"
        "  passes_dump/*after_ConvertTensorToTileOps.py   tiles appear\n"
        "  passes_dump/*after_InferTileMemorySpace.py     where each tensor lives\n"
        "  passes_dump/*after_LowerTileToBuffer.py        explicit buffers\n"
        "  cce_mlir/*.mlir                                same program in PTO MLIR\n"
        "  cce_passes/NN_<pass>.mlir                      one file per cce-mlir pass\n"
        "  cce_mlir/*.lowered.mlir                        device LLVM\n"
    )


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Trace a @pl.jit kernel from Python source down to device code.",
    )
    parser.add_argument("--file", type=Path, required=True, help="Python file defining the kernel")
    parser.add_argument("--function", help="kernel name when the file defines several")
    parser.add_argument(
        "--samples",
        help="dtype:shape per parameter for bare pl.Tensor annotations, "
        "e.g. f16:128x512,f16:512x128,f32:128x128",
    )
    parser.add_argument("--out", type=Path, help="artifact directory (default: build_output/trace_<stem>)")
    parser.add_argument("--compiler", default="cce_mlir", choices=("cce_mlir", "ptoas"))
    parser.add_argument(
        "--planner",
        default=None,
        choices=("PYPTO", "PTOAS"),
        help="memory planner; omit to use the default (an outer PassContext owns it)",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="also render <out>/trace.html: every stage diffed against the previous one",
    )
    args = parser.parse_args(argv)

    output = args.out or Path("build_output") / f"trace_{args.file.stem}"
    try:
        trace(
            args.file,
            function=args.function,
            samples=args.samples,
            output=output,
            compiler=args.compiler,
            planner=args.planner,
        )
    except ValueError as error:
        parser.error(str(error))
    except SystemExit as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    _report(output, args.compiler)
    if args.html:
        from pypto.tools.trace_html import write_trace_html  # noqa: PLC0415

        page = write_trace_html(output, args.compiler, args.function or args.file.stem)
        print(f"\nHTML descent: {page}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
