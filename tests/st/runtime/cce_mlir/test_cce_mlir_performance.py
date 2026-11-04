# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Fastest-baseline comparison between the PTOAS and cce-mlir kernel routes.

For each kernel in ``corpus.py`` the matrix compiles the same ``@pl.jit``
function through every route in ``bench_matrix.ROUTES``:

  - ``ptoas/PYPTO``   -- PTOAS compiler, PyPTO-baked offsets (``--pto-level=level3``)
  - ``ptoas/PTOAS``   -- PTOAS compiler, PTOAS memory planning (``--pto-level=level2``)
  - ``cce_mlir/PYPTO`` -- cce-mlir compiler, the only planner it can consume

Each route registers once on a ``ChipWorker``, dispatches over device tensors,
and reports the AICore task time from chip-swimlane level 1 as the primary
metric. The assertions are parity bounds, not targets: they catch a route that
becomes categorically slower without pinning either implementation to the
numbers this host happened to produce. ``best_by_case`` collapses the two
PTOAS planners to the compiler's faster result per kernel, which is the
"fastest baseline" for that compiler on that kernel.

Run through the queue on hardware::

    task-submit --device auto --device-num 1 --max-time 3600 --timeout 0 \\
        --run 'cd <repo> && .venv/bin/python -m pytest \\
              tests/st/runtime/cce_mlir/test_cce_mlir_performance.py -v \\
              --platform a2a3 --device $TASK_DEVICE'

Set ``PYPTO_CCE_MLIR_BENCH_JSON`` to choose where the machine-readable report
is written (defaults to ``<tmp_path>/bench_report.json``).
"""

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cce_mlir")
pytest.importorskip("simpler_setup")

import bench_matrix  # noqa: E402
import corpus  # noqa: E402

_DISPATCH_PARITY = 2.0
_TASK_PARITY = 2.0
_ITERATIONS = 30


def _f16(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.randn(shape, dtype=torch.float16)


def _f32(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.randn(shape, dtype=torch.float32)


def _mm_inputs() -> list[torch.Tensor]:
    return [_f16((128, 128)), _f16((128, 128)), torch.zeros((128, 128), dtype=torch.float32)]


def _kloop_inputs() -> list[torch.Tensor]:
    return [
        _f16((128, corpus.MM_K)),
        _f16((corpus.MM_K, 128)),
        torch.zeros((128, 128), dtype=torch.float32),
    ]


def _vector_inputs() -> list[torch.Tensor]:
    return [_f32((128, 128)), _f32((128, 128)), torch.zeros((128, 128), dtype=torch.float32)]


def _unary_vector_inputs() -> list[torch.Tensor]:
    return [_f32((128, 128)), torch.zeros((128, 128), dtype=torch.float32)]


def _mm_golden(tensors: list[torch.Tensor]) -> torch.Tensor:
    return tensors[0].float() @ tensors[1].float()


CASES = [
    bench_matrix.Case(
        "matmul_single",
        corpus.matmul_single,
        _mm_inputs,
        _mm_golden,
        tolerance=1e-3,
        tags=("matmul",),
    ),
    bench_matrix.Case(
        "matmul_accum",
        corpus.matmul_accum,
        _mm_inputs,
        lambda tensors: 65 * _mm_golden(tensors),
        tolerance=2e-2,
        tags=("matmul", "loop"),
    ),
    bench_matrix.Case(
        "matmul_kloop",
        corpus.matmul_kloop,
        _kloop_inputs,
        _mm_golden,
        tolerance=2e-2,
        tags=("matmul", "window"),
    ),
    bench_matrix.Case(
        "matmul_kloop_pipelined",
        corpus.matmul_kloop_pipelined,
        _kloop_inputs,
        _mm_golden,
        tolerance=2e-2,
        tags=("matmul", "pipeline"),
    ),
    bench_matrix.Case(
        "vector_add",
        corpus.vector_add,
        _vector_inputs,
        lambda tensors: tensors[0] + tensors[1],
        tolerance=1e-5,
        tags=("vector",),
    ),
    bench_matrix.Case(
        "vector_scale_add",
        corpus.vector_scale_add,
        _unary_vector_inputs,
        lambda tensors: tensors[0] * 1.5 + 1.0,
        tolerance=1e-5,
        tags=("vector", "scalar"),
    ),
    bench_matrix.Case(
        "mm_then_scale",
        corpus.mm_then_scale,
        _mm_inputs,
        lambda tensors: 0.5 * _mm_golden(tensors),
        tolerance=1e-3,
        tags=("fused",),
    ),
]


@pytest.fixture(scope="session")
def matrix(tmp_path_factory):
    """Measure the whole corpus once; every test reads the same results."""
    root = tmp_path_factory.mktemp("bench")
    measurements = bench_matrix.run_matrix(CASES, bench_matrix.ROUTES, root, _ITERATIONS)
    report = Path(os.environ.get("PYPTO_CCE_MLIR_BENCH_JSON", root / "bench_report.json"))
    bench_matrix.write_report(measurements, report)
    print("\n" + bench_matrix.render_table(measurements))
    print(f"\nfastest baseline (AICore task us): {bench_matrix.best_by_case(measurements)}")
    return measurements


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_case_across_routes(case, matrix):
    results = [result for result in matrix if result.case == case.name]

    supported = [result for result in results if result.ok]
    assert supported, f"{case.name} produced no successful route: " + "; ".join(
        f"{result.route}={result.status} ({result.diagnostics.splitlines()[0] if result.diagnostics else ''})"
        for result in results
    )
    for result in results:
        # A route either compiles, runs and matches its golden, or reports a
        # typed refusal. A compile/run failure is a regression, not a result.
        assert result.ok or result.status.startswith("refused:"), (
            f"{result.route} {case.name}: {result.status} "
            f"max_err={result.max_abs_err} diagnostics={result.diagnostics}"
        )
        if result.task_us is not None:
            assert result.task_us > 0, f"{result.route} {case.name} recorded no AICore task"

    if len(supported) < 2:
        return
    task_ratio = max(r.task_us for r in supported) / min(r.task_us for r in supported)
    dispatch_ratio = max(r.dispatch_ms for r in supported) / min(r.dispatch_ms for r in supported)
    assert task_ratio < _TASK_PARITY, f"{case.name} AICore task ratio {task_ratio:.2f}"
    assert dispatch_ratio < _DISPATCH_PARITY, f"{case.name} dispatch ratio {dispatch_ratio:.2f}"


def test_fastest_baseline_per_compiler(matrix):
    """Both compilers must produce a usable fastest baseline for every case."""
    best = bench_matrix.best_by_case(matrix)

    for case in CASES:
        per_compiler = best.get(case.name, {})
        missing = {"ptoas", "cce_mlir"} - set(per_compiler)
        assert not missing, f"{case.name}: no successful measurement for {sorted(missing)}"
        ratio = max(per_compiler.values()) / min(per_compiler.values())
        assert ratio < _TASK_PARITY, f"{case.name} compiler task ratio {ratio:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
