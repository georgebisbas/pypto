# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared measurement machinery for the PTOAS-versus-cce-mlir matrix.

Not collected by pytest: this module owns the route table, the case table's
measurement half, and the report format so the ST and any future driver agree on
one measurement.

The primary metric is the AICore task time from chip-swimlane level 1: the sum
of the AICore task cycles in ``chip_swimlane_records.json`` divided by the
recorded clock. It measures the compiled kernel rather than host dispatch. The
secondary metric is the median dispatch latency of a register-once
``ChipWorker`` handle.

Every measurement compiles the same ``@pl.jit`` kernel for the requested route,
so a route that refuses a kernel is reported as a typed status rather than
failing the whole matrix.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("simpler_setup")

from pypto.backend import BackendType  # noqa: E402
from pypto.backend.cce_mlir import CceMlirError  # noqa: E402
from pypto.pypto_core.passes import MemoryPlanner  # noqa: E402
from pypto.runtime import ChipWorker, ExecutionMode, RunConfig  # noqa: E402

__all__ = [
    "ROUTES",
    "Case",
    "Measurement",
    "Route",
    "aicore_task_us",
    "best_per_route",
    "measure",
    "render_table",
    "run_matrix",
    "write_report",
]

_RUNTIME = "tensormap_and_ringbuffer"
_WARMUP = 3


@dataclass(frozen=True)
class Route:
    """One (compiler, memory-planner) configuration under test."""

    label: str
    kernel_compiler: str
    memory_planner: MemoryPlanner


ROUTES: tuple[Route, ...] = (
    Route("ptoas/PYPTO", "ptoas", MemoryPlanner.PYPTO),
    Route("ptoas/PTOAS", "ptoas", MemoryPlanner.PTOAS),
    Route("cce_mlir/PYPTO", "cce_mlir", MemoryPlanner.PYPTO),
)


@dataclass(frozen=True)
class Case:
    """One kernel under test plus its inputs and golden."""

    name: str
    kernel: object
    make_inputs: Callable[[], list[torch.Tensor]]
    golden: Callable[[list[torch.Tensor]], torch.Tensor]
    tolerance: float
    tags: tuple[str, ...] = ()


@dataclass
class Measurement:
    """One (case, route) result. ``status`` is ``ok`` or a refusal reason."""

    case: str
    route: str
    status: str
    task_us: float | None = None
    task_count: int | None = None
    dispatch_ms: float | None = None
    max_abs_err: float | None = None
    compile_s: float = 0.0
    directory: str = ""
    diagnostics: str = ""

    @property
    def ok(self) -> bool:
        """Whether the route compiled, ran and matched its golden."""
        return self.status == "ok"


def _device() -> int:
    return int(__import__("os").environ.get("TASK_DEVICE", "0"))


def aicore_task_us(directory: Path) -> tuple[float, int]:
    """Return ``(AICore task microseconds, task count)`` for one run.

    ``chip_swimlane_records.json`` lists one record per AICore task as
    ``[core_id, task_token, reg_task_id, start_cycles, end_cycles, ...]``.
    """
    path = directory / "dfx_outputs" / "chip_swimlane_records.json"
    if not path.is_file():
        return float("nan"), 0
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data["aicore_tasks"]
    cycles = sum(record[4] - record[3] for record in records)
    clock = data["metadata"]["clock_freq_hz"]
    return cycles * 1e6 / clock, len(records)


def _run_config(route: Route, directory: Path, **overrides) -> RunConfig:
    base = {
        "kernel_compiler": route.kernel_compiler,
        "memory_planner": route.memory_planner,
        "backend_type": BackendType.Ascend910,
        "execution_mode": ExecutionMode.ONBOARD,
        "device_id": _device(),
        "save_kernels": True,
        "save_kernels_dir": str(directory),
    }
    base.update(overrides)
    return RunConfig(**base)


def measure(case: Case, route: Route, root: Path, iterations: int) -> Measurement:
    """Compile, validate and time one case on one route."""
    directory = root / case.name / route.label.replace("/", "_")
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        compiled = case.kernel.compile(config=_run_config(route, directory))
    except CceMlirError as exc:
        return Measurement(
            case=case.name,
            route=route.label,
            status=f"refused:{exc.reason}",
            directory=str(directory),
            diagnostics=str(exc),
        )
    except Exception as exc:  # a compile failure is data, not a test crash
        return Measurement(
            case=case.name,
            route=route.label,
            status="compile-failed",
            compile_s=time.perf_counter() - started,
            directory=str(directory),
            diagnostics=f"{type(exc).__name__}: {exc}",
        )
    compile_s = time.perf_counter() - started

    host = case.make_inputs()
    worker = ChipWorker(
        config=RunConfig(execution_mode=ExecutionMode.ONBOARD, device_id=_device()), runtime=_RUNTIME
    )
    try:
        device_tensors = [worker.alloc_tensor(tuple(t.shape), t.dtype, init=t) for t in host[:-1]]
        handle = worker.register(compiled)
        for _ in range(_WARMUP):
            handle(*device_tensors, host[-1])

        latency = []
        for _ in range(iterations):
            tick = time.perf_counter()
            handle(*device_tensors, host[-1])
            latency.append((time.perf_counter() - tick) * 1e3)

        expected = case.golden(host)
        error = (host[-1] - expected).abs().max().item()

        # One per-call run with chip-swimlane level 1 records the AICore task.
        compiled(*host, config=_run_config(route, directory, enable_chip_swimlane=1))
        task_us, task_count = aicore_task_us(directory)
    except Exception as exc:
        return Measurement(
            case=case.name,
            route=route.label,
            status="run-failed",
            compile_s=compile_s,
            directory=str(directory),
            diagnostics=f"{type(exc).__name__}: {exc}",
        )
    finally:
        worker.close()

    status = "ok" if error <= case.tolerance else "numerical-mismatch"
    return Measurement(
        case=case.name,
        route=route.label,
        status=status,
        task_us=task_us,
        task_count=task_count,
        dispatch_ms=statistics.median(latency),
        max_abs_err=error,
        compile_s=compile_s,
        directory=str(directory),
    )


def run_matrix(
    cases: list[Case], routes: tuple[Route, ...], root: Path, iterations: int
) -> list[Measurement]:
    """Measure every case on every route, in case-major order."""
    results = []
    for case in cases:
        for route in routes:
            results.append(measure(case, route, root, iterations))
    return results


def best_per_route(measurements: list[Measurement]) -> dict[str, float]:
    """The fastest AICore task time each compiler produced, across planners.

    The PTOAS compiler's two planners are collapsed to one entry
    (``ptoas``) so the comparison is compiler-to-compiler; the per-planner rows
    remain in the report for anyone who wants to see which planner won.
    """
    best: dict[str, float] = {}
    for measurement in measurements:
        if not measurement.ok or measurement.task_us is None:
            continue
        compiler = measurement.route.split("/", 1)[0]
        if compiler not in best or measurement.task_us < best[compiler]:
            best[compiler] = measurement.task_us
    return best


def render_table(measurements: list[Measurement]) -> str:
    """Render the matrix as a fixed-width table."""
    header = (
        f"{'case':22s} {'route':16s} {'status':10s} {'task_us':>9s} "
        f"{'tasks':>5s} {'dispatch_ms':>11s} {'max_err':>9s} {'compile_s':>9s}"
    )
    lines = [header, "-" * len(header)]
    for m in measurements:
        task = "" if m.task_us is None else f"{m.task_us:9.2f}"
        tasks = "" if m.task_count is None else f"{m.task_count:5d}"
        dispatch = "" if m.dispatch_ms is None else f"{m.dispatch_ms:11.3f}"
        error = "" if m.max_abs_err is None else f"{m.max_abs_err:9.2e}"
        lines.append(
            f"{m.case:22s} {m.route:16s} {m.status:10s} {task:>9s} "
            f"{tasks:>5s} {dispatch:>11s} {error:>9s} {m.compile_s:9.2f}"
        )
    return "\n".join(lines)


def write_report(measurements: list[Measurement], path: Path) -> None:
    """Write a machine-readable matrix plus the per-compiler bests."""
    payload = {
        "schema": 1,
        "measurements": [m.__dict__ for m in measurements],
        "best_per_compiler_task_us": best_per_route(measurements),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
