# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Performance parity between the PTOAS and cce-mlir device-kernel routes.

Both routes are compiled for the same kernel, registered once on a ``ChipWorker``
and dispatched over device tensors, so the measurement is the runtime dispatch
plus the AICore task — not the per-call artifact assembly. The AICore task time
comes from chip-swimlane level 1 (``chip_swimlane_records.json``, 50 MHz clock).

Two workloads are covered: one MMAD and a 65-MMAD accumulation loop. The loop
is the scheduling-sensitive case: it is where the two compilers can differ.

The assertions are parity bounds, not targets: they catch a route that becomes
categorically slower, without pinning either implementation to the numbers this
host happened to produce.

Run through the queue on hardware::

    task-submit --device auto --device-num 1 --max-time 1800 --timeout 0 \
        --run 'cd <repo> && .venv/bin/python -m pytest \
              tests/st/runtime/cce_mlir/test_cce_mlir_performance.py -v \
              --platform a2a3 --device $TASK_DEVICE'
"""

import json
import os
import statistics
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cce_mlir")
pytest.importorskip("simpler_setup")

import pypto.language as pl  # noqa: E402
from pypto.backend import BackendType  # noqa: E402
from pypto.runtime import ChipWorker, ExecutionMode, RunConfig  # noqa: E402

_RUNTIME = "tensormap_and_ringbuffer"
_LOOP_TRIPS = 64
_DISPATCH_PARITY = 2.0
_TASK_PARITY = 2.0


@pl.jit
def single_matmul(
    a: pl.Tensor[[128, 128], pl.FP16],
    b: pl.Tensor[[128, 128], pl.FP16],
    c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        ta = pl.load(a, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
        tb = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
        la = pl.move(ta, target_memory=pl.MemorySpace.Left)
        lb = pl.move(tb, target_memory=pl.MemorySpace.Right)
        acc = pl.matmul(la, lb, out_dtype=pl.FP32)
        pl.store(acc, [0, 0], c)
    return c


@pl.jit
def loop_matmul(
    a: pl.Tensor[[128, 128], pl.FP16],
    b: pl.Tensor[[128, 128], pl.FP16],
    c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        ta = pl.load(a, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
        tb = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
        la = pl.move(ta, target_memory=pl.MemorySpace.Left)
        lb = pl.move(tb, target_memory=pl.MemorySpace.Right)
        acc = pl.matmul(la, lb, out_dtype=pl.FP32)
        for _ in pl.range(_LOOP_TRIPS):
            acc = pl.matmul_acc(acc, la, lb)
        pl.store(acc, [0, 0], c)
    return c


def _device() -> int:
    return int(os.environ.get("TASK_DEVICE", "0"))


def _task_us(directory: Path) -> float:
    """AICore task duration summed over the swimlane records, in microseconds."""
    path = directory / "dfx_outputs" / "chip_swimlane_records.json"
    if not path.is_file():
        return float("nan")
    data = json.loads(path.read_text(encoding="utf-8"))
    cycles = [record[4] - record[3] for record in data["aicore_tasks"]]
    return sum(cycles) * 1e6 / data["metadata"]["clock_freq_hz"]


def _measure(kernel, multiplier: int, compiler: str, directory: Path, iterations: int):
    device = _device()
    config = RunConfig(
        kernel_compiler=compiler,
        backend_type=BackendType.Ascend910,
        execution_mode=ExecutionMode.ONBOARD,
        device_id=device,
        save_kernels=True,
        save_kernels_dir=str(directory),
    )
    a = torch.randn((128, 128), dtype=torch.float16)
    b = torch.randn((128, 128), dtype=torch.float16)
    c = torch.zeros((128, 128), dtype=torch.float32)
    compiled = kernel.compile(config=config)

    worker = ChipWorker(
        config=RunConfig(execution_mode=ExecutionMode.ONBOARD, device_id=device),
        runtime=_RUNTIME,
    )
    try:
        a_dev = worker.alloc_tensor((128, 128), torch.float16, init=a)
        b_dev = worker.alloc_tensor((128, 128), torch.float16, init=b)
        handle = worker.register(compiled)
        for _ in range(3):
            handle(a_dev, b_dev, c)
        latency = []
        for _ in range(iterations):
            started = time.perf_counter()
            handle(a_dev, b_dev, c)
            latency.append((time.perf_counter() - started) * 1e3)
        error = (c - multiplier * (a.float() @ b.float())).abs().max().item()

        # A per-call run with chip-swimlane level 1 records the AICore task.
        compiled(
            a,
            b,
            c,
            config=RunConfig(
                execution_mode=ExecutionMode.ONBOARD,
                device_id=device,
                save_kernels=True,
                save_kernels_dir=str(directory),
                enable_chip_swimlane=1,
            ),
        )
        return statistics.median(latency), _task_us(directory), error
    finally:
        worker.close()


@pytest.mark.parametrize(
    ("name", "kernel", "multiplier", "tolerance", "iterations"),
    [
        ("single_mmad", single_matmul, 1, 1e-3, 40),
        ("loop_65_mmad", loop_matmul, _LOOP_TRIPS + 1, 2e-2, 30),
    ],
)
def test_backends_stay_within_parity(name, kernel, multiplier, tolerance, iterations, tmp_path):
    results = {}
    for compiler in ("ptoas", "cce_mlir"):
        directory = tmp_path / compiler
        dispatch_ms, task_us, error = _measure(kernel, multiplier, compiler, directory, iterations)
        assert error < tolerance, f"{compiler} {name} numerical error {error:.2e}"
        assert task_us > 0, f"{compiler} {name} produced no AICore task record"
        results[compiler] = (dispatch_ms, task_us)
        print(
            f"{name:12s} {compiler:9s} dispatch_ms={dispatch_ms:6.3f} "
            f"task_us={task_us:6.2f} max_err={error:.2e}"
        )

    dispatch_ratio = max(v[0] for v in results.values()) / min(v[0] for v in results.values())
    task_ratio = max(v[1] for v in results.values()) / min(v[1] for v in results.values())
    assert dispatch_ratio < _DISPATCH_PARITY, f"{name} dispatch ratio {dispatch_ratio:.2f}"
    assert task_ratio < _TASK_PARITY, f"{name} AICore task ratio {task_ratio:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
