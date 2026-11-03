# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Numerical validation of the cce-mlir route through PyPTO's runtime.

Each case compiles with ``kernel_compiler="cce_mlir"`` and dispatches through
``CompiledProgram`` — the adapter object is linked under ``kernel_entry``, so a
matching result proves the ABI bridge rather than cce-mlir's own host runner.

Run through the queue on hardware::

    task-submit --device auto --device-num 1 --max-time 1800 --timeout 0 \
        --run 'cd <repo> && .venv/bin/python -m pytest \
              tests/st/runtime/cce_mlir/test_cce_mlir_numerical.py -v \
              --platform a2a3 --device $TASK_DEVICE'
"""

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cce_mlir")
pytest.importorskip("simpler_setup")

import pypto.language as pl  # noqa: E402
from pypto.backend import BackendType  # noqa: E402
from pypto.runtime import ExecutionMode, RunConfig  # noqa: E402


def _config(tmp_path: Path) -> RunConfig:
    return RunConfig(
        kernel_compiler="cce_mlir",
        backend_type=BackendType.Ascend910,
        execution_mode=ExecutionMode.ONBOARD,
        device_id=int(os.environ.get("TASK_DEVICE", "0")),
        save_kernels=True,
        save_kernels_dir=str(tmp_path),
    )


@pl.jit
def cube_matmul(
    a: pl.Tensor[[64, 128], pl.FP16],
    b: pl.Tensor[[128, 64], pl.FP16],
    result: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        a_mat = pl.load(a, [0, 0], [64, 128], target_memory=pl.MemorySpace.Mat)
        b_mat = pl.load(b, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
        a_left = pl.move(a_mat, target_memory=pl.MemorySpace.Left)
        b_right = pl.move(b_mat, target_memory=pl.MemorySpace.Right)
        acc = pl.matmul(a_left, b_right, out_dtype=pl.FP32)
        pl.store(acc, [0, 0], result)
    return result


@pl.jit
def vector_add(
    lhs: pl.Tensor[[128, 128], pl.FP32],
    rhs: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        lhs_tile = pl.load(lhs, [0, 0], [128, 128])
        rhs_tile = pl.load(rhs, [0, 0], [128, 128])
        out_tile = pl.add(lhs_tile, rhs_tile)
        pl.store(out_tile, [0, 0], out)
    return out


def _assert_cce_artifact(output_dir: Path) -> None:
    config = (output_dir / "kernel_config.py").read_text(encoding="utf-8")
    assert '"link_objects"' in config
    assert list(output_dir.glob("kernels/*/*.cce.o"))
    assert list(output_dir.glob("kernels/*/*.cpp"))


def test_cube_matmul_runs_through_the_cce_mlir_route(tmp_path):
    config = _config(tmp_path)
    torch.manual_seed(0)
    a = torch.randn((64, 128), dtype=torch.float16)
    b = torch.randn((128, 64), dtype=torch.float16)
    c = torch.zeros((64, 64), dtype=torch.float32)

    cube_matmul(a, b, c, config=config)

    expected = a.float() @ b.float()
    torch.testing.assert_close(c, expected, rtol=1e-3, atol=1e-3)
    _assert_cce_artifact(tmp_path)


def test_vector_add_runs_through_the_cce_mlir_route(tmp_path):
    config = _config(tmp_path)
    torch.manual_seed(0)
    lhs = torch.randn((128, 128), dtype=torch.float32)
    rhs = torch.randn((128, 128), dtype=torch.float32)
    out = torch.zeros((128, 128), dtype=torch.float32)

    vector_add(lhs, rhs, out, config=config)

    torch.testing.assert_close(out, lhs + rhs, rtol=1e-5, atol=1e-5)
    _assert_cce_artifact(tmp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
