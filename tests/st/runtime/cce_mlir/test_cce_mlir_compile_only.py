# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile-only coverage of the performance corpus.

No device work: this compiles every ``corpus.py`` kernel through the PTOAS and
cce-mlir routes with ``codegen_only=True`` and checks the artifacts each route
must produce. Run it in the same environment as the performance matrix (the
runtime conftest requires visible NPU device nodes even for compile-only
tests), but it never takes a device or the queue::

    docker exec <queue-container> task-submit --device auto --device-num 1 \
      --max-time 1800 --timeout 0 \
      --run 'cd /opt/pypto && PTOAS_ROOT=/opt/pypto/.venv \
        .venv/bin/python -m pytest tests/st/runtime/cce_mlir/test_cce_mlir_compile_only.py \
        -q --platform a2a3 --device $TASK_DEVICE'
"""

import corpus  # noqa: E402
import pytest
from pypto.backend import BackendType  # noqa: E402
from pypto.backend._ptoas_locate import find_ptoas_binary  # noqa: E402
from pypto.pypto_core.passes import MemoryPlanner  # noqa: E402
from pypto.runtime import ExecutionMode, RunConfig  # noqa: E402

_CASES = [
    ("matmul_single", corpus.matmul_single),
    ("matmul_accum", corpus.matmul_accum),
    ("matmul_kloop", corpus.matmul_kloop),
    ("matmul_kloop_pipelined", corpus.matmul_kloop_pipelined),
    ("vector_add", corpus.vector_add),
    ("vector_scale_add", corpus.vector_scale_add),
    ("mm_then_scale", corpus.mm_then_scale),
]


def _compile(kernel, directory, kernel_compiler: str):
    return kernel.compile(
        config=RunConfig(
            kernel_compiler=kernel_compiler,
            memory_planner=MemoryPlanner.PYPTO,
            backend_type=BackendType.Ascend910,
            execution_mode=ExecutionMode.ONBOARD,
            codegen_only=True,
            save_kernels=True,
            save_kernels_dir=str(directory),
        )
    )


@pytest.mark.parametrize(("name", "kernel"), _CASES, ids=[name for name, _ in _CASES])
def test_ptoas_route_emits_pto(name, kernel, tmp_path):
    if find_ptoas_binary() is None:
        pytest.skip("ptoas is not installed")
    _compile(kernel, tmp_path, "ptoas")

    assert list(tmp_path.glob("ptoas/*.pto")), f"{name} produced no .pto"


@pytest.mark.parametrize(("name", "kernel"), _CASES, ids=[name for name, _ in _CASES])
def test_cce_mlir_route_emits_a_device_object(name, kernel, tmp_path):
    pytest.importorskip("cce_mlir")
    _compile(kernel, tmp_path, "cce_mlir")

    objects = list(tmp_path.glob("kernels/*/*.cce.o"))
    assert objects, f"{name} produced no .cce.o device object"
    config = (tmp_path / "kernel_config.py").read_text(encoding="utf-8")
    assert '"link_objects"' in config
    assert list(tmp_path.glob("cce_mlir/*.mlir"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
