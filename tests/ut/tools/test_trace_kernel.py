# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the kernel IR-descent tracer."""

from pathlib import Path

import pytest
from pypto.tools import trace_kernel

_KERNELS = Path(__file__).resolve().parents[1] / "backend" / "cce_mlir" / "kernels.py"


def test_parse_samples_builds_one_tensor_per_parameter():
    torch = pytest.importorskip("torch")

    tensors = trace_kernel.parse_samples("f16:128x512,f32:128x128")

    assert [tuple(t.shape) for t in tensors] == [(128, 512), (128, 128)]
    assert tensors[0].dtype == torch.float16
    assert tensors[1].dtype == torch.float32


def test_parse_samples_rejects_an_unknown_dtype():
    pytest.importorskip("torch")

    with pytest.raises(ValueError, match="invalid sample"):
        trace_kernel.parse_samples("f64:16x16")


def test_parse_samples_rejects_a_malformed_shape():
    pytest.importorskip("torch")

    with pytest.raises(ValueError, match="invalid sample shape"):
        trace_kernel.parse_samples("f32:16xz")


def test_load_kernel_needs_a_name_when_the_file_has_several():
    with pytest.raises(SystemExit, match="--function"):
        trace_kernel.load_kernel(_KERNELS, None)

    kernel = trace_kernel.load_kernel(_KERNELS, "cube_matmul")
    assert "cube_matmul" in kernel.__name__


def test_trace_writes_the_full_descent(tmp_path):
    pytest.importorskip("cce_mlir")
    from pypto.backend.cce_mlir import CceMlirError  # noqa: PLC0415
    from pypto.backend.cce_mlir.pipeline import cann_toolkit  # noqa: PLC0415

    try:
        cann_toolkit()
    except CceMlirError as exc:
        pytest.skip(f"CANN is unavailable: {exc}")

    output = trace_kernel.trace(
        _KERNELS,
        function="cube_matmul",
        samples=None,
        output=tmp_path,
        compiler="cce_mlir",
        planner=None,
    )

    dumps = [path.name for path in (output / "passes_dump").glob("*.py")]
    assert any(name.endswith("_after_SelectBufferIRFunctions.py") for name in dumps)
    assert any(name.endswith("_after_LowerTileToBuffer.py") for name in dumps)
    assert list((output / "cce_mlir").glob("*.mlir"))
    assert list((output / "cce_mlir").glob("*.lowered.mlir"))
    assert list((output / "kernels").glob("*/*.cce.o"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
