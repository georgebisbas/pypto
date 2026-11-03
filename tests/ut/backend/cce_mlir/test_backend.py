# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime artifact assembly for the cce-mlir route."""

import pytest

pytest.importorskip("cce_mlir")

from pypto.backend.cce_mlir import CceMlirError, CceTarget  # noqa: E402
from pypto.backend.cce_mlir.backend import CceMlirBackend  # noqa: E402
from pypto.backend.cce_mlir.pipeline import cann_toolkit  # noqa: E402

from . import kernels  # noqa: E402


def _generate(kernel):
    try:
        cann_toolkit()
    except CceMlirError as exc:
        pytest.skip(f"CANN is unavailable: {exc}")
    program = kernel.lower()
    return CceMlirBackend().generate(program, "/tmp/unused", target=CceTarget.A3)


def test_artifact_map_contains_sources_objects_and_config():
    files = _generate(kernels.cube_matmul)

    sources = [path for path in files if path.startswith("kernels/aic/") and path.endswith(".cpp")]
    objects = [path for path in files if path.startswith("kernels/aic/") and path.endswith(".cce.o")]
    assert len(sources) == 1
    assert len(objects) == 1
    assert files[objects[0]][:4] == b"\x7fELF"
    assert b"kernel_entry" in files[sources[0]].encode()

    config = files["kernel_config.py"]
    assert isinstance(config, str)
    assert '"link_objects"' in config
    assert objects[0] in config
    assert any(path.startswith("orchestration/") for path in files)


def test_ptoas_options_are_ignored():
    program = kernels.cube_matmul.lower()
    files = CceMlirBackend().generate(program, "/tmp/unused", True, target=CceTarget.A3)

    assert "kernel_config.py" in files


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
