# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end standalone compilation of the cce-mlir fixture.

Skipped without the cce-mlir bindings and a CANN installation; the Docker image
is the authoritative environment.
"""

import json

import pytest

pytest.importorskip("cce_mlir")

from pypto.backend.cce_mlir import CceMlirError, CceTarget  # noqa: E402
from pypto.backend.cce_mlir.driver import CompileStatus, compile_program  # noqa: E402
from pypto.backend.cce_mlir.manifest import _revision  # noqa: E402
from pypto.backend.cce_mlir.pipeline import cann_toolkit  # noqa: E402


def _program():
    from examples.cce_mlir.__main__ import _DEFAULT_FIXTURE, _load_program

    return _load_program(_DEFAULT_FIXTURE)


def _toolkit():
    try:
        return cann_toolkit()
    except CceMlirError as exc:
        pytest.skip(f"CANN is unavailable: {exc}")


def test_fixture_compiles_to_a_relocatable_device_object(tmp_path):
    _toolkit()
    result = compile_program(_program(), tmp_path, target=CceTarget.A3)

    assert result.status is CompileStatus.LOWERED, result.diagnostics
    (artifact,) = result.artifacts
    assert artifact.mlir_path.is_file()
    assert artifact.lowered_path.is_file()
    assert artifact.object_path.read_bytes()[:4] == b"\x7fELF"

    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert manifest["kernel_kind"] == "cube"
    assert manifest["target"] == "a3"
    assert manifest["device_symbol"] in manifest["object_symbols"]
    assert manifest["object_sha256"]
    assert [parameter["direction"] for parameter in manifest["parameters"]] == [
        "In",
        "In",
        "Out",
    ]


def test_missing_git_metadata_does_not_emit_diagnostics(tmp_path):
    assert _revision(tmp_path) == "unknown"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
