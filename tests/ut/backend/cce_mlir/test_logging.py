# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""DEBUG artifact-path logging for the cce-mlir route."""

import os
from pathlib import Path

import pypto
import pytest
from pypto.backend.cce_mlir.backend import log_written_artifacts


@pytest.fixture
def log_level():
    original = pypto.get_log_level()
    yield
    pypto.set_log_level(original)


def _object(tmp_path: Path) -> Path:
    path = tmp_path / "kernels" / "aic" / "kernel.cce.o"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\x7fELF")
    return path


def test_debug_names_the_resolved_artifact_root_and_object(capfd, tmp_path, log_level):
    obj = _object(tmp_path)
    pypto.set_log_level(pypto.LogLevel.DEBUG)

    log_written_artifacts(tmp_path, {"kernels/aic/kernel.cce.o": b"\x7fELF"})

    err = capfd.readouterr().err
    assert "cce-mlir artifacts written to" in err
    assert os.path.realpath(tmp_path) in err
    assert os.path.realpath(obj) in err


def test_info_does_not_log_the_artifact_path(capfd, tmp_path, log_level):
    _object(tmp_path)
    pypto.set_log_level(pypto.LogLevel.INFO)

    log_written_artifacts(tmp_path, {"kernels/aic/kernel.cce.o": b"\x7fELF"})

    assert "cce-mlir artifacts written to" not in capfd.readouterr().err


def test_relative_output_dir_is_logged_absolute(capfd, tmp_path, monkeypatch, log_level):
    _object(tmp_path)
    monkeypatch.chdir(tmp_path)
    pypto.set_log_level(pypto.LogLevel.DEBUG)

    log_written_artifacts("relative_out", {"kernels/aic/kernel.cce.o": b"\x7fELF"})

    err = capfd.readouterr().err
    assert os.path.realpath(tmp_path / "relative_out") in err


def test_symlink_component_is_resolved(capfd, tmp_path, monkeypatch, log_level):
    (tmp_path / "target" / "sub").mkdir(parents=True)
    (tmp_path / "link").symlink_to(tmp_path / "target" / "sub")
    _object(tmp_path / "target")
    monkeypatch.chdir(tmp_path)
    pypto.set_log_level(pypto.LogLevel.DEBUG)

    log_written_artifacts("link/../out", {"kernels/aic/kernel.cce.o": b"\x7fELF"})

    err = capfd.readouterr().err
    assert os.path.realpath(tmp_path / "target" / "out") in err


def test_non_cce_artifacts_are_silent(capfd, tmp_path, log_level):
    pypto.set_log_level(pypto.LogLevel.DEBUG)

    log_written_artifacts(tmp_path, {"kernels/aic/kernel.cpp": "source"})

    assert "cce-mlir artifacts written to" not in capfd.readouterr().err


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
