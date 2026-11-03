# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Link the generated adapter with the device object under ``kernel_entry``.

Requires cce-mlir, CANN's device compiler and PyPTO's runtime headers. This is
the structural half of the runtime ABI adapter: execution is validated on the
NPU, not here.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("cce_mlir")

from pypto.backend.cce_mlir import CceMlirError, CceTarget, lower_program  # noqa: E402
from pypto.backend.cce_mlir.adapter import generate_adapter_source  # noqa: E402
from pypto.backend.cce_mlir.elf import entry_point, symbol_values  # noqa: E402
from pypto.backend.cce_mlir.emit import emit_module  # noqa: E402
from pypto.backend.cce_mlir.pipeline import (  # noqa: E402
    cann_toolkit,
    compile_device_object,
)

from . import kernels  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _toolchain():
    root = Path(os.environ.get("ASCEND_HOME_PATH") or "/usr/local/Ascend/cann-9.0.0/aarch64-linux")
    ccec = root / "bin" / "ccec"
    linker = root / "bin" / "ld.lld"
    if not ccec.is_file() or not linker.is_file() or shutil.which("readelf") is None:
        pytest.skip("CANN device compiler is unavailable")
    includes = [
        _REPO_ROOT / "runtime" / "src" / "a2a3" / "runtime" / "tensormap_and_ringbuffer" / "runtime",
        _REPO_ROOT / "runtime" / "src" / "common",
        _REPO_ROOT / "runtime" / "src" / "common" / "task_interface",
    ]
    if not all(path.is_dir() for path in includes):
        pytest.skip("PyPTO runtime headers are unavailable")
    return ccec, linker, includes


def _compile_flags(ccec, includes, output, source):
    return [
        str(ccec),
        "-c",
        "-O3",
        "-g",
        "-x",
        "cce",
        "-Wall",
        "-std=c++17",
        "--cce-aicore-only",
        "--cce-aicore-arch=dav-c220-cube",
        "-mllvm",
        "-cce-aicore-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-function-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-record-overflow=false",
        "-mllvm",
        "-cce-aicore-addr-transform",
        "-mllvm",
        "-cce-aicore-dcci-insert-for-scalar=false",
        "-DMEMORY_BASE",
        *(f"-I{path}" for path in includes),
        "-o",
        str(output),
        str(source),
    ]


def test_adapter_links_with_kernel_entry_first(tmp_path):
    try:
        toolkit = cann_toolkit()
    except CceMlirError as exc:
        pytest.skip(f"CANN is unavailable: {exc}")
    ccec, linker, includes = _toolchain()

    (lowered,) = lower_program(kernels.cube_matmul.lower(), target=CceTarget.A3)
    # cce-mlir's full pipeline needs the host driver to keep the device module
    # alive; the extracted object is independent of it.
    module = emit_module(lowered)
    device_object = compile_device_object(module, toolkit=toolkit, expected_symbol=lowered.device_symbol)
    device_path = tmp_path / "device.o"
    device_path.write_bytes(device_object)

    source_path = tmp_path / "adapter.cpp"
    source_path.write_text(generate_adapter_source(lowered), encoding="utf-8")
    adapter_path = tmp_path / "adapter.o"
    subprocess.run(
        _compile_flags(ccec, includes, adapter_path, source_path),
        check=True,
        capture_output=True,
        text=True,
    )

    relocatable = tmp_path / "combined.o"
    subprocess.run(
        [
            str(linker),
            "-m",
            "aicorelinux",
            "-r",
            "-Ttext=0",
            "-q",
            "-x",
            "-static",
            str(adapter_path),
            str(device_path),
            "-o",
            str(relocatable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = symbol_values(relocatable.read_bytes())
    assert "kernel_entry" in values
    assert values["kernel_entry"] == 0
    assert lowered.device_symbol in values

    final = tmp_path / "combined.elf"
    subprocess.run(
        [str(linker), "-e", "kernel_entry", "-o", str(final), str(adapter_path), str(device_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    final_values = symbol_values(final.read_bytes())
    assert entry_point(final.read_bytes()) == final_values["kernel_entry"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
