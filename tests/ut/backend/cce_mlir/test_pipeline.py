# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""In-process pipeline tests: require cce-mlir bindings and a CANN install."""

import pytest

pytest.importorskip("cce_mlir")

from pypto.backend.cce_mlir import CceMlirError, CceTarget, lower_program  # noqa: E402
from pypto.backend.cce_mlir.elf import is_relocatable, symbol_names  # noqa: E402
from pypto.backend.cce_mlir.emit import emit_module  # noqa: E402
from pypto.backend.cce_mlir.pipeline import (  # noqa: E402
    cann_toolkit,
    compile_device_object,
)

from . import kernels  # noqa: E402


def _toolkit():
    try:
        return cann_toolkit()
    except CceMlirError as exc:
        pytest.skip(f"CANN is unavailable: {exc}")


@pytest.mark.parametrize("kernel", [kernels.cube_matmul, kernels.vector_add])
def test_built_module_compiles_to_a_device_object(kernel):
    toolkit = _toolkit()
    (lowered,) = lower_program(kernel.lower(), target=CceTarget.A3)
    module = emit_module(lowered)

    data = compile_device_object(module, toolkit=toolkit, expected_symbol=lowered.device_symbol)

    assert data[:4] == b"\x7fELF"
    assert is_relocatable(data)
    assert lowered.device_symbol in symbol_names(data)


def test_missing_expected_symbol_is_refused():
    toolkit = _toolkit()
    (lowered,) = lower_program(kernels.cube_matmul.lower(), target=CceTarget.A3)
    module = emit_module(lowered)

    with pytest.raises(CceMlirError) as error:
        compile_device_object(module, toolkit=toolkit, expected_symbol="does_not_exist")

    assert error.value.reason == "missing-device-symbol"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
