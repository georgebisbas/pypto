# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""kernel_entry adapter generation tests (no cce-mlir dependency)."""

from pypto.backend.cce_mlir import CceTarget, lower_program
from pypto.backend.cce_mlir.adapter import generate_adapter_source

from . import kernels


def _adapter(kernel):
    (lowered,) = lower_program(kernel.lower(), target=CceTarget.A3)
    return lowered, generate_adapter_source(lowered)


def test_adapter_calls_the_derived_function_symbol_once():
    lowered, source = _adapter(kernels.cube_matmul)

    assert source.count(f'extern "C" __aicore__ void {lowered.symbol_name}(') == 1
    assert source.count('extern "C" __aicore__ void kernel_entry(') == 1
    assert source.count(f"    {lowered.symbol_name}(") == 1


def test_adapter_unpacks_one_pointer_per_parameter_in_order():
    lowered, source = _adapter(kernels.cube_matmul)

    for index, parameter in enumerate(lowered.parameters):
        assert f"args[{index}]" in source
        assert parameter.element.c_type in source
    call = source.split(f"    {lowered.symbol_name}(")[1].split(");")[0]
    assert len(call.split(",")) == len(lowered.parameters)


def test_adapter_does_not_carry_fixture_literals():
    _, source = _adapter(kernels.cube_matmul)

    for literal in ("16384", "pypto_matmul_mod", "[64", "[128"):
        assert literal not in source


def test_different_kernels_produce_different_adapters():
    _, cube = _adapter(kernels.cube_matmul)
    _, vector = _adapter(kernels.vector_add)

    assert cube != vector
    assert "half" in cube
    assert "float" in vector
