# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Environment-based kernel-compiler selection for unmodified examples."""

import pytest
from pypto.backend.kernel_compiler import ENV_VAR, default_kernel_compiler
from pypto.runtime import CompileOptions, ExecutionMode, RunConfig


def test_default_route_is_ptoas(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)

    assert default_kernel_compiler() == "ptoas"


def test_environment_selects_cce_mlir(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "cce_mlir")

    assert default_kernel_compiler() == "cce_mlir"


def test_environment_value_is_normalized(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "  CCE_MLIR ")

    assert default_kernel_compiler() == "cce_mlir"


def test_unknown_environment_value_is_refused(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "nvcc")

    with pytest.raises(ValueError) as error:
        default_kernel_compiler()

    assert ENV_VAR in str(error.value)
    assert "ptoas" in str(error.value)


def test_run_config_resolves_the_environment_route(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "cce_mlir")

    config = RunConfig()

    assert config.kernel_compiler == "cce_mlir"
    assert config.execution_mode is ExecutionMode.ONBOARD
    assert config.compile_kwargs()["kernel_compiler"] == "cce_mlir"


def test_explicit_run_config_option_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "cce_mlir")

    config = RunConfig(kernel_compiler="ptoas")

    assert config.kernel_compiler == "ptoas"
    assert config.execution_mode is ExecutionMode.SIM


def test_explicit_simulator_with_cce_mlir_is_refused(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)

    with pytest.raises(ValueError) as error:
        RunConfig(kernel_compiler="cce_mlir", execution_mode=ExecutionMode.SIM)

    assert "onboard" in str(error.value)


def test_compile_options_resolve_the_environment_route(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "cce_mlir")

    options = CompileOptions()
    kwargs = options.as_compile_kwargs()

    assert options.kernel_compiler == "cce_mlir"
    assert options.execution_mode is ExecutionMode.ONBOARD
    assert kwargs["kernel_compiler"] == "cce_mlir"
    assert kwargs["execution_mode"] is ExecutionMode.ONBOARD


def test_default_compile_options_keep_the_simulator(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)

    options = CompileOptions()

    assert options.kernel_compiler == "ptoas"
    assert options.execution_mode is ExecutionMode.SIM
    assert options.platform == "a2a3sim"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
