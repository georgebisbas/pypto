"""Unit tests for distributed target in ir.compile()."""

from pathlib import Path

import pypto.language as pl
import pytest
from pypto import ir
from pypto.ir import OptimizationStrategy


def test_compile_distributed_emits_deterministic_artifact(tmp_path: Path):
    """Distributed compile target writes the expected artifact path."""

    @pl.program
    class Input:
        @pl.function(level=pl.Level.POD, role=pl.Role.Orchestrator)
        def pod_orch(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(level=pl.Level.HOST, role=pl.Role.Worker):
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
            return y

    output_dir = tmp_path / "dist_build"
    result_dir = ir.compile(
        Input,
        output_dir=str(output_dir),
        strategy=OptimizationStrategy.Default,
        dump_passes=False,
        compile_target="distributed",
    )

    assert result_dir == str(output_dir)
    artifact = output_dir / "distributed" / "distributed_main.cpp"
    assert artifact.exists()
    code = artifact.read_text(encoding="utf-8")
    assert '#include "runtime/level_runtime.h"' in code
    assert "submit_worker" in code


def test_compile_distributed_rejects_program_without_distributed_metadata(tmp_path: Path):
    """Distributed compile target fails when no level/role metadata exists."""

    @pl.program
    class NoDistributed:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
            return y

    with pytest.raises(ValueError, match="at least one function with distributed metadata"):
        ir.compile(
            NoDistributed,
            output_dir=str(tmp_path / "dist_invalid_none"),
            strategy=OptimizationStrategy.Default,
            dump_passes=False,
            compile_target="distributed",
        )


def test_compile_distributed_rejects_missing_role_or_level(tmp_path: Path):
    """Distributed compile target requires both role and level metadata."""

    @pl.program
    class MissingRole:
        @pl.function(level=pl.Level.HOST)
        def host_func(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
            return y

    with pytest.raises(ValueError, match="must define both level and role metadata"):
        ir.compile(
            MissingRole,
            output_dir=str(tmp_path / "dist_invalid_partial"),
            strategy=OptimizationStrategy.Default,
            dump_passes=False,
            compile_target="distributed",
        )


def test_compile_rejects_unknown_compile_target(tmp_path: Path):
    """Compile target must be one of the supported values."""

    @pl.program
    class Input:
        @pl.function
        def main(self, x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
            return x

    with pytest.raises(ValueError, match="Unsupported compile_target"):
        ir.compile(
            Input,
            output_dir=str(tmp_path / "dist_invalid_target"),
            compile_target="not-a-target",
        )
