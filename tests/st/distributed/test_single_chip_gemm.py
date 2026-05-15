#!/usr/bin/env python3
"""Phase 1.5: runtime-ready validation of single-chip GEMM through distributed path.

Validates:
  1. DSL → IR → passes → distributed codegen → .pto MLIR → orchestration .cpp
  2. MLIR content correctness (contains matmul, load, store)
  3. Orchestration .cpp correctness (correct arg count, submits AIV task)
  4. host_orch.py correctness (submit_next_level, TaskArgs)
  5. skip_ptoas=False attempt (validated up to ptoas invocation point)

Prerequisites for full runtime (Phase 1.5-rt):
  - ptoas binary available at PTOAS_ROOT or in PATH
  - simpler Worker(level=3) working with a2a3sim simulator
"""

import os
import sys

import pypto.language as pl
import pytest
from pypto import ir


# ---------------------------------------------------------------------------
# DSL program: HOST-orch → CHIP-orch → InCore GEMM
# ---------------------------------------------------------------------------


@pl.program
class SingleChipGemm:
    """HOST orchestrator submits a 64×64 GEMM to one chip."""

    @pl.function(type=pl.FunctionType.InCore)
    def gemm_kernel(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ):
        """64×64 matmul on the cube unit."""
        with pl.incore():
            ta = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
            tb = pl.load(b, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
            ta_l = pl.move(ta, target_memory=pl.MemorySpace.Left)
            tb_l = pl.move(tb, target_memory=pl.MemorySpace.Right)
            tc = pl.matmul(ta_l, tb_l)
            pl.store(tc, [0, 0], c)

    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def chip_orch(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ):
        """CHIP orchestrator dispatches the GEMM kernel."""
        self.gemm_kernel(a, b, c)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ):
        """HOST orchestrator submits work to the chip."""
        self.chip_orch(a, b, c)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ptoas_available() -> bool:
    """Check if ptoas binary can be executed."""
    import shutil
    ptoas_root = os.environ.get("PTOAS_ROOT")
    if ptoas_root:
        ptoas_bin = os.path.join(ptoas_root, "ptoas")
        if os.path.isfile(ptoas_bin) and os.access(ptoas_bin, os.X_OK):
            return True
    return shutil.which("ptoas") is not None


def _simpler_worker_available() -> bool:
    try:
        from simpler.worker import Worker  # noqa: F401
        return True
    except ImportError:
        return False


def _simulator_available() -> bool:
    try:
        from simpler_setup.platform_info import available_platforms
        return "a2a3sim" in available_platforms()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    """Compile with skip_ptoas=True for artifact inspection."""
    output_dir = tmp_path_factory.mktemp("gemm_dist")
    return ir.compile(
        SingleChipGemm,
        output_dir=str(output_dir),
        skip_ptoas=True,
        dump_passes=False,
    )


# ---------------------------------------------------------------------------
# Compile-time structural tests
# ---------------------------------------------------------------------------


class TestCompileDistributedGemm:
    """Compile-time validation of distributed GEMM artifacts."""

    def test_returns_distributed_compiled_program(self, compiled):
        from pypto.ir.distributed_compiled_program import DistributedCompiledProgram
        assert isinstance(compiled, DistributedCompiledProgram)

    def test_post_pass_function_types(self, compiled):
        funcs = compiled.program.functions
        func_by_name = {f.name: f for f in funcs.values()}

        assert "gemm_kernel" in func_by_name
        gemm = func_by_name["gemm_kernel"]
        assert str(gemm.func_type) == "FunctionType.AIV", f"Expected AIV, got {gemm.func_type}"

        assert "chip_orch" in func_by_name
        chip = func_by_name["chip_orch"]
        assert str(chip.role) == "Role.Orchestrator"
        assert str(chip.level) == "Level.CHIP"

        assert "host_orch" in func_by_name
        host = func_by_name["host_orch"]
        assert str(host.role) == "Role.Orchestrator"
        assert str(host.level) == "Level.HOST"

    def test_host_orch_py_content(self, compiled):
        orch_path = compiled.output_dir / "orchestration" / "host_orch.py"
        assert orch_path.exists()

        content = orch_path.read_text()
        assert "submit_next_level" in content
        assert "TaskArgs" in content
        assert "callables" in content
        assert "TensorArgType" in content
        assert 'callables["chip_orch"]' in content

    def test_chip_orch_cpp_content(self, compiled):
        orch_cpp = (
            compiled.output_dir
            / "next_levels" / "chip_orch" / "orchestration" / "chip_orch.cpp"
        )
        assert orch_cpp.exists()

        content = orch_cpp.read_text()
        assert "aicpu_orchestration_entry" in content
        assert "gemm_kernel" in content
        assert "rt_submit_aiv_task" in content
        assert ".expected_arg_count = 3" in content

    def test_gemm_kernel_pto_content(self, compiled):
        kernel_pto = (
            compiled.output_dir
            / "next_levels" / "chip_orch" / "kernels" / "aiv" / "gemm_kernel.pto"
        )
        assert kernel_pto.exists()

        content = kernel_pto.read_text()
        assert "matmul" in content.lower(), (
            f"MLIR missing matmul. First 500 chars:\n{content[:500]}"
        )
        assert "func.func" in content

    def test_gemm_kernel_pto_mlir_structure(self, compiled):
        kernel_pto = (
            compiled.output_dir
            / "next_levels" / "chip_orch" / "kernels" / "aiv" / "gemm_kernel.pto"
        )
        content = kernel_pto.read_text()

        assert "module" in content
        assert content.count("func.func") >= 1
        assert "return" in content

        # PTO MLIR dialect ops for the GEMM pipeline
        assert "pto.tmatmul" in content, "Missing pto.tmatmul (cube matmul)"
        assert "pto.tload" in content, "Missing pto.tload (GM→L1 load)"
        assert "pto.tstore" in content, "Missing pto.tstore (L0C→GM store)"
        assert "pto.tmov" in content, "Missing pto.tmov (L1→L0A/L0B moves)"
        assert "pto.alloc_tile" in content, "Missing pto.alloc_tile"
        assert "pto.make_tensor_view" in content, "Missing pto.make_tensor_view"

        # Correct memory spaces
        assert 'loc=mat' in content, "Missing Mat (L1) buffer allocation"
        assert 'loc=left' in content, "Missing Left (L0A) buffer allocation"
        assert 'loc=right' in content, "Missing Right (L0B) buffer allocation"
        assert 'loc=acc' in content, "Missing Acc (L0C) buffer allocation"

        # Target architecture
        assert 'pto.target_arch = "a2a3"' in content

    def test_output_directory_structure(self, compiled):
        output_dir = compiled.output_dir
        assert (output_dir / "orchestration").is_dir()
        assert (output_dir / "next_levels").is_dir()
        assert (output_dir / "orchestration" / "host_orch.py").is_file()
        chip_dir = output_dir / "next_levels" / "chip_orch"
        assert chip_dir.is_dir()
        assert (chip_dir / "orchestration" / "chip_orch.cpp").is_file()
        assert (chip_dir / "kernels" / "aiv" / "gemm_kernel.pto").is_file()

    def test_skip_ptoas_false_attempt(self, tmp_path):
        """skip_ptoas=False triggers ptoas compilation.

        If ptoas unavailable: PartialCodegenError with partial files.
        If ptoas available: full compilation succeeds with kernel_config.py.
        """
        from pypto.backend.pto_backend import PartialCodegenError

        output_dir = tmp_path / "gemm_dist_full"
        if _ptoas_available():
            compiled = ir.compile(
                SingleChipGemm,
                output_dir=str(output_dir),
                skip_ptoas=False,
                dump_passes=False,
            )
            assert compiled is not None
            config_path = output_dir / "next_levels" / "chip_orch" / "kernel_config.py"
            assert config_path.exists(), "kernel_config.py missing"
        else:
            with pytest.raises(PartialCodegenError) as exc_info:
                ir.compile(
                    SingleChipGemm,
                    output_dir=str(output_dir),
                    skip_ptoas=False,
                    dump_passes=False,
                )
            assert exc_info.value.files
            has_orch = any("orchestration" in k for k in exc_info.value.files)
            assert has_orch, "Partial files should include orchestration"


# ---------------------------------------------------------------------------
# Runtime readiness documentation test
# ---------------------------------------------------------------------------


def test_runtime_prerequisites():
    """Document runtime prerequisites. Always passes; diagnostic only."""
    prereqs = {
        "ptoas binary": _ptoas_available(),
        "simpler Worker(level=3)": _simpler_worker_available(),
        "simulator (a2a3sim)": _simulator_available(),
    }
    missing = [k for k, v in prereqs.items() if not v]
    if missing:
        pytest.skip(
            f"Runtime prerequisites not met: {', '.join(missing)}. "
            "Set PTOAS_ROOT, install simpler, ensure a2a3sim platform."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
