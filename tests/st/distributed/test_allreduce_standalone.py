#!/usr/bin/env python3
"""Phase 2: standalone allreduce through the distributed compile path.

Validates that when a program declares a CommGroup (cross-rank communication),
the distributed codegen emits a built-in 4-phase symmetric allreduce chip task
with:
  - C++ PTO-ISA kernel (allreduce_kernel.cpp)
  - C++ orchestration shim (allreduce_kernel.cpp in orchestration/)
  - kernel_config.py for runtime assemble
  - CommGroup with scratch WindowBuffer in comm manifest

Does NOT require runtime — compile-time validation only.
"""

import json
import os
import sys

import pypto.language as pl
import pytest
from pypto import ir
from pypto.pypto_core import DataType, ir as _ir_core


# ---------------------------------------------------------------------------
# DSL program: minimal HOST orchestrator (needed to trigger distributed path)
# ---------------------------------------------------------------------------


@pl.program
class AllreduceProgram:
    """Minimal program that triggers the distributed compile path."""

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        x: pl.Tensor[[256], pl.FP32],
        y: pl.Out[pl.Tensor[[256], pl.FP32]],
    ):
        """HOST orchestrator — placeholder body (allreduce is a built-in)."""
        # In practice, the host orchestrator would submit allreduce chip tasks.
        # For now, the body is minimal to trigger distributed codegen.
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_file(files: dict[str, str], suffix: str) -> str | None:
    """Find a file in the result dict whose key ends with *suffix*."""
    for k in sorted(files):
        if k.endswith(suffix):
            return files[k]
    return None


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compiled_with_comm_group(tmp_path_factory):
    """Compile a program with a CommGroup to trigger allreduce emission."""
    from pypto.backend.pto_backend import generate
    from pypto.ir.pass_manager import PassManager, OptimizationStrategy

    output_dir = tmp_path_factory.mktemp("ar_dist")

    # Run passes manually (same as ir.compile() does)
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    transformed = pm.run_passes(AllreduceProgram, dump_ir=False)

    # Inject a CommGroup into the post-pass program.
    # comm_groups is read-only on existing programs, so we reconstruct.
    span = _ir_core.Span.unknown()
    size = _ir_core.ConstInt(1088, DataType.INT32, span)
    buf = _ir_core.WindowBuffer(
        _ir_core.Var("scratch", _ir_core.ScalarType(DataType.INT32), span), size
    )
    group = _ir_core.CommGroup(devices=[], slots=[buf], span=span)

    # Reconstruct program with CommGroup included
    from pypto.pypto_core.ir import Program as IRProgram
    funcs = list(transformed.functions.values())
    transformed_with_comm = IRProgram(funcs, [group], transformed.name, transformed.span)

    # Now run codegen
    files = generate(transformed_with_comm, str(output_dir), skip_ptoas=True)

    # Write files to disk for inspection
    for filepath, content in files.items():
        full = output_dir / filepath
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)

    return transformed_with_comm, files, output_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAllreduceCompileArtifacts:
    """Compile-time validation of allreduce chip task artifacts."""

    def test_allreduce_kernel_cpp_exists(self, compiled_with_comm_group):
        program, files, _ = compiled_with_comm_group
        kernel = _find_file(files, "allreduce_kernel.cpp")
        assert kernel is not None, (
            f"No allreduce_kernel.cpp in files. Keys: {sorted(files.keys())}"
        )

    def test_allreduce_kernel_has_four_phases(self, compiled_with_comm_group):
        program, files, _ = compiled_with_comm_group
        kernel = _find_file(files, "allreduce_kernel.cpp")
        assert kernel is not None

        assert "TNOTIFY" in kernel, "Missing TNOTIFY (cross-rank notify)"
        assert "TWAIT" in kernel, "Missing TWAIT (cross-rank wait)"
        assert "TADD" in kernel, "Missing TADD (reduce)"
        assert "TLOAD" in kernel, "Missing TLOAD"
        assert "TSTORE" in kernel, "Missing TSTORE"

    def test_allreduce_kernel_has_comm_remote_ptr(self, compiled_with_comm_group):
        program, files, _ = compiled_with_comm_group
        kernel = _find_file(files, "allreduce_kernel.cpp")
        assert kernel is not None
        assert "CommRemotePtr" in kernel, "Missing CommRemotePtr helper"

    def test_allreduce_orchestration_cpp_exists(self, compiled_with_comm_group):
        program, files, _ = compiled_with_comm_group
        orch = _find_file(files, "orchestration/allreduce_kernel.cpp")
        assert orch is not None, f"No orchestration shim. Keys: {sorted(files.keys())}"

    def test_allreduce_orchestration_correct(self, compiled_with_comm_group):
        program, files, _ = compiled_with_comm_group
        orch = _find_file(files, "orchestration/allreduce_kernel.cpp")
        assert orch is not None

        assert "allreduce_orchestration" in orch, "Missing entry function"
        assert ".expected_arg_count = 5" in orch, "Wrong arg count"
        assert "rt_submit_aiv_task" in orch, "Missing AIV task submission"
        assert "params.add_inout" in orch, "Missing INOUT arg for scratch"

    def test_kernel_config_py_exists(self, compiled_with_comm_group):
        program, files, _ = compiled_with_comm_group
        config = _find_file(files, "kernel_config.py")
        assert config is not None, f"No kernel_config.py. Keys: {sorted(files.keys())}"
        assert "allreduce_kernel" in config, "Missing allreduce_kernel in config"
        assert "tensormap_and_ringbuffer" in config, "Wrong runtime"

    def test_comm_manifest_has_scratch(self, compiled_with_comm_group):
        program, files, _ = compiled_with_comm_group
        groups = list(program.comm_groups)
        assert len(groups) > 0, "No CommGroup in program"

        group = groups[0]
        assert len(group.slots) > 0, "CommGroup has no slots"

        scratch_found = False
        for slot in group.slots:
            if hasattr(slot, "name_hint") and slot.name_hint == "scratch":
                scratch_found = True
                break
        assert scratch_found, "No scratch WindowBuffer in CommGroup"

    def test_allreduce_files_under_next_levels(self, compiled_with_comm_group):
        program, files, _ = compiled_with_comm_group
        ar_keys = [k for k in files if "allreduce" in k.lower()]
        assert len(ar_keys) >= 3, (
            f"Expected at least 3 allreduce files, got {len(ar_keys)}: {ar_keys}"
        )
        for k in ar_keys:
            assert "next_levels/allreduce_kernel" in k, (
                f"Allreduce file {k} not under next_levels/allreduce_kernel/"
            )


# ---------------------------------------------------------------------------
# Integration: allreduce + GEMM compile together
# ---------------------------------------------------------------------------


def test_allreduce_and_gemm_compile_together():
    """A program with GEMM kernel + CommGroup should emit both GEMM .pto and allreduce .cpp."""
    from pypto.backend.pto_backend import generate
    from pypto.ir.pass_manager import PassManager, OptimizationStrategy
    from pypto.pypto_core.ir import Program as IRProgram

    @pl.program
    class GemmWithAllreduce:
        @pl.function(type=pl.FunctionType.InCore)
        def gemm_kernel(self, a: pl.Tensor[[64,64],pl.FP32], b: pl.Tensor[[64,64],pl.FP32], c: pl.Out[pl.Tensor[[64,64],pl.FP32]]):
            with pl.incore():
                ta = pl.load(a, [0,0], [64,64], target_memory=pl.MemorySpace.Mat)
                tb = pl.load(b, [0,0], [64,64], target_memory=pl.MemorySpace.Mat)
                ta_l = pl.move(ta, target_memory=pl.MemorySpace.Left)
                tb_l = pl.move(tb, target_memory=pl.MemorySpace.Right)
                tc = pl.matmul(ta_l, tb_l)
                pl.store(tc, [0,0], c)

        @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
        def chip_orch(self, a: pl.Tensor[[64,64],pl.FP32], b: pl.Tensor[[64,64],pl.FP32], c: pl.Out[pl.Tensor[[64,64],pl.FP32]]):
            self.gemm_kernel(a, b, c)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self, a: pl.Tensor[[64,64],pl.FP32], b: pl.Tensor[[64,64],pl.FP32], c: pl.Out[pl.Tensor[[64,64],pl.FP32]]):
            self.chip_orch(a, b, c)

    import tempfile
    from pypto.pypto_core import backend as _backend_core
    _backend_core.set_backend_type(_backend_core.BackendType.Ascend910B)

    # Run passes manually
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    transformed = pm.run_passes(GemmWithAllreduce, dump_ir=False)

    # Inject CommGroup
    span = _ir_core.Span.unknown()
    size = _ir_core.ConstInt(1088, DataType.INT32, span)
    buf = _ir_core.WindowBuffer(
        _ir_core.Var("scratch", _ir_core.ScalarType(DataType.INT32), span), size
    )
    group = _ir_core.CommGroup(devices=[], slots=[buf], span=span)

    funcs = list(transformed.functions.values())
    transformed_with_comm = IRProgram(funcs, [group], transformed.name, transformed.span)

    with tempfile.TemporaryDirectory() as tmpdir:
        files = generate(transformed_with_comm, tmpdir, skip_ptoas=True)

        # Must have GEMM artifacts
        assert "orchestration/host_orch.py" in files
        gemm_keys = [k for k in files if "gemm_kernel" in k]
        assert len(gemm_keys) >= 1, f"No GEMM kernel files. Keys: {sorted(files.keys())}"

        # Must have allreduce artifacts
        ar_keys = [k for k in files if "allreduce" in k.lower()]
        assert len(ar_keys) >= 3, (
            f"Expected at least 3 allreduce files, got {len(ar_keys)}. "
            f"Keys: {sorted(files.keys())}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
