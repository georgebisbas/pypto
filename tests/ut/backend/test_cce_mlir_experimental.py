import json
from pathlib import Path

from examples.cce_mlir.compiler import CceMlirCompiler
from examples.cce_mlir.lowering import CceTarget, emit_cce_mlir, lower_pypto_matmul
from examples.cce_mlir.manifest import write_manifest

FIXTURE = Path(__file__).parents[3] / "examples" / "cce_mlir" / "fixtures" / "pypto_matmul.py"


def test_lowering_extracts_the_real_pypto_buffer_contract():
    spec = lower_pypto_matmul(FIXTURE, target=CceTarget.A3)

    assert spec.name == "matmul_64x128x64"
    assert spec.target is CceTarget.A3
    assert [(argument.name, argument.shape, argument.dtype) for argument in spec.arguments] == [
        ("a", (64, 128), "f16"),
        ("b", (128, 64), "f16"),
        ("c", (64, 64), "f32"),
    ]
    assert [operation.name for operation in spec.operations] == [
        "load",
        "load",
        "move",
        "move",
        "matmul",
        "store",
    ]


def test_emitter_produces_self_contained_cce_pto_module():
    mlir = emit_cce_mlir(lower_pypto_matmul(FIXTURE, target=CceTarget.A3))

    assert "npu.kernel_target<a3>" in mlir
    assert "npu.kernel_kind<cube>" in mlir
    assert "layout.pack_layout" in mlir
    assert "layout.repack_layout" in mlir
    assert "layout.unpack_layout" in mlir
    assert "pto.matmul" in mlir
    assert "npu.launch_func @pypto_matmul_mod::@matmul_64x128x64" in mlir


def test_compiler_reports_missing_tool_without_running_a_process(tmp_path):
    compiler = CceMlirCompiler(
        cce_opt="/does/not/exist/cce-opt",
        cann_root="/opt/cann",
        cann_arch="x86_64-linux",
    )

    result = compiler.compile(lower_pypto_matmul(FIXTURE, target=CceTarget.A3), tmp_path)

    assert result.status == "blocked_missing_cce_opt"
    assert result.input_path == tmp_path / "matmul_64x128x64.mlir"
    assert (tmp_path / "matmul_64x128x64.manifest.json").is_file()
    report = json.loads((tmp_path / "matmul_64x128x64.report.json").read_text())
    assert report["status"] == "blocked_missing_cce_opt"


def test_compiler_reports_missing_cann_configuration(tmp_path, monkeypatch):
    monkeypatch.delenv("CANN_ROOT", raising=False)
    monkeypatch.delenv("CANN_ARCH", raising=False)
    fake_cce_opt = tmp_path / "cce-opt"
    fake_cce_opt.write_text("#!/bin/sh\nexit 0\n")
    fake_cce_opt.chmod(0o755)
    compiler = CceMlirCompiler(cce_opt=str(fake_cce_opt))

    result = compiler.compile(lower_pypto_matmul(FIXTURE, target=CceTarget.A3), tmp_path)

    assert result.status == "blocked_missing_cann_configuration"
    report = json.loads((tmp_path / "matmul_64x128x64.report.json").read_text())
    assert report["status"] == "blocked_missing_cann_configuration"


def test_manifest_records_toolchain_and_acl_abi(tmp_path):
    spec = lower_pypto_matmul(FIXTURE, target=CceTarget.A3)
    artifact = tmp_path / "lowered.mlir"
    artifact.write_text(
        '@matmul_64x128x64_name("matmul_64x128x64__kernel0\\00")\n'
        '"aclrtBinaryLoadFromData\\00"\n'
        '"aclrtLaunchKernelWithConfig\\00"\n'
    )

    path = write_manifest(
        spec,
        artifact,
        tmp_path / "manifest.json",
        revisions={"pypto": "pypto-test", "cce_mlir": "cce-test", "cann": "cann-test"},
    )
    manifest = json.loads(path.read_text())

    assert manifest["revisions"]["pypto"] == "pypto-test"
    assert manifest["kernel"] == "matmul_64x128x64"
    assert manifest["abi"]["cce_device_symbols"] == ["matmul_64x128x64__kernel0"]
    assert manifest["abi"]["status"] == "adapter_required"
