# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the HTML IR-descent renderer."""

from pathlib import Path

import pytest
from pypto.tools import trace_html


def _write_output(root):
    dumps = root / "passes_dump"
    dumps.mkdir()
    (dumps / "00_frontend.py").write_text("a\nb\nc\n", encoding="utf-8")
    (dumps / "01_after_First.py").write_text("a\nB\nc\n", encoding="utf-8")
    # Identical to the previous snapshot: must render as unchanged.
    (dumps / "02_after_Second.py").write_text("a\nB\nc\n", encoding="utf-8")
    # cce-opt writes one file per pass under a module subdirectory.
    passes = root / "cce_passes" / "builtin_module_no-symbol-name"
    passes.mkdir(parents=True)
    (passes / "0_alpha-pass.mlir").write_text("%0 = <x>\n", encoding="utf-8")
    (passes / "1_beta-pass.mlir").write_text("%0 = y\n", encoding="utf-8")
    (root / "cce_mlir").mkdir()
    (root / "cce_mlir" / "k.mlir").write_text("module {}\n", encoding="utf-8")
    (root / "cce_mlir" / "k.lowered.mlir").write_text("llvm.func @k()\n", encoding="utf-8")


def test_collect_stages_reads_per_pass_files_in_numeric_order(tmp_path):
    _write_output(tmp_path)

    stages = trace_html.collect_stages(tmp_path, "cce_mlir")

    assert [stage.name for stage in stages if stage.group == "cce-mlir pipeline"] == [
        "alpha-pass",
        "beta-pass",
    ]


def test_pass_files_sort_numerically_not_lexicographically():
    paths = [Path("10_ten.mlir"), Path("9_nine.mlir")]

    assert [path.name for path in sorted(paths, key=trace_html._pass_file_key)] == [
        "9_nine.mlir",
        "10_ten.mlir",
    ]


def test_collect_stages_groups_representations_in_pipeline_order(tmp_path):
    _write_output(tmp_path)

    stages = trace_html.collect_stages(tmp_path, "cce_mlir")

    assert [stage.group for stage in stages] == [
        "PyPTO IR",
        "PyPTO IR",
        "PyPTO IR",
        "PTO MLIR",
        "cce-mlir pipeline",
        "cce-mlir pipeline",
        "Device LLVM",
    ]
    assert stages[0].name == "frontend"
    assert stages[-1].representation == "LLVM dialect"


def test_render_marks_changed_and_unchanged_stages_and_escapes_code(tmp_path):
    _write_output(tmp_path)
    stages = trace_html.collect_stages(tmp_path, "cce_mlir")

    page = trace_html.render_trace_html(stages, "demo IR descent", "subtitle")

    assert page.startswith("<title>demo IR descent</title>")
    assert "<html" not in page and "<body" not in page
    assert "alpha-pass" in page
    assert "beta-pass" in page
    # `%0 = <x>` must not escape as a tag.
    assert "%0 = &lt;x&gt;" in page
    assert 'class="add-c"' in page
    assert "unchanged</span>" in page


def test_diff_collapses_long_unchanged_runs():
    before = "\n".join(f"line {index}" for index in range(40))
    after = before.replace("line 20", "line twenty")

    rows, inserted, deleted = trace_html._diff_rows(before, after)

    assert (inserted, deleted) == (1, 1)
    assert any(kind == "skip" for kind, *_ in rows)
    # Two changed runs, each keeping at most 2 * context lines around the change.
    assert sum(1 for kind, *_ in rows if kind == "ctx") <= 4 * trace_html._CONTEXT_LINES


def test_write_trace_html_writes_beside_the_artifacts(tmp_path):
    _write_output(tmp_path)

    page = trace_html.write_trace_html(tmp_path, "cce_mlir", "matmul_kloop")

    assert page == tmp_path / "trace.html"
    assert "matmul_kloop IR descent" in page.read_text(encoding="utf-8")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
