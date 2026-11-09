# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for ``audit_stub_container_params`` and the contract it audits.

Two layers. The first runs the audit over the real stubs and asserts it is
clean, which guards every parameter the sweep widened and any new one at once.
The second calls a handful of bindings with a tuple where the stub names a
``Sequence``, so the runtime fact the widened annotations claim is pinned
independently of the audit's own correctness -- if the audit regressed into
always returning nothing, these would still fail.
"""

import sys
from pathlib import Path
from types import MappingProxyType

import pytest
from pypto import DataType, backend, ir, passes

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_stub_container_params import DEFAULT_MODULES, audit  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
STUB_DIR = REPO_ROOT / "python" / "pypto" / "pypto_core"


def make_span() -> ir.Span:
    return ir.Span("test.py", 1, 1, 1, 10)


def make_var(name: str = "x") -> ir.Var:
    return ir.Var(name, ir.ScalarType(DataType.INT32), make_span())


@pytest.mark.parametrize("module", DEFAULT_MODULES)
def test_no_stub_parameter_is_narrower_than_its_binding(module: str) -> None:
    """Every container parameter must name what the binding actually accepts.

    A stub saying ``list[T]`` where nanobind reports ``Sequence[T]`` makes
    pyright reject a tuple the runtime takes, and gives the author no hint that
    the stub rather than the call is wrong.
    """
    stub = STUB_DIR / f"{module}.pyi"
    if not stub.is_file():
        pytest.skip(f"{module}.pyi is not present")

    findings = audit(module, stub)

    assert findings == [], "\n".join(
        f"{stub.name}:{line}: {qual}({param}) is {declared}, binding takes {rendered}"
        for qual, param, declared, rendered, line in findings
    )


def test_audit_detects_a_narrow_annotation(tmp_path: Path) -> None:
    """The audit must fail on drift, or a clean run above would mean nothing."""
    stub = tmp_path / "ir.pyi"
    stub.write_text(
        "class SeqStmts:\n    def __init__(self, stmts: list[Stmt], span: Span) -> None: ...\n",
        encoding="utf-8",
    )

    findings = audit("ir", stub)

    assert [(q, p) for q, p, _, _, _ in findings] == [("SeqStmts.__init__", "stmts")]


class TestSequenceParametersAcceptTuples:
    """The runtime facts that make the widened annotations correct."""

    def test_seq_stmts_accepts_a_tuple(self) -> None:
        span = make_span()
        stmt = ir.ReturnStmt([make_var()], span)

        assert len(ir.SeqStmts((stmt,), span).stmts) == 1

    def test_return_and_yield_accept_a_tuple(self) -> None:
        span = make_span()
        var = make_var()

        assert len(ir.ReturnStmt((var,), span).value) == 1
        assert len(ir.YieldStmt((var,), span).value) == 1

    def test_program_accepts_a_tuple_of_functions(self) -> None:
        span = make_span()
        body = ir.SeqStmts([], span)
        func = ir.Function("f", [], [], body, span)

        assert len(ir.Program((func,), "prog", span).functions) == 1

    def test_pass_context_accepts_a_tuple_of_instruments(self) -> None:
        with passes.PassContext(()):
            pass

    def test_core_accepts_a_tuple_of_mems(self) -> None:
        mem = backend.Mem(ir.MemorySpace.Left, 256 * 1024, 64)

        assert len(backend.Core(ir.CoreType.CUBE, (mem,)).mems) == 1

    def test_cluster_accepts_a_non_dict_mapping_by_keyword(self) -> None:
        mem = backend.Mem(ir.MemorySpace.Left, 256 * 1024, 64)
        core = backend.Core(ir.CoreType.CUBE, [mem])

        cluster = backend.Cluster(core_counts=MappingProxyType({core: 2}))

        assert cluster.total_core_count() == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
