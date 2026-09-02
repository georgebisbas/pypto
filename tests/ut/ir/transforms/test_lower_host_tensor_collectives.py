# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821

"""Tests for ``LowerHostTensorCollectives``.

The lowering emits ``builtin.tensor.*`` internal ops, which the printer renders
as ``pl.builtin.tensor.*`` — a machine-only surface the parser reads back, so
the lowered dispatch survives print -> parse. A whole-``@pl.program``
``Expected`` still cannot be parsed for these tests, but for an upstream
reason: ``MaterializeCommDomainScopes`` must run first, and neither the
``CommDomainScopeStmt`` it synthesizes nor the ``WindowBuffer`` back-references
it stamps on ``DistributedTensorType`` has a DSL surface.

So, as in the materialize_comm_domain_scopes module, the structural
Before/Expected pattern is applied at the granularity of the pass's comparable
output product — the emitted builtin dispatch — via
:func:`_assert_builtin_dispatch`, which pins the full dispatch (world-size
loop, every window-bound arg in order, arg directions, and the complete
kwarg/attr dicts) both in the pass output and in its re-parse.
"""

from typing import Any, cast

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto.language.parser.diagnostics import InvalidOperationError
from pypto.pypto_core import backend as _backend
from pypto.pypto_core import ir, passes

# Builtin names are validated at import via the registry getter (raises on a
# typo) and matched by name at runtime — see operator-identity-checks.md.
_BUILTIN_BARRIER = ir.get_op("builtin.tensor.barrier").name
_BUILTIN_BROADCAST = ir.get_op("builtin.tensor.broadcast").name
_BUILTIN_REDUCE_SCATTER = ir.get_op("builtin.tensor.reduce_scatter").name


@pytest.fixture(autouse=True)
def _basic_verification_context():
    """Property verification only — the conftest default adds the roundtrip
    instrument, which these programs cannot satisfy: they run
    ``MaterializeCommDomainScopes``, whose ``CommDomainScopeStmt`` and
    ``WindowBuffer`` back-references have no DSL surface and so fail
    whole-program structural equality after a re-parse. The builtin dispatch
    itself does round-trip; ``_assert_builtin_dispatch`` checks that directly.
    """
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        yield


def _get_func(program: ir.Program, name: str) -> ir.Function:
    gvar = program.get_global_var(name)
    assert gvar is not None
    return program.functions[gvar]


def _as_call(expr: ir.Expr) -> ir.Call:
    assert isinstance(expr, ir.Call)
    return expr


def _as_var(expr: ir.Expr) -> ir.Var:
    assert isinstance(expr, ir.Var)
    return expr


def _collect_for_stmts(stmt: ir.Stmt) -> list[ir.ForStmt]:
    found: list[ir.ForStmt] = []

    def walk(s: ir.Stmt) -> None:
        if isinstance(s, ir.ForStmt):
            found.append(s)
            walk(s.body)
        if isinstance(s, ir.SeqStmts):
            for child in s.stmts:
                walk(child)
        if isinstance(s, ir.ScopeStmt):
            walk(s.body)

    walk(stmt)
    return found


def _collect_assign_stmts(stmt: ir.Stmt) -> list[ir.AssignStmt]:
    found: list[ir.AssignStmt] = []

    def walk(s: ir.Stmt) -> None:
        if isinstance(s, ir.AssignStmt):
            found.append(s)
        if isinstance(s, ir.SeqStmts):
            for child in s.stmts:
                walk(child)
        if isinstance(s, ir.ScopeStmt):
            walk(s.body)

    walk(stmt)
    return found


def _collect_return_stmts(stmt: ir.Stmt) -> list[ir.ReturnStmt]:
    found: list[ir.ReturnStmt] = []

    def walk(s: ir.Stmt) -> None:
        if isinstance(s, ir.ReturnStmt):
            found.append(s)
        if isinstance(s, ir.SeqStmts):
            for child in s.stmts:
                walk(child)
        if isinstance(s, ir.ScopeStmt):
            walk(s.body)

    walk(stmt)
    return found


def _last_stmt(stmt: ir.Stmt) -> ir.Stmt:
    if isinstance(stmt, ir.ScopeStmt):
        return _last_stmt(stmt.body)
    if isinstance(stmt, ir.SeqStmts):
        assert len(stmt.stmts) > 0
        return _last_stmt(stmt.stmts[-1])
    return stmt


def _assert_alias_keeps_window_buffer(alias: ir.AssignStmt) -> None:
    lhs_type = alias.var.type
    rhs = alias.value
    assert isinstance(rhs, ir.Var)
    rhs_type = rhs.type
    assert isinstance(lhs_type, ir.DistributedTensorType)
    assert isinstance(rhs_type, ir.DistributedTensorType)
    assert lhs_type.window_buffer is not None
    assert lhs_type.window_buffer is rhs_type.window_buffer


def _assert_builtin_dispatch(
    result: ir.Program,
    builtin_name: str,
    *,
    arg_names: list[str],
    arg_directions: list[ir.ArgDirection],
    kwargs: dict[str, object],
    attrs: dict[str, object] | None = None,
) -> ir.Call:
    """Pin the lowered ``builtin_name`` dispatch, in the pass output *and* in the
    print -> parse round-trip of that output.

    The printer renders the lowered internal op as ``pl.builtin.tensor.<op>`` and
    the parser reads that machine-only surface back, so re-parsing the printed
    program must rebuild the same dispatch. Matching both sides against one set
    of expectations keeps the two halves from drifting.

    Whole-program ``assert_structural_equal`` cannot be used for that half:
    ``MaterializeCommDomainScopes`` (which must run first) synthesizes
    ``CommDomainScopeStmt`` and the ``WindowBuffer`` back-references on
    ``DistributedTensorType``, and neither has a DSL surface — the scope prints
    as a leading comment and the back-reference is not printed at all. Hence
    ``window_bound_args=False`` on the re-parsed side; everything the lowering
    itself produces is still matched exactly.

    Returns the dispatch call from the lowered (non-re-parsed) program.
    """
    expectations: dict[str, Any] = {
        "arg_names": arg_names,
        "arg_directions": arg_directions,
        "kwargs": kwargs,
        "attrs": attrs,
    }
    call = _match_builtin_dispatch(_get_func(result, "host_orch").body, builtin_name, **expectations)
    reparsed = pl.parse_program(ir.python_print(result, format=False))
    assert isinstance(reparsed, ir.Program)
    _match_builtin_dispatch(
        _get_func(reparsed, "host_orch").body, builtin_name, window_bound_args=False, **expectations
    )
    return call


def _match_builtin_dispatch(
    host_body: ir.Stmt,
    builtin_name: str,
    *,
    arg_names: list[str],
    arg_directions: list[ir.ArgDirection],
    kwargs: dict[str, object],
    attrs: dict[str, object] | None = None,
    window_bound_args: bool = True,
) -> ir.Call:
    """Assert the host body dispatches ``builtin_name`` exactly once and pin the
    full emitted structure of the dispatch.

    This is the structural-comparison equivalent of the Before/Expected pattern
    for ``LowerHostTensorCollectives``, applied — as in the
    materialize_comm_domain_scopes module — at the granularity of the pass's
    comparable output product: the builtin dispatch. The helper pins:

    * exactly one ``for r in pl.range(pld.system.world_size())`` loop whose body
      is exactly the builtin EvalStmt (extra or missing surrounding statements
      fail the match),
    * every argument as the expected window-bound DistributedTensor, in order
      (a reordered arg or a wrong window view for an arg fails),
    * the exact arg directions, and
    * the complete kwarg dict and attr set, with ``device`` bound to the loop
      var.
    """

    op_name = ir.get_op(builtin_name).name
    loops = _collect_for_stmts(host_body)
    dispatch = [
        loop
        for loop in loops
        if isinstance(loop.body, ir.EvalStmt)
        and isinstance(loop.body.expr, ir.Call)
        and loop.body.expr.op.name == op_name
    ]
    assert len(dispatch) == 1, f"expected exactly one {builtin_name} dispatch loop, found {len(dispatch)}"
    loop = dispatch[0]

    # The dispatch loop iterates exactly the distributed world size.
    world_size_name = ir.get_op("pld.system.world_size").name
    stop = loop.stop
    if isinstance(stop, ir.Var):
        # CSE / NormalizeStmtStructure can hoist the bound to a temp
        # ``t = pld.system.world_size(); for r in pl.range(t):``.
        stop = next(
            (assign.value for assign in _collect_assign_stmts(host_body) if assign.var is stop),
            stop,
        )
    assert isinstance(stop, ir.Call) and stop.op.name == world_size_name, (
        "dispatch loop bound must be pld.system.world_size()"
    )

    body = loop.body
    assert isinstance(body, ir.EvalStmt)
    call = _as_call(body.expr)

    # device attr is bound to the loop induction var
    assert call.attrs["device"] is loop.loop_var

    # every argument is the expected window-bound DistributedTensor, in order
    assert len(call.args) == len(arg_names), f"expected {len(arg_names)} args, got {len(call.args)}"
    for actual, expected in zip(call.args, arg_names):
        var = _as_var(actual)
        assert var.name_hint == expected, f"expected arg window var {expected!r}, got {var.name_hint!r}"
        assert isinstance(var.type, ir.DistributedTensorType)
        if window_bound_args:
            assert var.type.window_buffer is not None, f"arg {expected!r} must be window-bound"

    assert list(call.arg_directions) == arg_directions
    # arg_directions is mirrored in attrs
    assert list(call.attrs["arg_directions"]) == arg_directions

    # complete kwarg dict
    assert dict(call.kwargs) == kwargs, f"kwargs mismatch: {dict(call.kwargs)} != {kwargs}"

    # exact attr key set plus op-specific values (device / arg_directions above)
    expected_attr_keys = {"device", "arg_directions", *(attrs or {})}
    assert set(call.attrs.keys()) == expected_attr_keys, (
        f"attr key mismatch: {set(call.attrs.keys())} != {expected_attr_keys}"
    )
    for key, value in (attrs or {}).items():
        assert call.attrs[key] == value, f"attr {key!r} mismatch: {call.attrs[key]!r} != {value!r}"
    return call


def test_host_allreduce_lowers_to_builtin_world_size_loop():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return 0

    program = cast(ir.Program, passes.materialize_comm_domain_scopes()(P))
    result = cast(ir.Program, passes.lower_host_tensor_collectives()(program))

    _assert_builtin_dispatch(
        result,
        "builtin.tensor.allreduce",
        arg_names=["data", "signal"],
        arg_directions=[ir.ArgDirection.InOut, ir.ArgDirection.InOut],
        kwargs={"op": int(pld.ReduceOp.Sum), "dtype": pl.FP32, "core_num": 1},
        attrs={"op": int(pld.ReduceOp.Sum), "dtype": pl.FP32, "core_num": 1},
    )


def test_implicit_host_allreduce_synthesizes_signal_then_lowers():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            data = pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)
            return data

    program = passes.synthesize_allreduce_signals()(P)
    program = passes.materialize_comm_domain_scopes()(program)
    result = passes.lower_host_tensor_collectives()(program)

    _assert_builtin_dispatch(
        result,
        "builtin.tensor.allreduce",
        arg_names=["data", "__allreduce_signal_0"],
        arg_directions=[ir.ArgDirection.InOut, ir.ArgDirection.InOut],
        kwargs={"op": int(pld.ReduceOp.Sum), "dtype": pl.FP32, "core_num": 1},
        attrs={"op": int(pld.ReduceOp.Sum), "dtype": pl.FP32, "core_num": 1},
    )


def test_return_implicit_host_allreduce_synthesizes_signal_then_lowers():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            return pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)

    program = cast(ir.Program, passes.synthesize_allreduce_signals()(P))
    program = cast(ir.Program, passes.materialize_comm_domain_scopes()(program))
    result = cast(ir.Program, passes.lower_host_tensor_collectives()(program))
    host = _get_func(result, "host_orch")

    returns = [
        stmt
        for stmt in _collect_assign_stmts(host.body)
        if isinstance(stmt.value, ir.Var) and stmt.var.name_hint.startswith("__allreduce_result_")
    ]
    return_stmts = _collect_return_stmts(host.body)

    _assert_builtin_dispatch(
        result,
        "builtin.tensor.allreduce",
        arg_names=["data", "__allreduce_signal_0"],
        arg_directions=[ir.ArgDirection.InOut, ir.ArgDirection.InOut],
        kwargs={"op": int(pld.ReduceOp.Sum), "dtype": pl.FP32, "core_num": 1},
        attrs={"op": int(pld.ReduceOp.Sum), "dtype": pl.FP32, "core_num": 1},
    )
    assert len(returns) == 1
    _assert_alias_keeps_window_buffer(returns[0])
    assert len(return_stmts) == 1
    assert isinstance(return_stmts[0].value[0], ir.Var)
    assert return_stmts[0].value[0] is returns[0].var
    assert _last_stmt(host.body) is return_stmts[0]


def test_return_explicit_host_allreduce_lowers_with_user_signal():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [pld.world_size(), 1], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            return pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)

    program = cast(ir.Program, passes.synthesize_allreduce_signals()(P))
    program = cast(ir.Program, passes.materialize_comm_domain_scopes()(program))
    result = cast(ir.Program, passes.lower_host_tensor_collectives()(program))
    host = _get_func(result, "host_orch")

    returns = [
        stmt
        for stmt in _collect_assign_stmts(host.body)
        if isinstance(stmt.value, ir.Var) and stmt.var.name_hint.startswith("__allreduce_result_")
    ]
    return_stmts = _collect_return_stmts(host.body)

    _assert_builtin_dispatch(
        result,
        "builtin.tensor.allreduce",
        arg_names=["data", "signal"],
        arg_directions=[ir.ArgDirection.InOut, ir.ArgDirection.InOut],
        kwargs={"op": int(pld.ReduceOp.Sum), "dtype": pl.FP32, "core_num": 1},
        attrs={"op": int(pld.ReduceOp.Sum), "dtype": pl.FP32, "core_num": 1},
    )
    assert len(returns) == 1
    _assert_alias_keeps_window_buffer(returns[0])
    assert len(return_stmts) == 1
    assert isinstance(return_stmts[0].value[0], ir.Var)
    assert return_stmts[0].value[0] is returns[0].var
    assert _last_stmt(host.body) is return_stmts[0]


def test_host_allreduce_assign_result_var_carries_window_buffer():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            self.chip_orch(data, device=0)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    result = passes.lower_host_tensor_collectives()(program)
    host = _get_func(result, "host_orch")

    allreduce_aliases = [
        stmt
        for stmt in _collect_assign_stmts(host.body)
        if isinstance(stmt.value, ir.Var)
        and isinstance(stmt.var.type, ir.DistributedTensorType)
        and isinstance(stmt.value.type, ir.DistributedTensorType)
        and stmt.var.name_hint == "data"
    ]
    assert len(allreduce_aliases) == 1
    _assert_alias_keeps_window_buffer(allreduce_aliases[0])


def test_host_allreduce_chained_assign_uses_remapped_result_var():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            signal_buf_1 = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            signal_1 = pld.window(signal_buf_1, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            data = pld.tensor.allreduce(data, signal_1, op=pld.ReduceOp.Sum)
            self.chip_orch(data, device=0)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    result = passes.lower_host_tensor_collectives()(program)
    host = _get_func(result, "host_orch")

    allreduce_aliases = [
        stmt
        for stmt in _collect_assign_stmts(host.body)
        if isinstance(stmt.value, ir.Var)
        and isinstance(stmt.var.type, ir.DistributedTensorType)
        and isinstance(stmt.value.type, ir.DistributedTensorType)
        and stmt.var.name_hint == "data"
    ]
    assert len(allreduce_aliases) == 2
    for alias in allreduce_aliases:
        _assert_alias_keeps_window_buffer(alias)


def test_host_allreduce_resolves_non_innermost_comm_domain_scope():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(type=pl.FunctionType.Orchestration)
        def other_chip_orch(self, data: pld.DistributedTensor[[128], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            other_buf = pld.alloc_window_buffer(128 * pl.FP32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            other = pld.window(other_buf, [128], dtype=pl.FP32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            self.other_chip_orch(other, device=0)
            pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    result = passes.lower_host_tensor_collectives()(program)
    host = _get_func(result, "host_orch")

    loops = _collect_for_stmts(host.body)
    builtin_loops = [
        loop
        for loop in loops
        if isinstance(loop.body, ir.EvalStmt)
        and isinstance(loop.body.expr, ir.Call)
        and loop.body.expr.op.name == ir.get_op("builtin.tensor.allreduce").name
    ]
    assert len(builtin_loops) == 1
    assert isinstance(builtin_loops[0].stop, ir.Call)
    assert builtin_loops[0].stop.op.name == ir.get_op("pld.system.world_size").name


def test_host_allreduce_rejects_static_signal_smaller_than_explicit_device_count():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [1], dtype=pl.INT32)
            self.chip_orch(data, device=0)
            self.chip_orch(data, device=1)
            pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(ValueError, match=r"signal shape\[0\].*participating device count"):
        passes.lower_host_tensor_collectives()(program)


def test_host_allreduce_rejects_rank2_signal_with_dynamic_second_extent():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            width = pld.world_size()
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [pld.world_size(), width], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(ValueError, match=r"rank-2 signal shape\[1\] must be constant"):
        passes.lower_host_tensor_collectives()(program)


def test_host_allreduce_multicore_propagates_core_num_and_accepts_wider_signal():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(pld.world_size() * 8 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [pld.world_size(), 8], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, core_num=4)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    result = passes.lower_host_tensor_collectives()(program)

    # A stride of 8 is wider than the 4 requested lanes: the spare lanes are
    # accepted and `core_num` reaches the builtin unchanged.
    _assert_builtin_dispatch(
        result,
        "builtin.tensor.allreduce",
        arg_names=["data", "signal"],
        arg_directions=[ir.ArgDirection.InOut, ir.ArgDirection.InOut],
        kwargs={"op": int(pld.ReduceOp.Sum), "dtype": pl.FP32, "core_num": 4},
        attrs={"op": int(pld.ReduceOp.Sum), "dtype": pl.FP32, "core_num": 4},
    )


def test_host_allreduce_multicore_rejects_rank1_signal():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(pld.world_size() * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [pld.world_size()], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, core_num=4)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(ValueError, match=r"rank-1 signal is valid only when one signal lane is required"):
        passes.lower_host_tensor_collectives()(program)


def test_host_allreduce_multicore_rejects_narrow_rank2_signal():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(pld.world_size() * 2 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [pld.world_size(), 2], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, core_num=4)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(ValueError, match=r"must be at least the required lane count"):
        passes.lower_host_tensor_collectives()(program)


def test_host_allreduce_rejects_core_num_above_backend_capacity(ascend_backend):
    """The builtin is submitted with ``rt_submit_aiv_task``, so one block maps to
    one AIV core and the bound is the vector-core count, not the cube-core count.

    The launch also sets ``require_sync_start``, so an unsatisfiable request would
    hang the device rather than fail — it has to be rejected at compile time.
    """
    backend = _backend.get_backend_instance(ascend_backend)
    aiv_cores = backend.get_core_count(ir.CoreType.VECTOR)
    aic_cores = backend.get_core_count(ir.CoreType.CUBE)
    # Guards the regression this test exists for: a cube-derived bound would
    # wrongly reject every core_num in (aic_cores, aiv_cores].
    assert aiv_cores > aic_cores

    def _build(core_num: int):
        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.Orchestration)
            def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
                return data

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self):
                data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
                signal_buf = pld.alloc_window_buffer(pld.world_size() * core_num * pl.INT32.get_byte())
                data = pld.window(data_buf, [256], dtype=pl.FP32)
                signal = pld.window(signal_buf, [pld.world_size(), core_num], dtype=pl.INT32)
                for r in pl.range(pld.world_size()):
                    self.chip_orch(data, device=r)
                pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, core_num=core_num)
                return 0

        return passes.materialize_comm_domain_scopes()(P)

    # Exactly at capacity is accepted.
    passes.lower_host_tensor_collectives()(_build(aiv_cores))

    with pytest.raises(ValueError, match="exceeds the backend AIV core count"):
        passes.lower_host_tensor_collectives()(_build(aiv_cores + 1))


def test_host_allreduce_ring_rejects_multicore():
    """Multicore is a mesh-only capability: the ring builtin runs one block per rank."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(pld.world_size() * 8 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [pld.world_size(), 8], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode="ring", core_num=4)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(ValueError, match=r'mode="ring" does not support core_num > 1'):
        passes.lower_host_tensor_collectives()(program)


# ---------------------------------------------------------------------------
# Phase A (plan 100): an absent `core_num` on a HOST mesh allreduce auto-selects
# the SPMD AIV width at lowering from (per-rank payload bytes, P). Policy table
# source: pypto-profiling corenum-message-size-crossover-2026-08-31.md — cn8's
# monotone-stable crossover is 256 KiB (P=2), 128 KiB (P=4), 64 KiB (P>=8); cn16
# only at >= 2 MiB. Unknown/dynamic world_size falls back to the P=2 column.
# Explicit `core_num=` always wins; `PYPTO_ALLREDUCE_CORE_NUM` forces the width.
# ---------------------------------------------------------------------------


def _builtin_allreduce_core_nums(host_body: ir.Stmt) -> list[int]:
    """Resolved ``core_num`` of every builtin.tensor.allreduce dispatch, in order."""
    found: list[int] = []
    name = ir.get_op("builtin.tensor.allreduce").name

    def walk(s: ir.Stmt) -> None:
        if isinstance(s, ir.EvalStmt) and isinstance(s.expr, ir.Call) and s.expr.op.name == name:
            found.append(s.expr.kwargs["core_num"])
        if isinstance(s, ir.SeqStmts):
            for c in s.stmts:
                walk(c)
        if isinstance(s, ir.ScopeStmt):
            walk(s.body)
        if isinstance(s, ir.ForStmt):
            walk(s.body)

    walk(host_body)
    return found


def _auto_allreduce_program(
    n_ranks: int,
    size: int,
    *,
    lanes: int,
    core_num: int | None = None,
    mode: str = "mesh",
    dynamic_world_size: bool = False,
):
    """Build a HOST mesh/ring allreduce program over ``n_ranks`` devices with an
    explicit rank-2 ``[rows, lanes]`` signal.

    ``core_num=None`` leaves the DSL default (auto-select);
    ``dynamic_world_size=True`` dispatches inside ``pl.range(pld.world_size())``
    instead of a static constant loop (the fully-dynamic all-device domain).
    Ring requires ``[2*(NR-1)+1, NR]`` (pass ``lanes=NR``).
    """
    rows = 2 * (n_ranks - 1) + 1 if mode == "ring" else n_ranks

    @pl.program
    class AutoAllreduceStatic:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[1, size], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(size * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(rows * lanes * pl.INT32.get_byte())
            data = pld.window(data_buf, [1, size], dtype=pl.FP32)
            signal = pld.window(signal_buf, [rows, lanes], dtype=pl.INT32)
            for r in pl.range(n_ranks):
                self.chip_orch(data, device=r)
            data = pld.window(data_buf, [1, size], dtype=pl.FP32)
            signal = pld.window(signal_buf, [rows, lanes], dtype=pl.INT32)
            pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode=mode, core_num=core_num)
            return 0

    @pl.program
    class AutoAllreduceDynamic:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[1, size], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(size * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(rows * lanes * pl.INT32.get_byte())
            data = pld.window(data_buf, [1, size], dtype=pl.FP32)
            signal = pld.window(signal_buf, [rows, lanes], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            data = pld.window(data_buf, [1, size], dtype=pl.FP32)
            signal = pld.window(signal_buf, [rows, lanes], dtype=pl.INT32)
            pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode=mode, core_num=core_num)
            return 0

    return AutoAllreduceDynamic if dynamic_world_size else AutoAllreduceStatic


def _lower_auto_allreduce(program) -> ir.Program:
    """Synthesize signals, materialize comm domains, and lower host collectives."""
    P = passes.convert_to_ssa()(program)
    P = passes.synthesize_allreduce_signals()(P)
    P = passes.materialize_comm_domain_scopes()(P)
    return passes.lower_host_tensor_collectives()(P)


@pytest.mark.parametrize(
    ("n_ranks", "size", "expected"),
    [
        # fp32 payload = size * 4 bytes. Rows tie each band to the measured
        # crossover (cn8 stable from 256 KiB @ P=2, 128 KiB @ P=4, 64 KiB @ P>=8).
        pytest.param(2, 100, 1, id="p2-400B-tiny"),
        pytest.param(2, 60000, 1, id="p2-240KiB-just-below-256KiB-crossover"),
        pytest.param(2, 70000, 8, id="p2-280KiB-just-above-256KiB-crossover"),
        pytest.param(2, 500000, 8, id="p2-2MiB-below-large-16-boundary"),
        pytest.param(2, 600000, 16, id="p2-2.4MiB-above-2MiB-16-boundary"),
        pytest.param(4, 30000, 1, id="p4-120KiB-just-below-128KiB-crossover"),
        pytest.param(4, 40000, 8, id="p4-160KiB-just-above-128KiB-crossover"),
        pytest.param(4, 70000, 8, id="p4-280KiB"),
        pytest.param(8, 12000, 1, id="p8-48KiB-just-below-64KiB-crossover"),
        pytest.param(8, 20000, 8, id="p8-80KiB-just-above-64KiB-crossover"),
        pytest.param(8, 70000, 8, id="p8-280KiB"),
        pytest.param(16, 20000, 8, id="p16-80KiB"),
    ],
)
def test_host_allreduce_auto_selects_core_num_by_payload_and_rank_count(n_ranks, size, expected):
    """Absent ``core_num`` selects the (payload, P) policy width at lowering."""
    program = _auto_allreduce_program(n_ranks, size, lanes=16)
    host = _get_func(_lower_auto_allreduce(program), "host_orch")
    nums = _builtin_allreduce_core_nums(host.body)
    assert len(nums) == n_ranks
    assert all(n == expected for n in nums), f"expected core_num={expected}, got {nums}"


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        pytest.param(20000, 1, id="80KiB-below-256KiB-P2-column"),
        pytest.param(70000, 8, id="280KiB-above-256KiB-P2-column"),
    ],
)
def test_host_allreduce_auto_dynamic_world_size_uses_conservative_p2_column(size, expected):
    """Fully-dynamic world_size (device loop over pld.world_size()) falls back to
    the P=2 column — the highest measured crossover, safe for every real P."""
    program = _auto_allreduce_program(4, size, lanes=16, dynamic_world_size=True)
    host = _get_func(_lower_auto_allreduce(program), "host_orch")
    nums = _builtin_allreduce_core_nums(host.body)
    assert nums == [expected], f"dynamic world_size: expected [{expected}], got {nums}"


def test_host_allreduce_auto_clamps_to_rank1_signal_lane_capacity():
    """A rank-1 [world_size] signal only carries one lane per rank: the purely
    auto width is clamped to 1 (single-AIV) so existing programs that sized a
    narrow signal keep compiling — matching the kernel's runtime clamp."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[1, 70000], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(70000 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(pld.world_size() * pl.INT32.get_byte())
            data = pld.window(data_buf, [1, 70000], dtype=pl.FP32)
            signal = pld.window(signal_buf, [pld.world_size()], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            data = pld.window(data_buf, [1, 70000], dtype=pl.FP32)
            signal = pld.window(signal_buf, [pld.world_size()], dtype=pl.INT32)
            pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return 0

    host = _get_func(_lower_auto_allreduce(P), "host_orch")
    nums = _builtin_allreduce_core_nums(host.body)
    assert nums == [1], f"rank-1 signal must clamp auto to 1, got {nums}"


def test_host_allreduce_auto_clamps_to_narrow_rank2_signal_lane_capacity():
    """A rank-2 signal with fewer lanes than the policy width clamps the auto
    width to the available lanes (the kernel would clamp identically)."""
    program = _auto_allreduce_program(4, 70000, lanes=2)
    host = _get_func(_lower_auto_allreduce(program), "host_orch")
    nums = _builtin_allreduce_core_nums(host.body)
    assert nums == [2, 2, 2, 2], f"2-lane signal must clamp auto to 2, got {nums}"


def test_host_allreduce_explicit_core_num_wins_over_auto_default():
    """Explicit ``core_num=1`` at a payload that would auto-select 8 stays 1."""
    program = _auto_allreduce_program(4, 70000, lanes=16, core_num=1)
    host = _get_func(_lower_auto_allreduce(program), "host_orch")
    nums = _builtin_allreduce_core_nums(host.body)
    assert nums == [1, 1, 1, 1], f"explicit core_num=1 must win, got {nums}"


def test_host_allreduce_auto_ring_stays_single_block():
    """Ring is single-block: an absent ``core_num`` resolves to 1 and the ring
    builtin carries no ``core_num`` at all (as before this default existed)."""
    program = _auto_allreduce_program(4, 70000, lanes=4, mode="ring")  # ring signal is [2*(NR-1)+1, NR]
    lowered = _lower_auto_allreduce(program)
    host = _get_func(lowered, "host_orch")
    ring_name = ir.get_op("builtin.tensor.allreduce_ring").name
    found = []

    def walk(s: ir.Stmt) -> None:
        if isinstance(s, ir.EvalStmt) and isinstance(s.expr, ir.Call) and s.expr.op.name == ring_name:
            found.append(s.expr)
        if isinstance(s, ir.SeqStmts):
            for c in s.stmts:
                walk(c)
        if isinstance(s, ir.ScopeStmt):
            walk(s.body)
        if isinstance(s, ir.ForStmt):
            walk(s.body)

    walk(host.body)
    assert len(found) == 4
    for call in found:
        assert "core_num" not in call.kwargs


def test_host_allreduce_auto_env_override(monkeypatch):
    """PYPTO_ALLREDUCE_CORE_NUM forces the width for absent-core_num calls (a
    forced width — a too-narrow signal is a hard error, like an explicit value)."""
    monkeypatch.setenv("PYPTO_ALLREDUCE_CORE_NUM", "8")
    program = _auto_allreduce_program(2, 100, lanes=16)  # tiny payload: auto would pick 1
    host = _get_func(_lower_auto_allreduce(program), "host_orch")
    nums = _builtin_allreduce_core_nums(host.body)
    assert nums == [8, 8], f"env override must force 8, got {nums}"


def test_host_allreduce_auto_env_override_rejects_narrow_signal(monkeypatch):
    """A forced env width on a too-narrow signal errors (it cannot be honored)."""
    monkeypatch.setenv("PYPTO_ALLREDUCE_CORE_NUM", "8")
    program = _auto_allreduce_program(2, 100, lanes=1)  # [2, 1] signal
    with pytest.raises(ValueError, match=r"at least the required lane count"):
        _lower_auto_allreduce(program)


def test_host_allreduce_explicit_core_num_wins_over_env_override(monkeypatch):
    """Explicit ``core_num=`` beats the env override."""
    monkeypatch.setenv("PYPTO_ALLREDUCE_CORE_NUM", "8")
    program = _auto_allreduce_program(2, 100, lanes=16, core_num=1)
    host = _get_func(_lower_auto_allreduce(program), "host_orch")
    nums = _builtin_allreduce_core_nums(host.body)
    assert nums == [1, 1], f"explicit core_num=1 must beat env=8, got {nums}"


def test_host_allreduce_rejects_unsupported_dtype_before_lowering():
    with pytest.raises(InvalidOperationError, match="target dtype must be FP16 or FP32"):

        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.Orchestration)
            def chip_orch(self, data: pld.DistributedTensor[[256], pl.BF16]):
                return data

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self):
                data_buf = pld.alloc_window_buffer(256 * pl.BF16.get_byte())
                signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
                data = pld.window(data_buf, [256], dtype=pl.BF16)
                signal = pld.window(signal_buf, [4], dtype=pl.INT32)
                for r in pl.range(pld.world_size()):
                    self.chip_orch(data, device=r)
                pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
                return 0


def test_host_allreduce_rejects_aliased_src_signal_windows():
    """Two pld.window views over one alloc must fail at host lowering — the
    generic pairwise-distinct check (CheckPairwiseDistinctWindows) applies to
    every HOST collective's window operands, not just input/target pairs."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(buf, [256], dtype=pl.FP32)
            # signal aliases data's own window buffer (mismatched dtype view
            # over the same allocation is exactly the race this check exists for).
            signal = pld.window(buf, [256], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(ValueError, match=r"src and signal must be different window allocations"):
        passes.lower_host_tensor_collectives()(program)


def test_host_barrier_lowers_to_builtin_world_size_loop():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self, data: pld.DistributedTensor[[256], pl.FP32], sig: pld.DistributedTensor[[4], pl.INT32]
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, signal, device=r)
            pld.tensor.barrier(signal)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    result = passes.lower_host_tensor_collectives()(program)
    _assert_builtin_dispatch(
        result,
        "builtin.tensor.barrier",
        arg_names=["signal"],
        arg_directions=[ir.ArgDirection.InOut],
        kwargs={},
    )


def test_host_barrier_accepts_rank1_signal_through_lowering():
    """Rank-1 [NR] must pass both the public deducer and the builtin
    validator inside LowerHostTensorCollectives."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self, data: pld.DistributedTensor[[256], pl.FP32], sig: pld.DistributedTensor[[4], pl.INT32]
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, signal, device=r)
            pld.tensor.barrier(signal)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    result = passes.lower_host_tensor_collectives()(program)
    host = _get_func(result, "host_orch")
    loops = _collect_for_stmts(host.body)
    builtin_loops = [
        loop
        for loop in loops
        if isinstance(loop.body, ir.EvalStmt)
        and isinstance(loop.body.expr, ir.Call)
        and loop.body.expr.op.name == _BUILTIN_BARRIER
    ]
    assert len(builtin_loops) == 1


def test_host_broadcast_lowers_to_builtin_world_size_loop():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self, data: pld.DistributedTensor[[256], pl.FP32], sig: pld.DistributedTensor[[4], pl.INT32]
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, signal, device=r)
            pld.tensor.broadcast(data, signal, root=0)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    result = passes.lower_host_tensor_collectives()(program)
    _assert_builtin_dispatch(
        result,
        "builtin.tensor.broadcast",
        arg_names=["data", "signal"],
        arg_directions=[ir.ArgDirection.InOut, ir.ArgDirection.InOut],
        kwargs={"root": 0, "dtype": pl.FP32},
        attrs={"root": 0, "dtype": pl.FP32},
    )


def test_host_broadcast_accepts_rank1_signal_through_lowering():
    """Rank-1 [NR] must survive public + builtin validation on the HOST
    broadcast path."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self, data: pld.DistributedTensor[[256], pl.FP32], sig: pld.DistributedTensor[[4], pl.INT32]
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, signal, device=r)
            pld.tensor.broadcast(data, signal, root=0)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    result = passes.lower_host_tensor_collectives()(program)
    host = _get_func(result, "host_orch")
    loops = _collect_for_stmts(host.body)
    builtin_loops = [
        loop
        for loop in loops
        if isinstance(loop.body, ir.EvalStmt)
        and isinstance(loop.body.expr, ir.Call)
        and loop.body.expr.op.name == _BUILTIN_BROADCAST
    ]
    assert len(builtin_loops) == 1


def test_host_reduce_scatter_lowers_to_builtin_world_size_loop():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self, data: pld.DistributedTensor[[4, 256], pl.FP32], sig: pld.DistributedTensor[[4], pl.INT32]
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(4 * 256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [4, 256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, signal, device=r)
            pld.tensor.reduce_scatter(data, signal, op=pld.ReduceOp.Sum)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    result = passes.lower_host_tensor_collectives()(program)
    _assert_builtin_dispatch(
        result,
        "builtin.tensor.reduce_scatter",
        arg_names=["data", "signal"],
        arg_directions=[ir.ArgDirection.InOut, ir.ArgDirection.InOut],
        kwargs={"op": int(pld.ReduceOp.Sum), "dtype": pl.FP32},
        attrs={"op": int(pld.ReduceOp.Sum), "dtype": pl.FP32},
    )


def test_host_reduce_scatter_accepts_rank1_signal_through_lowering():
    """Rank-1 [NR] must survive public + builtin validation on the HOST
    reduce_scatter path."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            data: pld.DistributedTensor[[4, 256], pl.FP32],
            sig: pld.DistributedTensor[[4], pl.INT32],
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(4 * 256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [4, 256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, signal, device=r)
            pld.tensor.reduce_scatter(data, signal, op=pld.ReduceOp.Sum)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    result = passes.lower_host_tensor_collectives()(program)
    host = _get_func(result, "host_orch")
    loops = _collect_for_stmts(host.body)
    builtin_loops = [
        loop
        for loop in loops
        if isinstance(loop.body, ir.EvalStmt)
        and isinstance(loop.body.expr, ir.Call)
        and loop.body.expr.op.name == _BUILTIN_REDUCE_SCATTER
    ]
    assert len(builtin_loops) == 1


def test_host_broadcast_rejects_aliased_target_signal_windows():
    """target and signal sharing an allocation must fail — the broadcast TPUT
    write into `target` races the notify/wait control path over `signal`."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self, data: pld.DistributedTensor[[256], pl.FP32], sig: pld.DistributedTensor[[4], pl.INT32]
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data = pld.window(buf, [256], dtype=pl.FP32)
            # signal aliases data's own window buffer.
            signal = pld.window(buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, signal, device=r)
            pld.tensor.broadcast(data, signal, root=0)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(ValueError, match=r"target and signal must be different window allocations"):
        passes.lower_host_tensor_collectives()(program)


def test_host_broadcast_rejects_out_of_range_root_on_explicit_device_subset():
    """root=5 on an explicit 2-device static subset must fail — the type
    deducer only checks root >= 0 (the participating device count isn't known
    there); the upper bound is enforced once the static subset size is known."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self, data: pld.DistributedTensor[[256], pl.FP32], sig: pld.DistributedTensor[[4], pl.INT32]
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            self.chip_orch(data, signal, device=0)
            self.chip_orch(data, signal, device=1)
            pld.tensor.broadcast(data, signal, root=5)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(ValueError, match=r"root \(5\) must be a valid rank in \[0, 2\)"):
        passes.lower_host_tensor_collectives()(program)


def test_host_reduce_scatter_rejects_aliased_target_signal_windows():
    """target and signal sharing an allocation must fail — same discipline as
    broadcast: the reduce TPUT write into `target` races the notify/wait
    control path over `signal`."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self, data: pld.DistributedTensor[[4, 256], pl.FP32], sig: pld.DistributedTensor[[4], pl.INT32]
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(4 * 256 * pl.FP32.get_byte())
            data = pld.window(buf, [4, 256], dtype=pl.FP32)
            # signal aliases data's own window buffer.
            signal = pld.window(buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, signal, device=r)
            pld.tensor.reduce_scatter(data, signal, op=pld.ReduceOp.Sum)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(ValueError, match=r"target and signal must be different window allocations"):
        passes.lower_host_tensor_collectives()(program)


def test_host_allgather_lowers_to_namesake_builtin():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            stage: pld.DistributedTensor[[1, 256], pl.FP32],
            data: pld.DistributedTensor[[4, 256], pl.FP32],
            sig: pld.DistributedTensor[[4], pl.INT32],
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            # `stage_buf` ([1, SIZE] TPUT source, this rank's chunk) and
            # `data_buf` ([NR, SIZE] TPUT destination / result) must be two
            # DISTINCT windows — same constraint as all_to_all (see allgather
            # kernel.cpp.in).
            stage_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            data_buf = pld.alloc_window_buffer(4 * 256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            stage = pld.window(stage_buf, [1, 256], dtype=pl.FP32)
            data = pld.window(data_buf, [4, 256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(stage, data, signal, device=r)
            data = pld.tensor.allgather(stage, data, signal)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    result = passes.lower_host_tensor_collectives()(program)
    _assert_builtin_dispatch(
        result,
        "builtin.tensor.allgather",
        arg_names=["stage", "data", "signal"],
        arg_directions=[
            ir.ArgDirection.Input,
            ir.ArgDirection.InOut,
            ir.ArgDirection.InOut,
        ],
        kwargs={"dtype": pl.FP32},
        attrs={"dtype": pl.FP32},
    )


def test_host_allgather_rejects_aliased_input_target_windows():
    """Two pld.window views over one alloc must fail at host lowering."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            stage: pld.DistributedTensor[[1, 256], pl.FP32],
            data: pld.DistributedTensor[[4, 256], pl.FP32],
            sig: pld.DistributedTensor[[4], pl.INT32],
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(4 * 256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            stage = pld.window(buf, [1, 256], dtype=pl.FP32)
            data = pld.window(buf, [4, 256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(stage, data, signal, device=r)
            data = pld.tensor.allgather(stage, data, signal)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(ValueError, match=r"different window allocations"):
        passes.lower_host_tensor_collectives()(program)


def test_host_allgather_rejects_aliased_input_signal_windows():
    """input aliasing signal must fail too — the generic pairwise check covers
    every pair of allgather's 3 window operands, not just input vs target
    (which was already rejected before this generalization)."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            stage: pld.DistributedTensor[[1, 256], pl.FP32],
            data: pld.DistributedTensor[[4, 256], pl.FP32],
            sig: pld.DistributedTensor[[4], pl.INT32],
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            # signal_buf is sized to fit the [1, 256] FP32 view so the test
            # exercises aliasing alone, not a view-vs-buffer footprint check.
            signal_buf = pld.alloc_window_buffer(1 * 256 * pl.FP32.get_byte())
            data_buf = pld.alloc_window_buffer(4 * 256 * pl.FP32.get_byte())
            # stage aliases signal's own window buffer.
            stage = pld.window(signal_buf, [1, 256], dtype=pl.FP32)
            data = pld.window(data_buf, [4, 256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(stage, data, signal, device=r)
            data = pld.tensor.allgather(stage, data, signal)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(ValueError, match=r"input and signal must be different window allocations"):
        passes.lower_host_tensor_collectives()(program)


def test_host_allgather_rejects_aliased_target_signal_windows():
    """target aliasing signal must fail too."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            stage: pld.DistributedTensor[[1, 256], pl.FP32],
            data: pld.DistributedTensor[[4, 256], pl.FP32],
            sig: pld.DistributedTensor[[4], pl.INT32],
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            stage_buf = pld.alloc_window_buffer(1 * 256 * pl.FP32.get_byte())
            # signal_buf is sized to fit the [4, 256] FP32 view so the test
            # exercises aliasing alone, not a view-vs-buffer footprint check.
            signal_buf = pld.alloc_window_buffer(4 * 256 * pl.FP32.get_byte())
            stage = pld.window(stage_buf, [1, 256], dtype=pl.FP32)
            # data aliases signal's own window buffer.
            data = pld.window(signal_buf, [4, 256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(stage, data, signal, device=r)
            data = pld.tensor.allgather(stage, data, signal)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(ValueError, match=r"target and signal must be different window allocations"):
        passes.lower_host_tensor_collectives()(program)


def test_host_allgather_rejects_plain_tensor_input():
    """pld.tensor.allgather's public deducer accepts a plain Tensor for
    `local_data` (legitimate on the InCore composite path), but the HOST
    builtin requires it window-bound. This must surface as a CHECK_SPAN
    ValueError, not an internal crash inside GetWindowBuffer."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            stage: pl.Tensor[[1, 256], pl.FP32],
            data: pld.DistributedTensor[[4, 256], pl.FP32],
            sig: pld.DistributedTensor[[4], pl.INT32],
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self, stage: pl.Tensor[[1, 256], pl.FP32]):
            data_buf = pld.alloc_window_buffer(4 * 256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [4, 256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(stage, data, signal, device=r)
            data = pld.tensor.allgather(stage, data, signal)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(ValueError, match=r"input must be a window-bound DistributedTensor"):
        passes.lower_host_tensor_collectives()(program)


def test_host_all_to_all_rejects_aliased_input_target_windows():
    """Two pld.window views over one alloc must fail at host lowering."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            stage: pld.DistributedTensor[[4, 256], pl.FP32],
            data: pld.DistributedTensor[[4, 256], pl.FP32],
            sig: pld.DistributedTensor[[4], pl.INT32],
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(4 * 256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            stage = pld.window(buf, [4, 256], dtype=pl.FP32)
            data = pld.window(buf, [4, 256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(stage, data, signal, device=r)
            data = pld.tensor.all_to_all(stage, data, signal)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(ValueError, match=r"different window allocations"):
        passes.lower_host_tensor_collectives()(program)


def test_host_all_to_all_rejects_aliased_input_signal_windows():
    """input aliasing signal must fail too — same generalization as allgather."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            stage: pld.DistributedTensor[[4, 256], pl.FP32],
            data: pld.DistributedTensor[[4, 256], pl.FP32],
            sig: pld.DistributedTensor[[4], pl.INT32],
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            # signal_buf is sized to fit the [4, 256] FP32 view so the test
            # exercises aliasing alone, not a view-vs-buffer footprint check.
            signal_buf = pld.alloc_window_buffer(4 * 256 * pl.FP32.get_byte())
            data_buf = pld.alloc_window_buffer(4 * 256 * pl.FP32.get_byte())
            # stage aliases signal's own window buffer.
            stage = pld.window(signal_buf, [4, 256], dtype=pl.FP32)
            data = pld.window(data_buf, [4, 256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(stage, data, signal, device=r)
            data = pld.tensor.all_to_all(stage, data, signal)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(ValueError, match=r"input and signal must be different window allocations"):
        passes.lower_host_tensor_collectives()(program)


def test_host_all_to_all_rejects_aliased_target_signal_windows():
    """target aliasing signal must fail too."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            stage: pld.DistributedTensor[[4, 256], pl.FP32],
            data: pld.DistributedTensor[[4, 256], pl.FP32],
            sig: pld.DistributedTensor[[4], pl.INT32],
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            stage_buf = pld.alloc_window_buffer(4 * 256 * pl.FP32.get_byte())
            # signal_buf is sized to fit the [4, 256] FP32 view so the test
            # exercises aliasing alone, not a view-vs-buffer footprint check.
            signal_buf = pld.alloc_window_buffer(4 * 256 * pl.FP32.get_byte())
            stage = pld.window(stage_buf, [4, 256], dtype=pl.FP32)
            # data aliases signal's own window buffer.
            data = pld.window(signal_buf, [4, 256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(stage, data, signal, device=r)
            data = pld.tensor.all_to_all(stage, data, signal)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(ValueError, match=r"target and signal must be different window allocations"):
        passes.lower_host_tensor_collectives()(program)


def test_host_all_to_all_rejects_plain_tensor_input():
    """Same as the allgather case, for `input` (also Tensor | DistributedTensor
    on the composite path, but window-bound-only at the HOST builtin layer)."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            stage: pl.Tensor[[4, 256], pl.FP32],
            data: pld.DistributedTensor[[4, 256], pl.FP32],
            sig: pld.DistributedTensor[[4], pl.INT32],
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self, stage: pl.Tensor[[4, 256], pl.FP32]):
            data_buf = pld.alloc_window_buffer(4 * 256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [4, 256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(stage, data, signal, device=r)
            data = pld.tensor.all_to_all(stage, data, signal)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(ValueError, match=r"input must be a window-bound DistributedTensor"):
        passes.lower_host_tensor_collectives()(program)


def test_host_all_to_all_lowers_to_namesake_builtin():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            stage: pld.DistributedTensor[[4, 256], pl.FP32],
            data: pld.DistributedTensor[[4, 256], pl.FP32],
            sig: pld.DistributedTensor[[4], pl.INT32],
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            # `stage_buf` (TPUT source) and `data_buf` (TPUT destination /
            # result) must be two DISTINCT windows — reusing one buffer for
            # both is a genuine cross-process data race (see
            # python/pypto/runtime/builtins/collectives/all_to_all/templates
            # /kernel.cpp.in for the full explanation).
            stage_buf = pld.alloc_window_buffer(4 * 256 * pl.FP32.get_byte())
            data_buf = pld.alloc_window_buffer(4 * 256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            stage = pld.window(stage_buf, [4, 256], dtype=pl.FP32)
            data = pld.window(data_buf, [4, 256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(stage, data, signal, device=r)
            data = pld.tensor.all_to_all(stage, data, signal)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    result = passes.lower_host_tensor_collectives()(program)

    _assert_builtin_dispatch(
        result,
        "builtin.tensor.all_to_all",
        arg_names=["stage", "data", "signal"],
        arg_directions=[
            ir.ArgDirection.Input,
            ir.ArgDirection.InOut,
            ir.ArgDirection.InOut,
        ],
        kwargs={"dtype": pl.FP32},
        attrs={"dtype": pl.FP32},
    )


def test_host_all_to_all_v_rejects_aliased_input_target_windows():
    """Two pld.window views over one alloc must fail at host lowering."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pld.DistributedTensor[[8, 256], pl.FP32],
            data: pld.DistributedTensor[[8, 256], pl.FP32],
            sig: pld.DistributedTensor[[4, 1], pl.INT32],
            counts: pld.DistributedTensor[[4, 1], pl.INT32],
            recv: pld.DistributedTensor[[4, 1], pl.INT32],
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            buf = pld.alloc_window_buffer(8 * 256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            counts_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            recv_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            inp = pld.window(buf, [8, 256], dtype=pl.FP32)
            data = pld.window(buf, [8, 256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4, 1], dtype=pl.INT32)
            counts = pld.window(counts_buf, [4, 1], dtype=pl.INT32)
            recv = pld.window(recv_buf, [4, 1], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(inp, data, signal, counts, recv, device=r)
            data = pld.tensor.all_to_all_v(inp, data, signal, counts, recv)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(ValueError, match=r"different window allocations"):
        passes.lower_host_tensor_collectives()(program)


def test_host_all_to_all_v_lowers_to_namesake_builtin():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pld.DistributedTensor[[8, 256], pl.FP32],
            data: pld.DistributedTensor[[8, 256], pl.FP32],
            sig: pld.DistributedTensor[[4, 1], pl.INT32],
            counts: pld.DistributedTensor[[4, 1], pl.INT32],
            recv: pld.DistributedTensor[[4, 1], pl.INT32],
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            # `input_buf` (TPUT source) and `data_buf` (TPUT destination /
            # result) must be two DISTINCT windows — same discipline as
            # symmetric all_to_all (see kernel.cpp.in for the race
            # explanation).
            input_buf = pld.alloc_window_buffer(8 * 256 * pl.FP32.get_byte())
            data_buf = pld.alloc_window_buffer(8 * 256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            counts_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            recv_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            inp = pld.window(input_buf, [8, 256], dtype=pl.FP32)
            data = pld.window(data_buf, [8, 256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4, 1], dtype=pl.INT32)
            counts = pld.window(counts_buf, [4, 1], dtype=pl.INT32)
            recv = pld.window(recv_buf, [4, 1], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(inp, data, signal, counts, recv, device=r)
            data = pld.tensor.all_to_all_v(inp, data, signal, counts, recv)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    result = passes.lower_host_tensor_collectives()(program)
    _assert_builtin_dispatch(
        result,
        "builtin.tensor.all_to_all_v",
        arg_names=["inp", "data", "signal", "counts", "recv"],
        arg_directions=[
            ir.ArgDirection.Input,
            ir.ArgDirection.InOut,
            ir.ArgDirection.InOut,
            ir.ArgDirection.Input,
            ir.ArgDirection.InOut,
        ],
        kwargs={"dtype": pl.FP32},
        attrs={"dtype": pl.FP32},
    )


def test_host_all_to_all_v_rejects_plain_tensor_input():
    """pld.tensor.all_to_all_v's public deducer accepts a plain Tensor for
    `input` (legitimate on the InCore composite path), but the HOST builtin
    requires it window-bound. This must surface as a CHECK_SPAN ValueError,
    not an internal crash inside GetWindowBuffer."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[8, 256], pl.FP32],
            data: pld.DistributedTensor[[8, 256], pl.FP32],
            sig: pld.DistributedTensor[[4, 1], pl.INT32],
            counts: pld.DistributedTensor[[4, 1], pl.INT32],
            recv: pld.DistributedTensor[[4, 1], pl.INT32],
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self, inp: pl.Tensor[[8, 256], pl.FP32]):
            data_buf = pld.alloc_window_buffer(8 * 256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            counts_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            recv_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [8, 256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4, 1], dtype=pl.INT32)
            counts = pld.window(counts_buf, [4, 1], dtype=pl.INT32)
            recv = pld.window(recv_buf, [4, 1], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(inp, data, signal, counts, recv, device=r)
            data = pld.tensor.all_to_all_v(inp, data, signal, counts, recv)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(ValueError, match=r"input must be a window-bound DistributedTensor"):
        passes.lower_host_tensor_collectives()(program)


def test_host_all_to_all_v_rejects_plain_tensor_send_counts():
    """Same as above, for `send_counts` (also AsTensorTypeLike on the
    composite path, but window-bound-only at the HOST builtin layer)."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pld.DistributedTensor[[8, 256], pl.FP32],
            data: pld.DistributedTensor[[8, 256], pl.FP32],
            sig: pld.DistributedTensor[[4, 1], pl.INT32],
            counts: pl.Tensor[[4, 1], pl.INT32],
            recv: pld.DistributedTensor[[4, 1], pl.INT32],
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self, counts: pl.Tensor[[4, 1], pl.INT32]):
            input_buf = pld.alloc_window_buffer(8 * 256 * pl.FP32.get_byte())
            data_buf = pld.alloc_window_buffer(8 * 256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            recv_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            inp = pld.window(input_buf, [8, 256], dtype=pl.FP32)
            data = pld.window(data_buf, [8, 256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4, 1], dtype=pl.INT32)
            recv = pld.window(recv_buf, [4, 1], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(inp, data, signal, counts, recv, device=r)
            data = pld.tensor.all_to_all_v(inp, data, signal, counts, recv)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(ValueError, match=r"send_counts must be a window-bound DistributedTensor"):
        passes.lower_host_tensor_collectives()(program)


def test_host_all_to_all_v_rejects_unbound_distributed_input():
    """A user-declared ``pld.DistributedTensor`` parameter has ``window_buffer_
    == nullopt`` (never bound by ``pld.tensor.window``), so it passes a
    kind-only type check and would crash inside ``GetWindowBuffer``. Passing
    one as ``input`` must surface as the same documented CHECK_SPAN ValueError,
    not an INTERNAL_CHECK_SPAN compiler-invariant failure."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pld.DistributedTensor[[8, 256], pl.FP32],
            data: pld.DistributedTensor[[8, 256], pl.FP32],
            sig: pld.DistributedTensor[[4, 1], pl.INT32],
            counts: pld.DistributedTensor[[4, 1], pl.INT32],
            recv: pld.DistributedTensor[[4, 1], pl.INT32],
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self, inp: pld.DistributedTensor[[8, 256], pl.FP32]):
            data_buf = pld.alloc_window_buffer(8 * 256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            counts_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            recv_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [8, 256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4, 1], dtype=pl.INT32)
            counts = pld.window(counts_buf, [4, 1], dtype=pl.INT32)
            recv = pld.window(recv_buf, [4, 1], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(inp, data, signal, counts, recv, device=r)
            data = pld.tensor.all_to_all_v(inp, data, signal, counts, recv)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(ValueError, match=r"input must be a window-bound DistributedTensor"):
        passes.lower_host_tensor_collectives()(program)


def test_host_all_to_all_v_rejects_unbound_distributed_send_counts():
    """Same as above, for `send_counts` (a user-declared DistributedTensor
    parameter, unbound by any ``pld.tensor.window``)."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pld.DistributedTensor[[8, 256], pl.FP32],
            data: pld.DistributedTensor[[8, 256], pl.FP32],
            sig: pld.DistributedTensor[[4, 1], pl.INT32],
            counts: pld.DistributedTensor[[4, 1], pl.INT32],
            recv: pld.DistributedTensor[[4, 1], pl.INT32],
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self, counts: pld.DistributedTensor[[4, 1], pl.INT32]):
            input_buf = pld.alloc_window_buffer(8 * 256 * pl.FP32.get_byte())
            data_buf = pld.alloc_window_buffer(8 * 256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            recv_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            inp = pld.window(input_buf, [8, 256], dtype=pl.FP32)
            data = pld.window(data_buf, [8, 256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4, 1], dtype=pl.INT32)
            recv = pld.window(recv_buf, [4, 1], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(inp, data, signal, counts, recv, device=r)
            data = pld.tensor.all_to_all_v(inp, data, signal, counts, recv)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(ValueError, match=r"send_counts must be a window-bound DistributedTensor"):
        passes.lower_host_tensor_collectives()(program)


def test_host_all_to_all_v_rejects_aliased_signal_recv_counts():
    """signal and recv_counts are separate INT32 control windows — aliasing
    them lets the barrier's notify(Set, 1) clobber a just-published count."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pld.DistributedTensor[[8, 256], pl.FP32],
            data: pld.DistributedTensor[[8, 256], pl.FP32],
            sig: pld.DistributedTensor[[4, 1], pl.INT32],
            counts: pld.DistributedTensor[[4, 1], pl.INT32],
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            input_buf = pld.alloc_window_buffer(8 * 256 * pl.FP32.get_byte())
            data_buf = pld.alloc_window_buffer(8 * 256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            counts_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            inp = pld.window(input_buf, [8, 256], dtype=pl.FP32)
            data = pld.window(data_buf, [8, 256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4, 1], dtype=pl.INT32)
            counts = pld.window(counts_buf, [4, 1], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(inp, data, signal, counts, device=r)
            # recv_counts aliases signal's own window buffer.
            recv = pld.window(signal_buf, [4, 1], dtype=pl.INT32)
            data = pld.tensor.all_to_all_v(inp, data, signal, counts, recv)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(ValueError, match=r"signal and recv_counts must be different window allocations"):
        passes.lower_host_tensor_collectives()(program)


def test_host_all_to_all_v_rejects_aliased_send_counts_recv_counts():
    """send_counts aliasing recv_counts lets the kernel's local count read
    race a peer's cross-rank notify write into the same memory."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pld.DistributedTensor[[8, 256], pl.FP32],
            data: pld.DistributedTensor[[8, 256], pl.FP32],
            counts: pld.DistributedTensor[[4, 1], pl.INT32],
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            input_buf = pld.alloc_window_buffer(8 * 256 * pl.FP32.get_byte())
            data_buf = pld.alloc_window_buffer(8 * 256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            recv_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            inp = pld.window(input_buf, [8, 256], dtype=pl.FP32)
            data = pld.window(data_buf, [8, 256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4, 1], dtype=pl.INT32)
            recv = pld.window(recv_buf, [4, 1], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(inp, data, recv, device=r)
            # send_counts aliases recv_counts's own window buffer.
            counts = pld.window(recv_buf, [4, 1], dtype=pl.INT32)
            data = pld.tensor.all_to_all_v(inp, data, signal, counts, recv)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(ValueError, match=r"send_counts and recv_counts must be different window allocations"):
        passes.lower_host_tensor_collectives()(program)


def test_host_all_to_all_v_rejects_aliased_input_signal_windows():
    """input is a data window and signal is a control window, but sharing an
    allocation still races: a peer's notify(Set, 1) can clobber data this
    rank is still TPUT-reading out of `input`."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pld.DistributedTensor[[8, 256], pl.FP32],
            data: pld.DistributedTensor[[8, 256], pl.FP32],
            sig: pld.DistributedTensor[[4, 1], pl.INT32],
            counts: pld.DistributedTensor[[4, 1], pl.INT32],
            recv: pld.DistributedTensor[[4, 1], pl.INT32],
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            # signal_buf is sized to fit the [8, 256] FP32 view so the test
            # exercises aliasing alone, not a view-vs-buffer footprint check.
            signal_buf = pld.alloc_window_buffer(8 * 256 * pl.FP32.get_byte())
            data_buf = pld.alloc_window_buffer(8 * 256 * pl.FP32.get_byte())
            counts_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            recv_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            # input aliases signal's own window buffer.
            inp = pld.window(signal_buf, [8, 256], dtype=pl.FP32)
            data = pld.window(data_buf, [8, 256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4, 1], dtype=pl.INT32)
            counts = pld.window(counts_buf, [4, 1], dtype=pl.INT32)
            recv = pld.window(recv_buf, [4, 1], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(inp, data, signal, counts, recv, device=r)
            data = pld.tensor.all_to_all_v(inp, data, signal, counts, recv)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(ValueError, match=r"input and signal must be different window allocations"):
        passes.lower_host_tensor_collectives()(program)


def test_host_all_to_all_v_rejects_aliased_target_recv_counts_windows():
    """target is a data window and recv_counts is a control window, but
    sharing an allocation still races: the cross-rank count publish can
    clobber data a peer's TPUT is writing into `target`."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pld.DistributedTensor[[8, 256], pl.FP32],
            data: pld.DistributedTensor[[8, 256], pl.FP32],
            sig: pld.DistributedTensor[[4, 1], pl.INT32],
            counts: pld.DistributedTensor[[4, 1], pl.INT32],
            recv: pld.DistributedTensor[[4, 1], pl.INT32],
        ):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            input_buf = pld.alloc_window_buffer(8 * 256 * pl.FP32.get_byte())
            recv_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            counts_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            inp = pld.window(input_buf, [8, 256], dtype=pl.FP32)
            # target aliases recv_counts's own window buffer.
            data = pld.window(recv_buf, [8, 256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4, 1], dtype=pl.INT32)
            counts = pld.window(counts_buf, [4, 1], dtype=pl.INT32)
            recv = pld.window(recv_buf, [4, 1], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(inp, data, signal, counts, recv, device=r)
            data = pld.tensor.all_to_all_v(inp, data, signal, counts, recv)
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(ValueError, match=r"target and recv_counts must be different window allocations"):
        passes.lower_host_tensor_collectives()(program)


def test_lowered_collective_is_printable_with_dtype_attr():
    """The lowered ``builtin.tensor.*`` call must survive the python printer.

    ``MakeBuiltinCallWithAttrs`` stamps a ``DataType``-valued ``dtype`` attr on
    every lowered collective. The pass-dump instrument (``pass_manager.after_pass``)
    prints the program after each pass, so a value type the printer has no codec
    arm for aborts the whole compile with an ``InternalError`` — which is how this
    surfaced in the distributed system tests. Printing here keeps the DataType
    codec arm wired to the pass that actually produces it.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return 0

    program = cast(ir.Program, passes.materialize_comm_domain_scopes()(P))
    result = cast(ir.Program, passes.lower_host_tensor_collectives()(program))

    printed = ir.python_print(result, format=False)
    assert "builtin.tensor.allreduce" in printed, printed
    # The DataType attr renders in the ``pl.<DTYPE>`` DSL form, not dropped.
    assert '"dtype": pl.FP32' in printed, printed


def test_host_allreduce_ring_lowers_to_ring_builtin():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(7 * 4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [7, 4], dtype=pl.INT32)
            for r in pl.range(pld.world_size()):
                self.chip_orch(data, device=r)
            pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode="ring")
            return 0

    program = cast(ir.Program, passes.materialize_comm_domain_scopes()(P))
    result = cast(ir.Program, passes.lower_host_tensor_collectives()(program))

    _assert_builtin_dispatch(
        result,
        "builtin.tensor.allreduce_ring",
        arg_names=["data", "signal"],
        arg_directions=[ir.ArgDirection.InOut, ir.ArgDirection.InOut],
        kwargs={"op": int(pld.ReduceOp.Sum), "dtype": pl.FP32},
        attrs={"op": int(pld.ReduceOp.Sum), "dtype": pl.FP32},
    )


def test_host_allreduce_rejects_unknown_mode():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [4], dtype=pl.INT32)
            self.chip_orch(data, device=0)
            self.chip_orch(data, device=1)
            self.chip_orch(data, device=2)
            self.chip_orch(data, device=3)
            pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode="star")
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(ValueError, match=r'mode must be "ring" or "mesh"'):
        passes.lower_host_tensor_collectives()(program)


def test_host_allreduce_ring_rejects_mismatched_signal_shape():
    """Ring signal [5, 4] fails: shape[0]=5 != 2*(4-1)+1 = 7 at P=4."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[256], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(5 * 4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [256], dtype=pl.FP32)
            signal = pld.window(signal_buf, [5, 4], dtype=pl.INT32)
            self.chip_orch(data, device=0)
            self.chip_orch(data, device=1)
            self.chip_orch(data, device=2)
            self.chip_orch(data, device=3)
            pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode="ring")
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(ValueError, match=r"must be at least 2\*\(NR-1\)"):
        passes.lower_host_tensor_collectives()(program)


def test_host_allreduce_ring_rejects_nondivisible_numel():
    """Ring allreduce rejects numel that is not an exact multiple of NR at P=4."""
    SIZE = 27  # 27 % 4 != 0

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[SIZE], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(7 * 4 * pl.INT32.get_byte())
            data = pld.window(data_buf, [SIZE], dtype=pl.FP32)
            signal = pld.window(signal_buf, [7, 4], dtype=pl.INT32)
            self.chip_orch(data, device=0)
            self.chip_orch(data, device=1)
            self.chip_orch(data, device=2)
            self.chip_orch(data, device=3)
            pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode="ring")
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(ValueError, match=r"exact multiple of the rank count"):
        passes.lower_host_tensor_collectives()(program)


def test_host_allreduce_ring_rejects_too_many_ranks():
    """Ring allreduce rejects more than 16 participating devices (P=17)."""
    SIZE = 64
    ROUNDS = 2 * (17 - 1) + 1  # 33 signal rows

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(self, data: pld.DistributedTensor[[SIZE], pl.FP32]):
            return data

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            data_buf = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(ROUNDS * 17 * pl.INT32.get_byte())
            data = pld.window(data_buf, [SIZE], dtype=pl.FP32)
            signal = pld.window(signal_buf, [ROUNDS, 17], dtype=pl.INT32)
            self.chip_orch(data, device=0)
            self.chip_orch(data, device=1)
            self.chip_orch(data, device=2)
            self.chip_orch(data, device=3)
            self.chip_orch(data, device=4)
            self.chip_orch(data, device=5)
            self.chip_orch(data, device=6)
            self.chip_orch(data, device=7)
            self.chip_orch(data, device=8)
            self.chip_orch(data, device=9)
            self.chip_orch(data, device=10)
            self.chip_orch(data, device=11)
            self.chip_orch(data, device=12)
            self.chip_orch(data, device=13)
            self.chip_orch(data, device=14)
            self.chip_orch(data, device=15)
            self.chip_orch(data, device=16)
            pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode="ring")
            return 0

    program = passes.materialize_comm_domain_scopes()(P)
    with pytest.raises(ValueError, match=r"16 or fewer"):
        passes.lower_host_tensor_collectives()(program)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
