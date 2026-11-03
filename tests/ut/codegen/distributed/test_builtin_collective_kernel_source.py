# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F722, F821

"""The HOST and CHIP rails must render one shared builtin collective kernel.

``pld.tensor.all_to_all_v`` has two managed lowering rails. Both render their
AIV kernel from the *same* template
(``pypto/runtime/builtins/collectives/all_to_all_v/templates/kernel.cpp.in``):

- HOST/L3 — ``LowerHostTensorCollectives`` emits ``builtin.tensor.all_to_all_v``
  and the distributed codegen renders the template into
  ``next_levels/builtin.tensor.all_to_all_v__fp32/``.
- CHIP/L2 — ``LowerL2TensorCollectives`` synthesizes an AIV function carrying
  ``builtin_template_dir`` / ``builtin_template_vars``, and the PTO backend
  renders the same template into the chip sub-build.

``dtype_cpp`` is the only substitution either rail makes and both give it the
same value, so the rendered sources must be **byte-identical**. That is the
property under test: it is what makes "one transport implementation, two rails"
true rather than aspirational. A change that reintroduces a per-rail difference
in the kernel body — an extra template variable, a rail-specific ``#ifdef`` —
fails here, instead of silently forking the transport into two implementations
that can drift apart on the wire.

This is a compile-only check: no device is dispatched and ptoas is skipped, so
it belongs here rather than in the hardware ST. (It lived in the ST briefly;
compiling a second distributed program inside a test that then dispatches its
own destabilised the run and cost device force-resets.)
"""

import re

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import ir
from pypto.ir import DistributedConfig
from pypto.pypto_core import passes


@pytest.fixture(autouse=True)
def _property_verification_only():
    """Property verification without the print -> parse roundtrip check.

    These tests assert rendered kernel artifacts. Complete distributed IR text
    roundtrips are checked with the default instrument in the materialization
    and host-collective transform tests.
    """
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        yield


SIZE = 64
NR = 2
MAX_RECV = 4
TOTAL = NR * MAX_RECV


def _dtype_of(name: str):
    return {"fp32": pl.FP32, "int8": pl.INT8}[name]


def _variant_of(name: str) -> str:
    return f"builtin.tensor.all_to_all_v__{name}"


def _l2_kernel_of(name: str) -> str:
    return f"__builtin_all_to_all_v__{name}.cpp"


def _build_chip_rail_program(dtype_name: str = "fp32", core_num: int = 1, signal_stride: int = 1):
    """CHIP/L2 rail: the collective written in a CHIP orchestration body.

    Identical to the HOST program below except for *where* the collective is
    written — same five windows, same host allocation, same comm domain. That
    one difference is the whole point of the comparison.

    ``core_num`` / ``signal_stride`` choose the requested block limit and the
    signal's per-block lane count ``S`` (the signal is ``[NR, S]``).
    """
    dtype = _dtype_of(dtype_name)
    payload_bytes = TOTAL * SIZE * dtype.get_byte()
    i32_bytes = NR * pl.INT32.get_byte()
    signal_bytes = NR * signal_stride * pl.INT32.get_byte()

    @pl.program
    class ChipRail:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_pipeline(
            self,
            stage: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], dtype]],
            data: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], dtype]],
            signal: pl.InOut[pld.DistributedTensor[[NR, signal_stride], pl.INT32]],
            counts: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            recv: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[TOTAL, SIZE], dtype]:
            return pld.tensor.all_to_all_v(stage, data, signal, counts, recv, core_num=core_num)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            stage_buf = pld.alloc_window_buffer(payload_bytes)
            data_buf = pld.alloc_window_buffer(payload_bytes)
            signal_buf = pld.alloc_window_buffer(signal_bytes)
            counts_buf = pld.alloc_window_buffer(i32_bytes)
            recv_buf = pld.alloc_window_buffer(i32_bytes)

            for r in pl.range(pld.world_size()):
                stage = pld.window(stage_buf, [TOTAL, SIZE], dtype=dtype)
                data = pld.window(data_buf, [TOTAL, SIZE], dtype=dtype)
                sig = pld.window(signal_buf, [NR, signal_stride], dtype=pl.INT32)
                counts = pld.window(counts_buf, [NR, 1], dtype=pl.INT32)
                recv = pld.window(recv_buf, [NR, 1], dtype=pl.INT32)
                self.chip_pipeline(stage, data, sig, counts, recv, device=r)

    return ChipRail


def _build_runtime_core_num_program(signal_stride: int = 8):
    """CHIP rail with ``core_num`` as a runtime scalar rather than a literal.

    Same five windows as ``_build_chip_rail_program``; the only difference is
    that the block limit arrives at run time, which is what RFC #2521 O2 adds.
    """
    dtype = pl.FP32
    payload_bytes = TOTAL * SIZE * dtype.get_byte()
    i32_bytes = NR * pl.INT32.get_byte()
    signal_bytes = NR * signal_stride * pl.INT32.get_byte()

    @pl.program
    class RuntimeCoreNum:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_pipeline(
            self,
            stage: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], dtype]],
            data: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], dtype]],
            signal: pl.InOut[pld.DistributedTensor[[NR, signal_stride], pl.INT32]],
            counts: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            recv: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            core_num: pl.Scalar[pl.INDEX],
        ) -> pld.DistributedTensor[[TOTAL, SIZE], dtype]:
            return pld.tensor.all_to_all_v(stage, data, signal, counts, recv, core_num=core_num)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self, core_num: pl.Scalar[pl.INDEX]):
            stage_buf = pld.alloc_window_buffer(payload_bytes)
            data_buf = pld.alloc_window_buffer(payload_bytes)
            signal_buf = pld.alloc_window_buffer(signal_bytes)
            counts_buf = pld.alloc_window_buffer(i32_bytes)
            recv_buf = pld.alloc_window_buffer(i32_bytes)

            for r in pl.range(pld.world_size()):
                stage = pld.window(stage_buf, [TOTAL, SIZE], dtype=dtype)
                data = pld.window(data_buf, [TOTAL, SIZE], dtype=dtype)
                sig = pld.window(signal_buf, [NR, signal_stride], dtype=pl.INT32)
                counts = pld.window(counts_buf, [NR, 1], dtype=pl.INT32)
                recv = pld.window(recv_buf, [NR, 1], dtype=pl.INT32)
                self.chip_pipeline(stage, data, sig, counts, recv, core_num, device=r)

    return RuntimeCoreNum


def _build_host_rail_program(dtype_name: str = "fp32"):
    """HOST/L3 rail: the collective written in the host orchestrator.

    Mirrors the CHIP program: same five windows, same one dispatch per rank.
    The dispatched orchestration here does *not* hold the collective — that is
    the single difference under test — but it must still exist, because
    ``MaterializeCommDomainScopes`` infers a window's comm domain from the
    dispatches that consume it.
    """
    dtype = _dtype_of(dtype_name)
    payload_bytes = TOTAL * SIZE * dtype.get_byte()
    i32_bytes = NR * pl.INT32.get_byte()

    @pl.program
    class HostRail:
        @pl.function(type=pl.FunctionType.InCore)
        def touch_step(
            self,
            stage: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], dtype]],
            data: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], dtype]],
            signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            counts: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            recv: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[NR, 1], pl.INT32]:
            """Touch every window, so each one has an inferable comm domain."""
            row = pl.load(stage, [0, 0], [1, SIZE])
            data = pl.store(row, [0, 0], data)
            # Read-back-and-write rather than a constant: the point is only to
            # give each window a consuming dispatch, and a scalar constant would
            # need a dtype-matched Scalar the DSL does not mint from a literal.
            for d in pl.range(NR):
                pl.write(counts, [d, 0], pl.read(counts, [d, 0]))
                pl.write(recv, [d, 0], pl.read(recv, [d, 0]))
                pl.write(signal, [d, 0], pl.read(signal, [d, 0]))
            return counts

        @pl.function(type=pl.FunctionType.Orchestration)
        def touch_orch(
            self,
            stage: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], dtype]],
            data: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], dtype]],
            signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            counts: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            recv: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[NR, 1], pl.INT32]:
            return self.touch_step(stage, data, signal, counts, recv)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            stage_buf = pld.alloc_window_buffer(payload_bytes)
            data_buf = pld.alloc_window_buffer(payload_bytes)
            signal_buf = pld.alloc_window_buffer(i32_bytes)
            counts_buf = pld.alloc_window_buffer(i32_bytes)
            recv_buf = pld.alloc_window_buffer(i32_bytes)

            stage = pld.window(stage_buf, [TOTAL, SIZE], dtype=dtype)
            data = pld.window(data_buf, [TOTAL, SIZE], dtype=dtype)
            sig = pld.window(signal_buf, [NR, 1], dtype=pl.INT32)
            counts = pld.window(counts_buf, [NR, 1], dtype=pl.INT32)
            recv = pld.window(recv_buf, [NR, 1], dtype=pl.INT32)

            for r in pl.range(pld.world_size()):
                r_stage = pld.window(stage_buf, [TOTAL, SIZE], dtype=dtype)
                r_data = pld.window(data_buf, [TOTAL, SIZE], dtype=dtype)
                r_sig = pld.window(signal_buf, [NR, 1], dtype=pl.INT32)
                r_counts = pld.window(counts_buf, [NR, 1], dtype=pl.INT32)
                r_recv = pld.window(recv_buf, [NR, 1], dtype=pl.INT32)
                self.touch_orch(r_stage, r_data, r_sig, r_counts, r_recv, device=r)

            pld.tensor.all_to_all_v(stage, data, sig, counts, recv)

    return HostRail


def _compile(program, tmp_path, name):
    return ir.compile(
        program,
        output_dir=str(tmp_path / name),
        platform="a2a3",
        skip_ptoas=True,
        dump_passes=False,
        distributed_config=DistributedConfig(device_ids=list(range(NR)), num_sub_workers=0),
    )


def _sole_file(directory, pattern):
    matches = sorted(directory.glob(pattern))
    assert len(matches) == 1, f"expected exactly one {pattern} under {directory}, got {matches}"
    return matches[0]


@pytest.mark.parametrize("dtype_name", ["fp32", "int8"])
def test_both_rails_render_a_byte_identical_builtin_kernel(tmp_path, dtype_name):
    """The two rails' rendered kernel sources must match exactly."""
    chip = _compile(_build_chip_rail_program(dtype_name), tmp_path, f"chip_{dtype_name}")
    host = _compile(_build_host_rail_program(dtype_name), tmp_path, f"host_{dtype_name}")

    chip_kernel = (
        chip.output_dir / "next_levels" / "chip_pipeline" / "kernels" / "aiv" / _l2_kernel_of(dtype_name)
    )
    assert chip_kernel.is_file(), f"expected the rendered CHIP kernel at {chip_kernel}"
    host_kernel = _sole_file(
        host.output_dir / "next_levels" / _variant_of(dtype_name) / "kernels" / "aiv", "*.cpp"
    )

    assert host_kernel.read_text() == chip_kernel.read_text(), (
        "HOST and CHIP rails must render the same builtin kernel source; a difference means "
        "the two rails no longer share one transport implementation"
    )


def _chip_orchestration_sources(output_dir) -> str:
    """Every emitted CHIP-orchestration C++ source of ``chip_pipeline``, concatenated."""
    sources = sorted((output_dir / "next_levels" / "chip_pipeline").rglob("*.cpp"))
    orchestration = [s for s in sources if "orchestration" in s.parts]
    assert orchestration, f"expected an emitted orchestration source under {output_dir}"
    return "\n".join(s.read_text() for s in orchestration)


def test_chip_rail_multi_block_request_emits_the_launch_spec(tmp_path):
    """``core_num=2`` over two ranks launches a two-block, sync-start gang.

    The pass folds ``B = CalAllToAllVBlocks(2, 2) = 2`` and the orchestration
    codegen renders it onto the kernel's task, next to ``require_sync_start``.
    """
    chip = _compile(_build_chip_rail_program(core_num=2, signal_stride=2), tmp_path, "chip_multi")
    source = _chip_orchestration_sources(chip.output_dir)
    assert re.search(r"\.launch_spec\.\w+\(2\);", source), source
    assert ".launch_spec.set_require_sync_start(true);" in source, source


def test_runtime_core_num_computes_the_width_from_the_comm_context(tmp_path):
    """``B`` is emitted arithmetic over a rank count read off the CommCtx.

    RFC #2521 frozen item #3 fixes ``B = CalAllToAllVBlocks(P, L)`` as a pure
    function of ``(P, L)``. With ``L`` only known at run time the compiler
    cannot fold it, so the same formula is emitted branch-free -- the
    expression codegen has no ternary, and ``min``/``max`` alone cannot
    express ``L < P ? L : (L / P) * P``.
    """
    chip = _compile(_build_runtime_core_num_program(), tmp_path, "chip_runtime")
    source = _chip_orchestration_sources(chip.output_dir)
    rank_read = re.search(r"int32_t (\w+) = \*reinterpret_cast<const int32_t\*>\(\w+ \+ 20\);", source)
    assert rank_read, source
    assert re.search(r"= \(\w+ % \w+\);", source), source
    assert re.search(r"= static_cast<int64_t>\(\(\w+ <= \w+\)\);", source), source
    assert re.search(r"= \(\w+ - \(\w+ \* \w+\)\);", source), source


def test_runtime_core_num_launch_width_is_clamped_and_unwrapped(tmp_path):
    """The width is clamped inline so the dispatch keeps its straight line.

    ``LegalizeSpmdLaunches`` wraps any launch it cannot prove positive in
    ``if (B > 0)``. The guards below are emitted *at* the dispatch, so that
    wrapper would hide them from a non-positive request -- the one input they
    exist to catch. Range analysis proves ``max(x, 1) >= 1`` from the
    expression, so the clamp rides the attr and no wrapper is emitted.
    """
    chip = _compile(_build_runtime_core_num_program(), tmp_path, "chip_runtime_clamp")
    source = _chip_orchestration_sources(chip.output_dir)
    assert re.search(r"\.launch_spec\.\w+\(std::max<int64_t>\(\w+, 1\)\);", source), source
    assert ".launch_spec.set_require_sync_start(true);" in source, source
    # A wrapper would materialise the count as ``spmd_core_num`` and guard the
    # whole dispatch. That name appears nowhere else in the emitted source, so
    # its absence is what proves the guards below stay reachable for a
    # non-positive request. ``if (`` alone is not enough: the guard chain itself
    # starts with one.
    assert "spmd_core_num" not in source, source
    dispatch = source[: source.index("rt_submit_aiv_task")]
    assert "rt_available_aiv_count" in dispatch, dispatch


def test_runtime_core_num_emits_the_runtime_checks(tmp_path):
    """Every RFC #2521 §12.3 failure the orchestration can see is reported.

    One chained ``else if`` so a dispatch names its first fault rather than
    four, and no ``return`` after the fatal: the runtime latches the error and
    no-ops every later submit, and the dispatch may sit inside a scope.
    """
    chip = _compile(_build_runtime_core_num_program(signal_stride=8), tmp_path, "chip_runtime_guards")
    source = _chip_orchestration_sources(chip.output_dir)
    kernel = "__builtin_all_to_all_v__fp32"
    messages = (
        "core_num must be positive",
        "communicator rank count must be positive",
        f"signal rows ({NR}) must equal the communicator rank count",
        "signal stride (8) is smaller than the admitted block count",
        "admitted block count exceeds the available AIV count",
    )
    for message in messages:
        assert f'rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "{kernel}: {message}' in source, source
    guard_block = source[source.index("computes its launch width") :]
    guard_block = guard_block[: guard_block.index("CoreTaskArgs")]
    assert guard_block.count("} else if (") == len(messages) - 1, guard_block
    assert "return;" not in guard_block, guard_block


def test_runtime_core_num_cannot_divide_by_a_zero_rank_count(tmp_path):
    """The width arithmetic is total, so a bad rank count reaches its guard.

    The arithmetic runs in the prelude and the guards run at the dispatch, so a
    communicator reporting ``P = 0`` would hit ``% P`` before anything could
    reject it — undefined behaviour in the generated C++ instead of the explicit
    error RFC #2521 §12.3 asks for. The divisor is floored to 1 and the real
    ``P`` is kept for the check, which still rejects the run because ``NR`` is
    deducer-enforced positive.
    """
    chip = _compile(_build_runtime_core_num_program(), tmp_path, "chip_runtime_div")
    source = _chip_orchestration_sources(chip.output_dir)
    divisor = re.search(r"int64_t (\w+) = std::max<int64_t>\((\w+), 1\);", source)
    assert divisor, source
    floored, raw = divisor.group(1), divisor.group(2)
    # Every division and comparison uses the floored copy...
    assert f"% {floored})" in source, source
    assert f"({floored} <= " in source, source
    assert f"% {raw})" not in source, source
    # ...while the guard tests the value the communicator actually reported.
    assert f"if ({raw} <= 0)" in source or f"else if ({raw} <= 0)" in source, source
    kernel = "__builtin_all_to_all_v__fp32"
    assert f'"{kernel}: communicator rank count must be positive"' in source, source


def test_literal_core_num_emits_no_runtime_checks(tmp_path):
    """A literal request is checked at compile time and costs nothing at run time."""
    chip = _compile(_build_chip_rail_program(core_num=2, signal_stride=2), tmp_path, "chip_literal_guards")
    source = _chip_orchestration_sources(chip.output_dir)
    assert "computes its launch width at run time" not in source, source
    assert "rt_available_aiv_count" not in source, source


def test_runtime_core_num_still_renders_the_shared_kernel(tmp_path):
    """One binary serves any runtime ``L`` (RFC #2521 §12.1).

    The width lives in the launch spec, so the kernel is the same bytes the
    literal rail renders -- which is also the tripwire for a change that tried
    to pass the width as a kernel argument.
    """
    runtime = _compile(_build_runtime_core_num_program(), tmp_path, "chip_runtime_kernel")
    literal = _compile(_build_chip_rail_program(core_num=2, signal_stride=2), tmp_path, "chip_literal_kernel")
    runtime_kernel = (
        runtime.output_dir / "next_levels" / "chip_pipeline" / "kernels" / "aiv" / _l2_kernel_of("fp32")
    )
    literal_kernel = (
        literal.output_dir / "next_levels" / "chip_pipeline" / "kernels" / "aiv" / _l2_kernel_of("fp32")
    )
    assert runtime_kernel.read_text() == literal_kernel.read_text()


def test_chip_rail_single_block_request_emits_no_launch_spec(tmp_path):
    """``core_num=1`` is unchanged: a plain single-block task, no gang launch."""
    chip = _compile(_build_chip_rail_program(core_num=1), tmp_path, "chip_single")
    assert ".launch_spec." not in _chip_orchestration_sources(chip.output_dir)


def test_multi_block_chip_rail_still_renders_the_shared_kernel(tmp_path):
    """A multi-block request must not fork the kernel from the HOST rail's.

    The launch spec is a call attr, not a kernel parameter, so the rendered
    source stays byte-identical — the tripwire for a future "just pass it as an
    argument" change that would shift the CommContext off ``args[5]``.
    """
    chip = _compile(_build_chip_rail_program(core_num=2, signal_stride=2), tmp_path, "chip_multi_kernel")
    host = _compile(_build_host_rail_program(), tmp_path, "host_multi_kernel")
    chip_kernel = (
        chip.output_dir / "next_levels" / "chip_pipeline" / "kernels" / "aiv" / _l2_kernel_of("fp32")
    )
    host_kernel = _sole_file(
        host.output_dir / "next_levels" / _variant_of("fp32") / "kernels" / "aiv", "*.cpp"
    )
    assert chip_kernel.is_file(), f"expected the rendered CHIP kernel at {chip_kernel}"
    assert host_kernel.read_text() == chip_kernel.read_text()


def test_chip_rail_emits_no_builtin_chip_dispatch(tmp_path):
    """The CHIP rail keeps the collective inside the caller's own pipeline.

    A ``next_levels/builtin.tensor.*`` directory would mean the HOST fan-out
    rail ran, i.e. the collective became a nested L2 orchestration task instead
    of one AIV task of ``chip_pipeline``.
    """
    chip = _compile(_build_chip_rail_program(), tmp_path, "chip")

    next_levels = sorted(p.name for p in (chip.output_dir / "next_levels").iterdir())
    assert next_levels == ["chip_pipeline"], next_levels


def test_host_rail_still_emits_its_builtin_chip_dispatch(tmp_path):
    """The HOST rail is unchanged — the fan-out dispatch is still its shape.

    Pins the contrast the CHIP rail is defined against, so a regression that
    silently routed HOST calls through the CHIP rail would not pass unnoticed.
    """
    host = _compile(_build_host_rail_program(), tmp_path, "host")

    next_levels = sorted(p.name for p in (host.output_dir / "next_levels").iterdir())
    assert _variant_of("fp32") in next_levels, next_levels


def _build_indexed_rail_program(*, host: bool, dtype_name: str):
    dtype = _dtype_of(dtype_name)
    input_bytes = 13 * SIZE * dtype.get_byte()
    target_bytes = TOTAL * SIZE * dtype.get_byte()

    @pl.program
    class IndexedRail:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_pipeline(
            self,
            inp: pld.DistributedTensor[[13, SIZE], dtype],
            data: pl.InOut[pld.DistributedTensor[[TOTAL, SIZE], dtype]],
            signal: pl.InOut[pld.DistributedTensor[[NR, NR], pl.INT32]],
            counts: pld.DistributedTensor[[NR, 1], pl.INT32],
            recv: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            index: pld.DistributedTensor[[NR, MAX_RECV], pl.INT32],
        ) -> pld.DistributedTensor[[TOTAL, SIZE], dtype]:
            return pld.tensor.indexed_all_to_all_v(inp, data, signal, counts, recv, index, core_num=NR)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self):
            inp_buf = pld.alloc_window_buffer(input_bytes)
            data_buf = pld.alloc_window_buffer(target_bytes)
            sig_buf = pld.alloc_window_buffer(NR * NR * 4)
            counts_buf = pld.alloc_window_buffer(NR * 4)
            recv_buf = pld.alloc_window_buffer(NR * 4)
            index_buf = pld.alloc_window_buffer(NR * MAX_RECV * 4)
            for r in pl.range(pld.world_size()):
                inp = pld.window(inp_buf, [13, SIZE], dtype=dtype)
                data = pld.window(data_buf, [TOTAL, SIZE], dtype=dtype)
                sig = pld.window(sig_buf, [NR, NR], dtype=pl.INT32)
                counts = pld.window(counts_buf, [NR, 1], dtype=pl.INT32)
                recv = pld.window(recv_buf, [NR, 1], dtype=pl.INT32)
                index = pld.window(index_buf, [NR, MAX_RECV], dtype=pl.INT32)
                self.chip_pipeline(inp, data, sig, counts, recv, index, device=r)

    if not host:
        return IndexedRail
    # Keep the same windows/domain registration while moving the exchange to HOST.
    source = ir.python_print(IndexedRail, format=False)
    source, replacements = re.subn(
        r"return pld\.tensor\.indexed_all_to_all_v\(inp, data, signal, counts, recv, index, core_num=2\)",
        "return data",
        source,
    )
    assert replacements == 1, source
    # This registration dispatch only reads its windows after moving the exchange.
    source = re.sub(r"pl\.InOut\[(pld\.DistributedTensor\[\[[^]]+\], pl\.\w+\])\]", r"\1", source)
    # HOST windows must be defined outside the registration loop to stay in scope.
    lines = source.splitlines()
    windows = [
        line[4:] for line in lines if line.startswith("            ") and " = pld.tensor.window(" in line
    ]
    lines = [
        line for line in lines if not (line.startswith("            ") and " = pld.tensor.window(" in line)
    ]
    loop = next(i for i, line in enumerate(lines) if line.startswith("        for r in "))
    lines[loop:loop] = windows
    source = "\n".join(lines)
    # The HOST call belongs inside host_orch, after its dispatch loop.
    source = source.rstrip() + (
        "\n        pld.tensor.indexed_all_to_all_v(inp, data, sig, counts, recv, index, core_num=2)\n"
    )
    return pl.parse_program(source)


@pytest.mark.parametrize("dtype_name", ["fp32", "int8"])
def test_indexed_l2_compiles_six_operand_kernel(tmp_path, dtype_name):
    result = ir.compile(
        _build_indexed_rail_program(host=False, dtype_name=dtype_name),
        output_dir=str(tmp_path / dtype_name),
        platform="a5",
        skip_ptoas=True,
        dump_passes=False,
        distributed_config=DistributedConfig(device_ids=list(range(NR)), num_sub_workers=0),
    )
    kernel = (
        result.output_dir
        / "next_levels"
        / "chip_pipeline"
        / "kernels"
        / "aiv"
        / f"__builtin_indexed_all_to_all_v__{dtype_name}.cpp"
    )
    assert kernel.is_file()


@pytest.mark.parametrize("dtype_name", ["fp32", "int8"])
def test_indexed_rejects_host_call(tmp_path, dtype_name):
    with pytest.raises(ValueError, match="L2 orchestration.*HOST is unsupported"):
        ir.compile(
            _build_indexed_rail_program(host=True, dtype_name=dtype_name),
            output_dir=str(tmp_path / dtype_name),
            platform="a5",
            skip_ptoas=True,
            distributed_config=DistributedConfig(device_ids=list(range(NR)), num_sub_workers=0),
        )


_OPERANDS = ("input", "target", "signal", "send_counts", "recv_counts")


def _assert_operand_guards(source: str, kernel: str, operands: tuple[str, ...]) -> None:
    """One chained ``is_contiguous`` check per operand, in order, ahead of the submit."""
    keywords = re.findall(r"(if|\} else if) \(!\w+\.is_contiguous\) \{", source)
    assert keywords == ["if"] + ["} else if"] * (len(operands) - 1), source
    for operand in operands:
        fatal = f"{kernel}: {operand} must be a contiguous row-major view"
        assert f'rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "{fatal}");' in source, source
    assert source.rindex(".is_contiguous") < source.index("rt_submit_aiv_task"), source


def test_chip_rail_checks_operand_contiguity_before_the_submit(tmp_path):
    """RFC #2521 O1 criterion 7: the runtime backstop. The orchestration checks each
    operand's ``is_contiguous`` flag ahead of the collective's submit, latching
    ``SIMPLER_ERROR_INVALID_ARGS`` on the first violation (which also turns the submit
    into a no-op). The compile-time proof lives in ``LowerL2TensorCollectives``."""
    chip = _compile(_build_chip_rail_program(), tmp_path, "chip_contig")
    _assert_operand_guards(
        _chip_orchestration_sources(chip.output_dir), "__builtin_all_to_all_v__fp32", _OPERANDS
    )


def test_indexed_rail_checks_all_six_operands_before_the_submit(tmp_path):
    """The indexed kernel reads `index` as dense row-major too, so it gets a sixth check."""
    result = ir.compile(
        _build_indexed_rail_program(host=False, dtype_name="fp32"),
        output_dir=str(tmp_path / "indexed_contig"),
        platform="a5",
        skip_ptoas=True,
        dump_passes=False,
        distributed_config=DistributedConfig(device_ids=list(range(NR)), num_sub_workers=0),
    )
    _assert_operand_guards(
        _chip_orchestration_sources(result.output_dir),
        "__builtin_indexed_all_to_all_v__fp32",
        (*_OPERANDS, "index"),
    )


def test_literal_core_num_reports_requested_and_admitted_widths(tmp_path):
    """RFC #2521 §13.3: ``L`` and ``B`` are reported separately, HOST-line shape.

    ``L=9`` over ``P=2`` folds to ``B=8``; a line that said 9 would be exactly
    the failure the RFC's DFX item exists to prevent.
    """
    chip = _compile(_build_chip_rail_program(core_num=9, signal_stride=8), tmp_path, "chip_dfx_literal")
    source = _chip_orchestration_sources(chip.output_dir)
    assert (
        "all_to_all_v: requested_core_num=%d launched_core_num=%d active_lanes=%d nranks=%d lanes_per_peer=%d"
    ) in source, source
    assert "const int64_t a2av_dfx_launched = 8;" in source, source
    assert "const int64_t a2av_dfx_nranks = 2;" in source, source
    assert "const int64_t a2av_dfx_stride = 8;" in source, source
    assert "static_cast<int32_t>(9)" in source, source


def test_runtime_core_num_reports_the_unclamped_admitted_width(tmp_path):
    """The DFX line reads the raw ``B``, never the ``max(B, 1)`` launch clamp.

    It is the guard chain's final ``else``, so a rejected request reports no
    width at all.
    """
    chip = _compile(_build_runtime_core_num_program(signal_stride=8), tmp_path, "chip_dfx_runtime")
    source = _chip_orchestration_sources(chip.output_dir)
    marker = source.index("// RFC #2521 §13.3")
    # The guard block ends before the task's parameters are declared, so this
    # slice contains the guards' final else -- not the launch spec, which does
    # carry the clamp.
    dfx = source[marker : source.index("CoreTaskArgs", marker)]
    assert "a2av_dfx_launched = a2av_blocks_" in dfx, dfx
    assert "std::max" not in dfx, dfx
    assert dfx.count("LOG_TIMING") == 1, dfx
    assert source[:marker].rstrip().endswith("} else {"), source[:marker][-300:]


def test_single_block_request_reports_the_width_without_a_launch_spec(tmp_path):
    """``L=1`` has no gang launch but is still one of §13.3's cases."""
    chip = _compile(_build_chip_rail_program(core_num=1), tmp_path, "chip_dfx_single")
    source = _chip_orchestration_sources(chip.output_dir)
    assert "const int64_t a2av_dfx_launched = 1;" in source, source
    assert "static_cast<int32_t>(1)" in source, source
    assert ".launch_spec." not in source, source


def test_ordinary_chip_tasks_get_no_contiguity_check(tmp_path):
    """Only the builtin kernel opts in; an ordinary InCore task is emitted as before.

    The HOST rail's ``touch_orch`` dispatches a plain DSL kernel over the same
    five windows. Its orchestration must carry no operand checks — the attribute,
    not the operand types, is what decides.
    """
    host = _compile(_build_host_rail_program(), tmp_path, "host_no_contig")
    sources = sorted((host.output_dir / "next_levels" / "touch_orch").rglob("*.cpp"))
    orchestration = [s for s in sources if "orchestration" in s.parts]
    assert orchestration, f"expected touch_orch's orchestration source under {host.output_dir}"
    for path in orchestration:
        assert "is_contiguous" not in path.read_text(), path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
