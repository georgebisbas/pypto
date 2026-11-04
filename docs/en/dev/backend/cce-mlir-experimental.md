# Experimental PyPTO to cce-mlir bridge

This page documents the cce-mlir compiler route on the
`feat/cce-mlir-experimental` branch. It is a standalone route: PyPTO's
production `ir.compile` path, `BackendType`, runtime and PTOAS are unchanged.

## What is implemented

`pypto.backend.cce_mlir` lowers PyPTO's **Buffer-stage InCore IR** into
cce-mlir's PTO/NPU MLIR and runs cce-mlir's pipelines in process:

```text
PyPTO @pl.jit kernel
  -> pass pipeline with enable_buffer_ir
  -> Buffer-stage InCore function
  -> typed CceKernel IR          (pypto.backend.cce_mlir.analysis)
  -> in-memory MLIR module       (dialect bindings, no text templates)
  -> cce-mlir PassManager        (no cce-opt subprocess)
  -> ET_REL device object + host driver
```

Every emitted value is derived from PyPTO IR or registered backend metadata;
the lowering refuses anything it cannot derive.

## Package layout

| Module | Responsibility |
| --- | --- |
| `analysis.py` | Buffer IR → typed kernel IR; contract validation |
| `layout.py` | Memory-space and fractal-encoding derivation |
| `kernel_ir.py` | Typed, backend-independent kernel representation |
| `registry.py`, `emitters.py` | Op emitter registry keyed by registered PyPTO op name |
| `emit.py` | Typed IR → `cce_mlir.ir.Module` through the dialect bindings |
| `pipeline.py` | In-process `PassManager` runs and device-object extraction |
| `elf.py` | ELF64 inspection (relocatable check, symbols, entry point) |
| `adapter.py` | `kernel_entry` C++ adapter generation |
| `manifest.py`, `driver.py` | Reproducibility manifest and standalone compile driver |

## Derivation rules

| Output | Source |
| --- | --- |
| Memory tier (`gm`, `ub`, `l1`, `l0a`, `l0b`, `l0c`) | `BufferType.memory_space` |
| Tensor encoding | cce-mlir layout factories: NZ staging `{16, 256/bits}` perm `{1,0}`, ZZ L0A, ZN L0B, fixed 16x16 L0C accumulator, dense GM/UB. Cross-checked against `BufferType.blayout/slayout/fractal` |
| Shapes, byte sizes | `BufferType.shape` and the dtype's byte width |
| Parameter order and direction | `Function.params` + `Function.param_directions` |
| Operand roles (read/write/destination) | `ir.get_op_buffer_arg_effect` |
| Kernel kind | `func_type` (AIC → cube, AIV → vector), cross-checked against allocated tiers |
| Target | `BackendHandler.get_pto_target_arch()` (`a2a3` → `a3`) |
| Symbols | `Function.name`. The standalone runner keeps cce-mlir's `<name>__kernel0`; the PyPTO runtime product uses the bare `<name>` |

No fixture shape, byte count, layout or symbol is written literally.

## Supported contract

- Cube: `buffer.matmul` and `buffer.matmul_acc` with `Left × Right → Acc`,
  FP16/BF16 operands and an FP32 accumulator.
- Vector: `buffer.add/mul/sub/div/maximum/minimum` and
  `buffer.abs/exp/sqrt/neg/relu/log/recip` over UB tiles.
- Transfers: `buffer.load` (GM → L1/UB), `buffer.store` (L0C/UB → GM),
  `buffer.copy` (L1 → L0A/L0B, UB → UB).
- Static rank-2 shapes, full-window transfers, A3, one block.
- Static-bound `pl.range` loops whose body operates on fixed operands.
  PyPTO's Buffer IR keeps loop state in buffers, so each tensor written in the
  body is threaded as an `scf.for` iter-arg and the post-loop value is the
  loop's result.

Refused with a typed diagnostic: windowed or partial transfers, dynamic valid
shapes, scalar kernel parameters, scalar elementwise variants, non-constant
loop bounds, loop-carried `pl.yield` values, `while`/`if`, dynamic tile offsets
inside loops, Bias/A5 scale spaces, mixed cube/vector kernels, SPMD block
queries, `buffer.extract/subview/reshape/set_validshape`, and any operator
without an emitter registration.

## Performance

The two routes are compared in
`tests/st/runtime/cce_mlir/test_cce_mlir_performance.py`, which registers each
compiled program once on a `ChipWorker`, dispatches over device tensors, and
reads the AICore task from chip-swimlane level 1. This avoids the per-call
artifact assembly that otherwise dominates `CompiledProgram` dispatch.

For the same kernels the routes are within roughly 10% of each other, and the
sign depends on the workload: cce-mlir was ~9% slower for a single 128³ MMAD
and ~7% faster for a 65-MMAD serial accumulation. Both compile to the same
hardware intrinsics (GM→L1, L1→L0A/L0B, MMAD, L0C→GM); the fixed per-task
start-up and DMA cost is several microseconds, so codegen differences only
become visible once the MMAD chain is long. The test enforces a 2× parity bound
rather than a target number.

## Selecting the route

The route is opt-in and leaves the PTOAS default untouched:

```python
from pypto.runtime import ExecutionMode, RunConfig

compiled = kernel.compile(
    config=RunConfig(kernel_compiler="cce_mlir", execution_mode=ExecutionMode.ONBOARD)
)
```

Any unmodified example can select the route from the environment instead of
editing its `RunConfig`:

```bash
PYPTO_KERNEL_COMPILER=cce_mlir python examples/beginner/05_matmul.py
```

The variable supplies the default only; an explicit `kernel_compiler=` argument
wins. Selecting `cce_mlir` also makes `ONBOARD` the default execution mode,
because the route cannot target the simulator. On a device host, run the
example through the queue as usual (`task-submit --device auto ... --run
'PYPTO_KERNEL_COMPILER=cce_mlir python examples/...'`).

`ir.compile(program, kernel_compiler="cce_mlir")` accepts the same option. The
route requires an onboard `a2a3` target, ignores the PTOAS-only options, and
emits a runtime artifact whose kernel entries carry `link_objects`:

```text
kernel_config.py                      per-kernel link_objects -> .cce.o
orchestration/<orch>.cpp              existing orchestration codegen
kernels/<aic|aiv>/<name>.cpp          kernel_entry adapter
kernels/<aic|aiv>/<name>.cce.o        cce-mlir device object
cce_mlir/<name>.mlir                  module as emitted
cce_mlir/<name>.lowered.mlir          module after cce-mlir's pipeline
cce_mlir/<name>.manifest.json         contract, revisions, object digest
```

At `DEBUG`, a successful compile names the resolved artifact root and the
linked device objects, so a caller that discarded the returned program can find
them:

```text
2026-11-03 15:20:11.123 D | cce-mlir artifacts written to /abs/out (device objects: /abs/out/kernels/aic/<name>.cce.o)
```

Set `PYPTO_LOG_LEVEL=debug` or call `set_log_level(LogLevel.DEBUG)`; the INFO
default is unchanged.

## Artifacts

| File | Meaning |
| --- | --- |
| `<kernel>.mlir` | Module constructed through cce-mlir's bindings |
| `<kernel>.lowered.mlir` | Module after `npu-lower-pto-to-llvm-pipeline` |
| `<kernel>.o` | Extracted `ET_REL` device object |
| `<kernel>.manifest.json` | Derived contract, revisions, object digest, symbols, queue task id |
| `report.json` | Compile status and diagnostics |

## Reproduce with Docker

Build from the PyPTO repository root (ARM64 Ascend host):

```bash
# One-time prerequisites: log in to the private registry and verify SSH access.
docker login registry.gitlab.huaweirc.ch
git ls-remote git@ssh.gitlab.huaweirc.ch:zrc-von-neumann-lab/skyline-software/cce-mlir.git HEAD
test -S "$SSH_AUTH_SOCK"
ssh-add -l
mkdir -p artifacts

docker build \
  --ssh cce_mlir_ssh="$SSH_AUTH_SOCK" \
  --secret id=cce_mlir_known_hosts,src="$HOME/.ssh/known_hosts" \
  --build-arg CCE_MLIR_IMAGE=registry.gitlab.huaweirc.ch/zrc-von-neumann-lab/skyline-software/cce-mlir/ci/cce-mlir-dev:npu-aarch64-latest \
  --build-arg CANN_ARCH=aarch64-linux \
  -f docker/pypto_cce_mlir.Dockerfile \
  -t pypto-cce-mlir:910b \
  .
```

Compile (no device needed):

```bash
docker run --rm \
  -v "$PWD/artifacts:/opt/pypto/artifacts" \
  pypto-cce-mlir:910b \
  compile --output-dir /opt/pypto/artifacts/cce_mlir
```

Run on an NPU **through the queue**, never with a hand-picked device. Start the
image as a queue-joined container per
`pypto-3.0-notes/tools/npu-task-submit-runbook.md`, then submit:

```bash
docker exec <queue-container> task-submit --device auto --device-num 1 \
  --max-time 1800 --timeout 0 \
  --run 'python3 -m examples.cce_mlir run --output-dir /tmp/cce_run'
```

A successful run prints `NPU smoke test passed`. The driver reads the granted
logical device from `TASK_DEVICE` (or the `--device` value that `task-submit`
appends) and never touches `ASCEND_RT_VISIBLE_DEVICES`.

## Runtime ABI status

PyPTO dispatches `kernel_entry(__gm__ int64_t* args)` with one device `Tensor`
descriptor per parameter, while cce-mlir emits `<name>__kernel0` taking raw GM
pointers. `adapter.py` generates the bridge and the runtime's kernel compiler
accepts extra link objects, linking the adapter object **first** so
`kernel_entry` sits at offset 0 of the extracted `.text`.

`ir.compile(kernel_compiler="cce_mlir")` produces the runtime artifact
(`kernel_config.py` with `link_objects`, adapter source, device object,
orchestration) and the link is verified in
`tests/ut/backend/cce_mlir/test_adapter_link.py` (`kernel_entry` at `.text`
offset 0, standalone symbol resolved).

The device object is a **standalone callable function**, not cce-mlir's kernel
entry. cce-mlir's translator stamps the per-function AI-core
`target-cpu`/`target-features` only on functions carrying the kernel marker, so
`pipeline.compile_device_function` keeps that marker through
`npu-lower-pto-to-cce-pipeline` and `npu-lower-cce-to-llvm-device-only-pipeline`,
translates with `cce-translate`, then rewrites the translated LLVM IR: the bare
`<name>` symbol replaces `<name>__kernel0` and the `!hivm.annotations` kernel
metadata is dropped. `bisheng` compiles the result, and the adapter links it
under `kernel_entry`. cce-mlir should eventually grow a native
callable-device-function mode to replace this rewrite.

`tests/st/runtime/cce_mlir/test_cce_mlir_numerical.py` runs cube matmul and
vector add through `CompiledProgram` on an Ascend 910B2 and passes.

## Known limitations

- A3 only; PyPTO's Buffer-IR selection gate is `a2a3`.
- One block per kernel; no SPMD block queries.
- `npu-lower-pto-to-llvm-pipeline` needs the standalone host driver to retain
  the device module, so the device-object path currently emits it too.
- cce-mlir is provided by the Docker image; without it the MLIR and pipeline
  tests are skipped and the typed front end still runs.
