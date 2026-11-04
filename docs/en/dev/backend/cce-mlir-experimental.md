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
  `buffer.abs/exp/sqrt/neg/relu/log/recip` over UB tiles, plus the
  one-tile-plus-scalar variants
  `buffer.adds/subs/muls/divs/maximums/minimums` and `buffer.lrelu`.
- Transfers: `buffer.load` (GM → L1/UB), `buffer.store` (L0C/UB → GM),
  `buffer.copy` (L1 → L0A/L0B, UB → UB), including **windowed** transfers: a
  non-zero offset with extents that fill the on-chip buffer. A window offset
  may be affine in an enclosing loop induction variable (`k * 128`), which is
  how a tiled K reduction reuses one tile. The window is expressed as
  `tensor.extract_slice` of the GM tensor, and cce-mlir folds it into the
  transfer itself (`cce.copy_gm_to_l1_nd2nz`) rather than materializing a copy.
  A partial store writes the window and splices it back with
  `tensor.insert_slice`, which bufferizes to a strided subview that the
  fixpipe writes directly.
- Static rank-2 shapes, A3, one block.
- Static-bound `pl.range` loops whose body operates on fixed operands.
  PyPTO's Buffer IR keeps loop state in buffers, so each tensor written in the
  body is threaded as an `scf.for` iter-arg and the post-loop value is the
  loop's result. Software-pipelined loops (`pl.pipeline`) arrive with the body
  replicated by `LowerPipelineLoops` under the PyPTO planner, so they need no
  extra lowering in the bridge.

Refused with a typed diagnostic: dynamic transfer extents, dynamic valid
shapes, scalar kernel parameters, non-constant loop bounds, loop-carried
`pl.yield` values, `while`/`if`, Bias/A5 scale spaces, mixed cube/vector
kernels, SPMD block queries, `buffer.full` (the cce-mlir build mis-lowers
`pto.expands` for rank-2), `buffer.extract/subview/reshape/set_validshape`, and
any operator without an emitter registration.

## Optimization passes

The route runs cce-mlir's canonical pipelines, which already carry the
allocation and synchronization work:

| Runs | Role |
| --- | --- |
| `pto-split-by-kind`, `pto-lower-cross-core` | cube/vector split and cross-core handoff |
| `pto-propagate-layout` | fold/cancel relayout chains (no-op on today's kernels) |
| `lower-pto-to-cce` | PTO to CCE tensor SSA |
| `cce-insert-sync-dataflow` | derive pipeline barriers from tensor dataflow |
| `cce-verify-flag-pairs`, `cce-allocate-flags` | check and assign event ids |
| `cce-materialize-rank`, `npu-bufferize`, `cce-bufferize`, `npu-void-kernel-abi` | layout rank and bufferization |
| `cce-assign-ub-buffers`, `cce-resolve-onchip-carries`, `cce-allocate-buffer-offsets`, `cce-lower-to-hivm` | UB placement, loop-carried buffers, byte offsets |
| `hivm-lower-to-llvm` + LLVM tail | device LLVM |
| `bisheng -O3` | device-object optimization |

Not used, and why:

| Pass | Why not |
| --- | --- |
| `pto-tile-cube-gemm` | tiles a whole-matrix `linalg`→PTO contraction; the bridge emits an explicit tile chain. It becomes relevant if the route accepts shapes larger than L0 |
| `npu-swizzle-grid` | CUTLASS-style SPMD grid rasterization; the route is single-block |
| `npu-lower-device-builtins` | for SPMD kernels that read `core_id`/`block_dim` |
| `convert-linalg-to-pto`, `outline-linalg-to-npu` | linalg front end; this bridge starts from Buffer IR |
| `npu-embed-kernel-binary`, `convert-npu-kernel-to-func` | alternate build/host-emulation paths |

## Performance

The two routes are compared in
`tests/st/runtime/cce_mlir/test_cce_mlir_performance.py`, which measures the
kernel corpus in `tests/st/runtime/cce_mlir/corpus.py` through every route in
`tests/st/runtime/cce_mlir/bench_matrix.py`. Each measurement compiles the same
`@pl.jit` kernel, registers it once on a `ChipWorker`, dispatches over device
tensors, and reads the AICore task time from chip-swimlane level 1. This
avoids the per-call artifact assembly that otherwise dominates
`CompiledProgram` dispatch, and it measures the compiled kernel rather than
host latency.

| Route | Compiler | Memory planner |
| --- | --- | --- |
| `ptoas/PYPTO` | PTOAS | PyPTO-baked offsets (`ptoas --pto-level=level3`) |
| `ptoas/PTOAS` | PTOAS | PTOAS `PlanMemory` (`--pto-level=level2`) |
| `cce_mlir/PYPTO` | cce-mlir | the only planner the bridge consumes |

`best_per_route` collapses the two PTOAS planners to the compiler's faster
result; that is the "fastest baseline" for each compiler, and the test asserts
the two stay inside a 2× bound. The corpus covers single-tile MMAD, an
accumulation loop, a tiled K reduction, the same reduction under
`pl.pipeline(stage=2)`, a vector add, a scalar elementwise chain, and a
two-kernel cube-then-vector program.

Measured on an Ascend 910B2 (AICore task time, microseconds, one run):

| Kernel | PTOAS best | cce-mlir |
| --- | ---: | ---: |
| single 128³ MMAD | 5.60 | 5.56 |
| 65 accumulating MMADs | 26.08 | 23.94 |
| tiled K reduction (K=512) | 13.22 | 13.04 |
| tiled K reduction, `stage=2` | 12.42 | 13.86 |
| vector add (128²) | 7.68 | 7.40 |
| vector `x * 1.5 + 1.0` | 5.60 | 6.18 |
| cube MMAD then vector scale | 8.72 | 9.52 |

Run-to-run spread on this host is several percent per kernel, so a gap smaller
than roughly 10% is noise and the winner can flip between runs; the tiled
K-reduction row did exactly that across two consecutive runs. Both compilers
compile to the same hardware intrinsics (GM→L1 ND2NZ, L1→L0A/L0B, MAD, L0C→GM
fixpipe); the fixed per-task start-up and DMA cost is several microseconds, so
codegen differences only become visible once the compute chain is long. The
test enforces a 2× parity bound rather than a target number, and writes a
machine-readable report (`PYPTO_CCE_MLIR_BENCH_JSON`).

Run it through the queue on hardware:

```bash
docker exec <queue-container> task-submit --device auto --device-num 1 \
  --max-time 3600 --timeout 0 \
  --run 'cd /opt/pypto && PTOAS_ROOT=/opt/pypto/.venv \
    PYPTO_CCE_MLIR_BENCH_JSON=/tmp/bench.json \
    .venv/bin/python -m pytest tests/st/runtime/cce_mlir/test_cce_mlir_performance.py -v \
    --platform a2a3 --device $TASK_DEVICE'
```

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
- Only rank-2, statically shaped tensors reach Buffer IR, so rank-3 batch
  kernels never enter this route.
- `buffer.full` is blocked by a cce-mlir bug: `pto.expands` mis-lowers to
  `tensor.extract` with one index for rank-2 destinations.
- `pl.pipeline` is compared under the PyPTO planner on both routes. The PTOAS
  planner's slot form (`MemRef.slot_count_ > 1`, `iv % F` indexing) is excluded
  by the Buffer-IR selection pass, so this route cannot consume it.
- `npu-lower-pto-to-llvm-pipeline` needs the standalone host driver to retain
  the device module, so the device-object path currently emits it too.
- cce-mlir is provided by the Docker image; without it the MLIR and pipeline
  tests are skipped and the typed front end still runs.
