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
| Symbols | `Function.name`; the `<name>__kernel0` suffix is cce-mlir's translation contract |

No fixture shape, byte count, layout or symbol is written literally.

## Supported contract

- Cube: `buffer.matmul` and `buffer.matmul_acc` with `Left × Right → Acc`,
  FP16/BF16 operands and an FP32 accumulator.
- Vector: `buffer.add/mul/sub/div/maximum/minimum` and
  `buffer.abs/exp/sqrt/neg/relu/log/recip` over UB tiles.
- Transfers: `buffer.load` (GM → L1/UB), `buffer.store` (L0C/UB → GM),
  `buffer.copy` (L1 → L0A/L0B, UB → UB).
- Static rank-2 shapes, full-window transfers, A3, one block.

Refused with a typed diagnostic: windowed or partial transfers, dynamic valid
shapes, scalar kernel parameters, scalar elementwise variants, control flow,
Bias/A5 scale spaces, mixed cube/vector kernels, SPMD block queries,
`buffer.extract/subview/reshape/set_validshape`, and any operator without an
emitter registration.

## Selecting the route

The route is opt-in and leaves the PTOAS default untouched:

```python
from pypto.runtime import ExecutionMode, RunConfig

compiled = kernel.compile(
    config=RunConfig(kernel_compiler="cce_mlir", execution_mode=ExecutionMode.ONBOARD)
)
```

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

The structural link is verified in
`tests/ut/backend/cce_mlir/test_adapter_link.py` (entry ordering and symbol
resolution), and `ir.compile(kernel_compiler="cce_mlir")` produces the runtime
artifact (`kernel_config.py` with `link_objects`, adapter source, device object,
orchestration). Executing that artifact through `CompiledProgram` on an Ascend
910B2 **launches the kernel but traps on AICore**
(`retCode=0x26 aicore exception`): cce-mlir compiles the body as a kernel entry
with launch-time metadata, not as a callable device function. Asking cce-mlir
for a non-kernel `npu.func` currently fails lowering
(`cce-lower-to-hivm left a live unrealized_conversion_cast`), because only
kernel functions receive the void ABI. The numerical ST is therefore a strict
`xfail` until cce-mlir gains a callable-device-function ABI, or until the
artifact is launched through ACL as a separate runtime artifact kind.

## Known limitations

- A3 only; PyPTO's Buffer-IR selection gate is `a2a3`.
- One block per kernel; no SPMD block queries.
- `npu-lower-pto-to-llvm-pipeline` needs the standalone host driver to retain
  the device module, so the device-object path currently emits it too.
- cce-mlir is provided by the Docker image; without it the MLIR and pipeline
  tests are skipped and the typed front end still runs.
