# Experimental PyPTO to cce-mlir bridge

This page documents the reproducible experimental bridge from a small PyPTO
kernel to cce-mlir PTO/NPU MLIR. It is intentionally a standalone slice on the
`feat/cce-mlir-experimental` branch of the PyPTO fork; it does not change
PyPTO's production `ir.compile`, `BackendType`, runtime, or PTOAS paths.

## What is implemented

The example lowers the fixed-shape `matmul_64x128x64` fixture in
`examples/cce_mlir/fixtures/pypto_matmul.py`:

```text
PyPTO JITFunction
  -> PyPTO Buffer/InCore IR
  -> KernelSpec extracted by an IR visitor
  -> cce-mlir PTO/NPU MLIR with explicit layouts and memory spaces
  -> cce-opt npu-lower-pto-to-llvm-pipeline
  -> CANN/ACL device artifact
```

The supported contract is A3 cube matmul with A and B as FP16, C as FP32,
static rank-2 shapes, and dimensions that are multiples of 16. The lowering
recognizes the actual PyPTO `Buffer` IR operations (`buffer.load`,
`buffer.copy`, `buffer.matmul`, and `buffer.store`) rather than parsing the
Python fixture's source text. The emitter makes the GM/L1/L0A/L0B/L0C
transfers and layout conversions explicit.

The generated artifact directory contains:

| File | Meaning |
| --- | --- |
| `<kernel>.mlir` | Input module emitted by the bridge |
| `<kernel>.lowered.mlir` | `cce-opt` stdout when compilation succeeds |
| `<kernel>.manifest.json` | Kernel contract, revisions, command, and ABI audit |
| `<kernel>.report.json` | Status, command, return code, and diagnostic text |

The manifest explicitly records that cce-mlir's ACL launcher symbols do not
match PyPTO's current external-kernel `kernel_entry(__gm__ int64_t*)` ABI. The
artifact therefore needs a dedicated adapter if it is to be called through the
PyPTO runtime; it is not silently presented as a PyPTO external source.

## Reproduce with Docker

The Dockerfile is self-contained with respect to this repository. It does not
read `pypto-3.0-notes` or copy files from that checkout. The cce-mlir source is
cloned at `983beae64d7cfec3903782d34621150a68a1df87` by default and built
against the LLVM layer supplied by the cce-mlir development image. The final
stage is the Ascend CANN 9.0.0 A3/A2-A3 image.

Build from the PyPTO repository root:

```bash
git ls-remote git@ssh.gitlab.huaweirc.ch:zrc-von-neumann-lab/skyline-software/cce-mlir.git HEAD

docker build \
  --ssh cce_mlir_ssh="$SSH_AUTH_SOCK" \
  --secret id=cce_mlir_known_hosts,src="$HOME/.ssh/known_hosts" \
  --build-arg CCE_MLIR_IMAGE=registry.gitlab.huaweirc.ch/zrc-von-neumann-lab/skyline-software/cce-mlir/ci/cce-mlir-dev:npu-aarch64-latest \
  --build-arg CANN_ARCH=aarch64-linux \
  -f docker/pypto_cce_mlir.Dockerfile \
  -t pypto-cce-mlir:910b \
  .
```

The command above is for an ARM64 Ascend host. The cce-mlir project publishes
separate `cpu-latest`, `npu-x86_64-latest`, and `npu-aarch64-latest` builder
images; the builder image and `CANN_ARCH` must match the host architecture.
The Dockerfile defaults to `cpu-latest` and `x86_64-linux` for compatibility
with ordinary x86 Docker hosts. Registry authentication (`docker login`) and
repository authentication (the forwarded SSH agent) are separate. The SSH
agent and the host's verified `known_hosts` file are available only during the
clone step and are not copied into the image. The `known_hosts` file must
contain the `ssh.gitlab.huaweirc.ch` host key; running the `git ls-remote`
command above once establishes it interactively.

For an auditable build, override both image references with immutable
`@sha256:<digest>` references and record the selected cce-mlir commit in the
build log. The Dockerfile keeps readable tags as defaults because registry
digests vary by mirror and architecture.

Compile inside the image:

```bash
docker run --rm \
  pypto-cce-mlir:910b \
  compile --output-dir /opt/pypto/artifacts/cce_mlir
```

Run the host wrapper on an Ascend NPU when the host has the matching Ascend
container runtime and device access:

```bash
docker run --rm --runtime=ascend \
  --device=/dev/davinci0 \
  -e ASCEND_VISIBLE_DEVICES=0 \
  pypto-cce-mlir:910b \
  run --device 0 --output-dir /opt/pypto/artifacts/cce_mlir
```

The `run` command is a cce-mlir `npu_runner` smoke test. It checks the output
against NumPy FP32 matmul and prints `NPU smoke test passed` only after the
device invocation returns successfully. A compile-only result is not evidence
of NPU execution.

## Reproduce without Docker

With PyPTO and cce-mlir installed in the environment:

```bash
PYTHONPATH=. python3 -m examples.cce_mlir compile \
  --cce-opt /opt/cce-mlir/build/bin/cce-opt \
  --cann-root /usr/local/Ascend/cann-9.0.0 \
  --cann-arch x86_64-linux \
  --output-dir artifacts/cce_mlir
```

If `cce-opt` or the CANN configuration is absent, the command still emits the
input MLIR, manifest, and report and returns a clearly labelled blocked status:
`blocked_missing_cce_opt` or `blocked_missing_cann_configuration`.

## Scope and next integration level

This branch proves the lowest useful compiler-infrastructure path:

1. PyPTO remains the source-level and Buffer-IR producer.
2. cce-mlir owns target-specific memory spaces, layouts, PTO operations, and
   CANN/LLVM lowering.
3. The bridge owns only contract extraction, emission, artifact metadata, and
   the cce-mlir runner smoke test.

The next integration step is an explicit adapter from cce-mlir's generated ACL
launcher ABI to a PyPTO external-kernel ABI. That should be implemented only
after the generated binary and symbol contract are stable on a real NPU. A
full PyPTO backend would additionally need a target registration, shape/layout
coverage, runtime ownership, and numerical tests; those are intentionally out
of scope for this branch.

## Known limitations

- Only the fixed A3 cube contract is emitted.
- A5/vector lowering is not implemented by this slice.
- The local development host may be unable to execute `cce-opt` because its
  shared libraries and CANN installation are container-provided.
- The Docker build requires access to the cce-mlir GitLab registry/repository
  and the Ascend CANN registry.
- NPU execution requires an Ascend host, compatible driver/runtime, and device
  passthrough; CPU compilation alone does not validate hardware execution.
