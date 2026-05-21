# Installing PyPTO

## Quick start: Docker (standalone image)

The standalone Docker image requires nothing from your host except the Ascend
kernel driver.  Everything else — CANN, pypto, simpler, pto-isa, PTOAS — is
cloned and built inside the image.

### Build

```bash
docker build -t pypto3-hw-native-sys:cann9 - < Dockerfile.hw-native-sys.cann9.0
```

To install under a custom prefix or pin a specific pto-isa commit:

```bash
docker build \
  --build-arg INSTALL_PREFIX=/workspace \
  --build-arg PTO_ISA_COMMIT=2c607938 \
  -t pypto3-hw-native-sys:cann9 \
  - < Dockerfile.hw-native-sys.cann9.0
```

### Run

Single-device:

```bash
docker run --rm -it --privileged --ipc=host \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /dev:/dev \
  pypto3-hw-native-sys:cann9
```

Multi-device (HCCL / distributed):

```bash
docker run --rm -it --privileged --ipc=host \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /dev:/dev \
  pypto3-hw-native-sys:cann9
```

> **Important:** Do NOT mount `/usr/local/Ascend` from the host.  The image
> already contains CANN 9.0.0.  Mounting the host's `/usr/local/Ascend` would
> shadow the image's CANN with the host's (potentially older) version.
> Only the kernel driver at `/usr/local/Ascend/driver` is needed.

### Common commands

```bash
# simpler (pre-installed; rebuild only after editing source)
cd /opt/pypto/runtime
pytest tests/ -v --platform=a2a3 --device="0,1,2,3"
python examples/workers/l3/allreduce_distributed/main.py -p a2a3 -d 0-1

# pypto
cd /opt/pypto
python -c "import pypto; print('pypto ok')"
which ptoas
pytest tests/st -v --device="0,1,2,3" --precompile-workers=128 --pto-isa-commit=2c607938 --ignore=tests/st/distributed
pytest tests/st/distributed -v --device="0,1,2,3" --pto-isa-commit=2c607938
```

---

## Development install (from source)

See the [developer guide](en/developer_guide.md) for a full walkthrough.
