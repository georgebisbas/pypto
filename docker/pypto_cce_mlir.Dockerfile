# syntax=docker/dockerfile:1
# Reproducible PyPTO -> cce-mlir experimental image.
#
# The PyPTO source is the Docker build context. cce-mlir is fetched at an
# immutable commit and built against the pinned/prebuilt LLVM layer supplied by
# its development image. The final stage is the CANN NPU image so the same
# container can compile and run the example on Ascend 910B.

ARG CANN_IMAGE=quay.io/ascend/cann:9.0.0-910b-ubuntu22.04-py3.12
ARG CCE_MLIR_IMAGE=registry.gitlab.huaweirc.ch/zrc-von-neumann-lab/skyline-software/cce-mlir/ci/cce-mlir-dev:cpu-latest
ARG CCE_MLIR_REPO=git@ssh.gitlab.huaweirc.ch:zrc-von-neumann-lab/skyline-software/cce-mlir.git
ARG CCE_MLIR_REF=983beae64d7cfec3903782d34621150a68a1df87
ARG CANN_ARCH=x86_64-linux

FROM ${CCE_MLIR_IMAGE} AS cce_build
ARG CCE_MLIR_REPO
ARG CCE_MLIR_REF

RUN apt-get update \
 && apt-get install -y --no-install-recommends openssh-client \
 && rm -rf /var/lib/apt/lists/*

RUN --mount=type=ssh,id=cce_mlir_ssh,required=true \
 --mount=type=secret,id=cce_mlir_known_hosts,target=/tmp/cce-mlir-known-hosts,required=true \
 set -eu; \
 export GIT_SSH_COMMAND="ssh -o UserKnownHostsFile=/tmp/cce-mlir-known-hosts -o StrictHostKeyChecking=yes"; \
 git clone "$CCE_MLIR_REPO" /opt/cce-mlir \
 && git -C /opt/cce-mlir checkout --detach "$CCE_MLIR_REF" \
 && python3 -m pip install -e /opt/cce-mlir --no-build-isolation \
      --config-settings=build-dir=/opt/cce-mlir/build \
      --config-settings=cmake.define.MLIR_DIR=/usr/local/llvm/lib/cmake/mlir \
      --config-settings=cmake.define.LLVM_DIR=/usr/local/llvm/lib/cmake/llvm \
      --config-settings=cmake.define.CCE_BUILD_MLIR_FROM_SUBMODULE=OFF \
      --config-settings=cmake.define.CCE_BUILD_TESTS=OFF \
      --config-settings=cmake.define.CCE_ENABLE_LSP_SERVER=OFF \
 && test -x /opt/cce-mlir/build/bin/cce-opt \
 && test -x /opt/cce-mlir/build/bin/cce-translate

FROM ${CANN_IMAGE} AS runtime
ARG CCE_MLIR_REF
ARG CANN_ARCH

# cce-mlir's Python bindings (_ccePasses, _cceDialects, _mlir, ...) are compiled
# by the CMake build and installed only into the build/install trees. The source
# tree under /opt/cce-mlir/python is a partial package whose _mlir_libs contains
# no extension modules, so listing it on PYTHONPATH makes it shadow the built
# package and `import cce_mlir.cpu_runner` fails with
# "ImportError: cannot import name '_ccePasses'". Keep only the install trees,
# install/platlib first.
ENV CCE_MLIR_COMMIT=${CCE_MLIR_REF} \
    CANN_VERSION=9.0.0-910b \
    CANN_ROOT=/usr/local/Ascend/cann-9.0.0 \
    CANN_ARCH=${CANN_ARCH} \
    PATH=/opt/cce-mlir/build/bin:/usr/local/llvm/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/llvm/lib:/opt/cce-mlir/build/lib:${LD_LIBRARY_PATH} \
    PYTHONPATH=/opt/cce-mlir/build/install/platlib:/opt/cce-mlir/build/python_packages

COPY --from=cce_build /usr/local/llvm /usr/local/llvm
COPY --from=cce_build /opt/cce-mlir/build /opt/cce-mlir/build
COPY --from=cce_build /opt/cce-mlir/python /opt/cce-mlir/python

RUN python3 -m pip install --no-cache-dir \
      "numpy>=2.0" \
      "scikit-build-core>=0.10.0" \
      "nanobind>=2.0.0,<3" \
      "ninja>=1.11.0" \
      "cmake>=3.15" \
 && python3 -m pip install --no-cache-dir \
      torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /opt/pypto
COPY . /opt/pypto

RUN python3 -m pip install --no-cache-dir --no-build-isolation -e /opt/pypto \
 && python3 -c "import cce_mlir.cpu_runner, cce_mlir.ir" \
 && python3 -m examples.cce_mlir compile \
      --cce-opt /opt/cce-mlir/build/bin/cce-opt \
      --cann-root /usr/local/Ascend/cann-9.0.0 \
      --cann-arch ${CANN_ARCH} \
      --output-dir /opt/pypto/docker-check

ENTRYPOINT ["python3", "-m", "examples.cce_mlir"]
CMD ["compile", "--output-dir", "/opt/pypto/artifacts/cce_mlir"]
