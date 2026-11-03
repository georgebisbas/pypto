# 实验性的 PyPTO 到 cce-mlir 桥接

本文档记录从小型 PyPTO kernel 到 cce-mlir PTO/NPU MLIR 的可复现实验桥接。
它位于 PyPTO fork 的 `feat/cce-mlir-experimental` 分支，是独立的实验切片；
不会修改 PyPTO 生产环境中的 `ir.compile`、`BackendType`、runtime 或 PTOAS 路径。

## 已实现内容

示例降低（lower）`examples/cce_mlir/fixtures/pypto_matmul.py` 中固定形状的
`matmul_64x128x64` fixture：

```text
PyPTO JITFunction
  -> PyPTO Buffer/InCore IR
  -> IR visitor 提取 KernelSpec
  -> 带显式 layout 和 memory space 的 cce-mlir PTO/NPU MLIR
  -> cce-opt npu-lower-pto-to-llvm-pipeline
  -> CANN/ACL device artifact
```

支持的合约是 A3 cube matmul：A、B 为 FP16，C 为 FP32，静态二维 shape，且各维是
16 的倍数。降低过程识别真实的 PyPTO `Buffer` IR 操作（`buffer.load`、
`buffer.copy`、`buffer.matmul`、`buffer.store`），而不是解析 Python fixture 的
源代码。发射器显式表达 GM/L1/L0A/L0B/L0C 之间的数据搬运和 layout 转换。

产物目录包含：

| 文件 | 含义 |
| --- | --- |
| `<kernel>.mlir` | 桥接器发射的输入 module |
| `<kernel>.lowered.mlir` | 编译成功时的 `cce-opt` 标准输出 |
| `<kernel>.manifest.json` | kernel 合约、版本、命令和 ABI 审计 |
| `<kernel>.report.json` | 状态、命令、返回码和诊断文本 |

manifest 明确记录 cce-mlir 的 ACL launcher symbol 与 PyPTO 当前外部 kernel 的
`kernel_entry(__gm__ int64_t*)` ABI 不匹配。因此若要通过 PyPTO runtime 调用，
仍需要专用 adapter；该产物不会被静默地标记为 PyPTO external source。

## 使用 Docker 复现

该 Dockerfile 仅依赖本仓库，不读取 `pypto-3.0-notes`，也不会从该 checkout
复制文件。默认会在固定 commit
`983beae64d7cfec3903782d34621150a68a1df87` checkout cce-mlir，并使用
cce-mlir 开发镜像提供的 LLVM 层构建。最终 stage 是 Ascend CANN 9.0.0 A3/A2-A3
镜像。

在 PyPTO 仓库根目录构建：

```bash
docker build \
  -f docker/pypto_cce_mlir.Dockerfile \
  -t pypto-cce-mlir:910b \
  .
```

为了审计构建过程，应使用不可变的 `@sha256:<digest>` 覆盖两个镜像引用，并在构建
日志中记录 cce-mlir commit。Dockerfile 保留可读 tag 作为默认值，是因为不同 registry
镜像和架构的 digest 可能不同。

在镜像中编译：

```bash
docker run --rm \
  pypto-cce-mlir:910b \
  compile --output-dir /opt/pypto/artifacts/cce_mlir
```

在具备匹配 Ascend container runtime 和设备访问权限的 NPU 主机上运行：

```bash
docker run --rm --runtime=ascend \
  --device=/dev/davinci0 \
  -e ASCEND_VISIBLE_DEVICES=0 \
  pypto-cce-mlir:910b \
  run --device 0 --output-dir /opt/pypto/artifacts/cce_mlir
```

`run` 是 cce-mlir `npu_runner` smoke test，并使用 NumPy FP32 matmul 校验输出。
只有设备调用成功后才会打印 `NPU smoke test passed`；仅编译成功并不能证明 NPU 已执行。

## 不使用 Docker 复现

在环境中已安装 PyPTO 和 cce-mlir 时：

```bash
PYTHONPATH=. python3 -m examples.cce_mlir compile \
  --cce-opt /opt/cce-mlir/build/bin/cce-opt \
  --cann-root /usr/local/Ascend/cann-9.0.0 \
  --cann-arch x86_64-linux \
  --output-dir artifacts/cce_mlir
```

当 `cce-opt` 或 CANN 配置缺失时，命令仍会生成输入 MLIR、manifest 和 report，
并返回明确的阻塞状态：`blocked_missing_cce_opt` 或
`blocked_missing_cann_configuration`。

## 范围和下一层集成

本分支验证了最小但有用的 compiler-infrastructure 路径：

1. PyPTO 继续作为 source-level 和 Buffer-IR producer。
2. cce-mlir 负责 target-specific memory space、layout、PTO 操作和 CANN/LLVM 降低。
3. 桥接器只负责合约提取、发射、产物元数据以及 cce-mlir runner smoke test。

下一步应先实现 cce-mlir 生成的 ACL launcher ABI 到 PyPTO external-kernel ABI 的
显式 adapter，并在真实 NPU 上确认 binary 和 symbol 合约稳定后再做。完整 PyPTO
backend 还需要 target registration、shape/layout 覆盖、runtime ownership 和数值测试；
这些内容不在本分支范围内。

## 已知限制

- 只发射固定的 A3 cube 合约。
- 本切片尚未实现 A5/vector 降低。
- 本地开发主机可能无法运行 `cce-opt`，因为共享库和 CANN 安装由容器提供。
- Docker 构建需要访问 cce-mlir GitLab registry/repository 和 Ascend CANN registry。
- NPU 执行需要 Ascend 主机、兼容的 driver/runtime 和设备透传；CPU 编译不能验证硬件执行。
