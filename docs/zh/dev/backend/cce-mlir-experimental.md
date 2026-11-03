# 实验性的 PyPTO 到 cce-mlir 桥接

本文档描述 `feat/cce-mlir-experimental` 分支上的 cce-mlir 编译路线。它是独立路线：
PyPTO 生产环境的 `ir.compile`、`BackendType`、runtime 和 PTOAS 均未改变。

## 已实现内容

`pypto.backend.cce_mlir` 将 PyPTO 的 **Buffer 阶段 InCore IR** 降低为 cce-mlir 的
PTO/NPU MLIR，并在进程内运行 cce-mlir 的 pipeline：

```text
PyPTO @pl.jit kernel
  -> pass pipeline with enable_buffer_ir
  -> Buffer-stage InCore function
  -> typed CceKernel IR          (pypto.backend.cce_mlir.analysis)
  -> in-memory MLIR module       (dialect bindings, no text templates)
  -> cce-mlir PassManager        (no cce-opt subprocess)
  -> ET_REL device object + host driver
```

每个发射（emit）的值都来自 PyPTO IR 或已注册的后端元数据；无法推导的内容一律拒绝。

## 包结构

| 模块 | 职责 |
| --- | --- |
| `analysis.py` | Buffer IR → 有类型的 kernel IR；合约校验 |
| `layout.py` | memory space 与 fractal 编码推导 |
| `kernel_ir.py` | 与后端无关的有类型 kernel 表示 |
| `registry.py`、`emitters.py` | 以已注册 PyPTO op 名为键的 emitter 注册表 |
| `emit.py` | 有类型 IR → `cce_mlir.ir.Module`（使用 dialect 绑定） |
| `pipeline.py` | 进程内 `PassManager` 运行与 device object 提取 |
| `elf.py` | ELF64 检查（可重定位、符号、入口点） |
| `adapter.py` | `kernel_entry` C++ 适配器生成 |
| `manifest.py`、`driver.py` | 可复现 manifest 与独立编译驱动 |

## 推导规则

| 输出 | 来源 |
| --- | --- |
| Memory tier（`gm`、`ub`、`l1`、`l0a`、`l0b`、`l0c`） | `BufferType.memory_space` |
| Tensor 编码 | cce-mlir layout 工厂：NZ staging `{16, 256/bits}` perm `{1,0}`、ZZ L0A、ZN L0B、固定 16x16 L0C 累加器、GM/UB 为 dense。并与 `BufferType.blayout/slayout/fractal` 交叉校验 |
| 形状、字节数 | `BufferType.shape` 与 dtype 的字节宽度 |
| 参数顺序与方向 | `Function.params` + `Function.param_directions` |
| 操作数角色（读/写/目标） | `ir.get_op_buffer_arg_effect` |
| Kernel 类型 | `func_type`（AIC → cube，AIV → vector），并与分配的 tier 交叉校验 |
| Target | `BackendHandler.get_pto_target_arch()`（`a2a3` → `a3`） |
| 符号 | `Function.name`。独立 runner 保留 cce-mlir 的 `<name>__kernel0`；PyPTO runtime 产物使用裸 `<name>` |

不字面写入任何 fixture 形状、字节数、layout 或符号。

## 支持的合约

- Cube：`buffer.matmul`、`buffer.matmul_acc`，`Left × Right → Acc`，FP16/BF16
  输入、FP32 累加器。
- Vector：UB tile 上的 `buffer.add/mul/sub/div/maximum/minimum` 与
  `buffer.abs/exp/sqrt/neg/relu/log/recip`。
- 搬运：`buffer.load`（GM → L1/UB）、`buffer.store`（L0C/UB → GM）、
  `buffer.copy`（L1 → L0A/L0B、UB → UB）。
- 静态二维形状、整窗搬运、A3、单 block。

带类型诊断拒绝：窗口或部分搬运、动态 valid shape、标量 kernel 参数、标量
elementwise 变体、控制流、Bias/A5 scale 空间、cube/vector 混合 kernel、SPMD block
查询、`buffer.extract/subview/reshape/set_validshape`，以及任何未注册 emitter 的算子。

## 选择该路线

该路线需要显式启用，PTOAS 默认路径不变：

```python
from pypto.runtime import ExecutionMode, RunConfig

compiled = kernel.compile(
    config=RunConfig(kernel_compiler="cce_mlir", execution_mode=ExecutionMode.ONBOARD)
)
```

任何未修改的示例都可以通过环境变量选择该路线，无需编辑其 `RunConfig`：

```bash
PYPTO_KERNEL_COMPILER=cce_mlir python examples/beginner/05_matmul.py
```

该变量只提供默认值；显式的 `kernel_compiler=` 参数优先。选择 `cce_mlir` 时，
`ONBOARD` 也成为默认 execution mode，因为该路线无法面向模拟器。在设备主机上，
仍通过队列运行（`task-submit --device auto ... --run
'PYPTO_KERNEL_COMPILER=cce_mlir python examples/...'`）。

`ir.compile(program, kernel_compiler="cce_mlir")` 接受同样的选项。该路线要求
onboard `a2a3` 目标，忽略仅属于 PTOAS 的选项，并生成带有 `link_objects` 的
runtime 产物：

```text
kernel_config.py                      每个 kernel 的 link_objects -> .cce.o
orchestration/<orch>.cpp              复用现有 orchestration codegen
kernels/<aic|aiv>/<name>.cpp          kernel_entry 适配器
kernels/<aic|aiv>/<name>.cce.o        cce-mlir device object
cce_mlir/<name>.mlir                  发射出的 module
cce_mlir/<name>.lowered.mlir          经 cce-mlir pipeline 后的 module
cce_mlir/<name>.manifest.json         合约、版本、object 摘要
```

在 `DEBUG` 级别，成功编译后还会打印解析后的产物根目录和链接的 device object，
便于丢弃返回值的调用方定位文件：

```text
2026-11-03 15:20:11.123 D | cce-mlir artifacts written to /abs/out (device objects: /abs/out/kernels/aic/<name>.cce.o)
```

设置 `PYPTO_LOG_LEVEL=debug` 或调用 `set_log_level(LogLevel.DEBUG)`；默认 INFO
行为不变。

## 产物

| 文件 | 含义 |
| --- | --- |
| `<kernel>.mlir` | 通过 cce-mlir 绑定构造的 module |
| `<kernel>.lowered.mlir` | 经 `npu-lower-pto-to-llvm-pipeline` 后的 module |
| `<kernel>.o` | 提取出的 `ET_REL` device object |
| `<kernel>.manifest.json` | 推导合约、版本、object 摘要、符号、队列任务 id |
| `report.json` | 编译状态与诊断 |

## 使用 Docker 复现

在 PyPTO 仓库根目录（ARM64 Ascend 主机）构建：

```bash
# 一次性准备：登录私有 registry 并验证 SSH 访问。
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

编译（不需要设备）：

```bash
docker run --rm \
  -v "$PWD/artifacts:/opt/pypto/artifacts" \
  pypto-cce-mlir:910b \
  compile --output-dir /opt/pypto/artifacts/cce_mlir
```

在 NPU 上运行**必须经过队列**，不能手选设备。按
`pypto-3.0-notes/tools/npu-task-submit-runbook.md` 启动加入队列的容器，然后提交：

```bash
docker exec <queue-container> task-submit --device auto --device-num 1 \
  --max-time 1800 --timeout 0 \
  --run 'python3 -m examples.cce_mlir run --output-dir /tmp/cce_run'
```

成功时打印 `NPU smoke test passed`。驱动从 `TASK_DEVICE`（或 `task-submit`
追加的 `--device` 值）读取分配到的逻辑设备，绝不修改 `ASCEND_RT_VISIBLE_DEVICES`。

## Runtime ABI 状态

PyPTO 以 `kernel_entry(__gm__ int64_t* args)` 调度，每个参数是一个设备 `Tensor`
描述符；cce-mlir 发射的 `<name>__kernel0` 接收裸 GM 指针。`adapter.py` 生成桥接代码，
runtime 的 kernel 编译器接受额外链接对象，并**首先**链接适配器对象，使 `kernel_entry`
位于提取出的 `.text` 偏移 0。

`ir.compile(kernel_compiler="cce_mlir")` 已生成 runtime 产物（带 `link_objects`
的 `kernel_config.py`、适配器源码、device object、orchestration），链接已在
`tests/ut/backend/cce_mlir/test_adapter_link.py` 验证（`kernel_entry` 位于 `.text`
偏移 0，standalone 符号已解析）。

device object 是**可独立调用的函数**，而不是 cce-mlir 的 kernel entry。cce-mlir
的翻译器只为带 kernel 标记的函数添加每个函数的 AI-core
`target-cpu`/`target-features`，因此 `pipeline.compile_device_function` 在
`npu-lower-pto-to-cce-pipeline` 与 `npu-lower-cce-to-llvm-device-only-pipeline`
期间保留该标记，经 `cce-translate` 翻译后再重写 LLVM IR：用裸 `<name>` 替换
`<name>__kernel0`，并丢弃 `!hivm.annotations` kernel 元数据。随后由 `bisheng`
编译，适配器将其链接到 `kernel_entry` 之下。cce-mlir 后续应提供原生的可调用
device function 模式来替代这一重写。

`tests/st/runtime/cce_mlir/test_cce_mlir_numerical.py` 已在 Ascend 910B2 上通过
`CompiledProgram` 运行 cube matmul 与 vector add 并通过。

## 已知限制

- 仅 A3；PyPTO 的 Buffer IR 选择门是 `a2a3`。
- 每个 kernel 一个 block；不支持 SPMD block 查询。
- `npu-lower-pto-to-llvm-pipeline` 需要独立 host driver 才能保留 device module，
  因此 device object 路径目前也会发射它。
- cce-mlir 由 Docker 镜像提供；缺失时跳过 MLIR 与 pipeline 测试，有类型前端仍可运行。
