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
  `buffer.abs/exp/sqrt/neg/relu/log/recip`，以及单 tile 加标量的变体
  `buffer.adds/subs/muls/divs/maximums/minimums` 与 `buffer.lrelu`。
- 搬运：`buffer.load`（GM → L1/UB）、`buffer.store`（L0C/UB → GM）、
  `buffer.copy`（L1 → L0A/L0B、UB → UB），包括**带窗口**的搬运：非零 offset，
  且 extent 填满片上 buffer。窗口 offset 可以是外层循环归纳变量的仿射表达式
  （`k * 128`），分块 K 归约正是靠它复用同一块 tile。窗口以 GM tensor 的
  `tensor.extract_slice` 表示，cce-mlir 会把它折叠进搬运本身
  （`cce.copy_gm_to_l1_nd2nz`），不会物化额外拷贝。部分写回先写窗口，再用
  `tensor.insert_slice` 拼接回父 tensor；bufferize 后成为 fixpipe 直接写入的
  strided subview。
- 静态二维形状、A3、单 block。
- 静态边界的 `pl.range` 循环，循环体操作固定操作数。PyPTO 的 Buffer IR 将循环
  状态保存在 buffer 中，因此循环体内写入的每个 tensor 都作为 `scf.for` 的
  iter-arg 传递，循环后的值取循环结果。软件流水循环（`pl.pipeline`）在 PyPTO
  planner 下由 `LowerPipelineLoops` 复制循环体，因此桥接层无需额外 lowering。

带类型诊断拒绝：动态搬运 extent、动态 valid shape、标量 kernel 参数、非常量循环
边界、`pl.yield` 携带的循环值、`while`/`if`、Bias/A5 scale 空间、cube/vector
混合 kernel、SPMD block 查询、`buffer.full`（该 cce-mlir 构建对 rank-2 的
`pto.expands` lowering 有误）、`buffer.extract/subview/reshape/set_validshape`，
以及任何未注册 emitter 的算子。

## 优化 Pass

该路线运行 cce-mlir 的规范 pipeline，已包含分配与同步工作：

| 运行 | 作用 |
| --- | --- |
| `pto-split-by-kind`、`pto-lower-cross-core` | cube/vector 拆分与跨核交接 |
| `pto-propagate-layout` | 折叠/抵消 relayout 链（当前 kernel 为 no-op） |
| `lower-pto-to-cce` | PTO 降至 CCE tensor SSA |
| `cce-insert-sync-dataflow` | 从 tensor dataflow 推导 pipeline barrier |
| `cce-verify-flag-pairs`、`cce-allocate-flags` | 校验并分配 event id |
| `cce-materialize-rank`、`npu-bufferize`、`cce-bufferize`、`npu-void-kernel-abi` | layout rank 与 bufferization |
| `cce-assign-ub-buffers`、`cce-resolve-onchip-carries`、`cce-allocate-buffer-offsets`、`cce-lower-to-hivm` | UB 放置、循环携带 buffer、字节偏移 |
| `hivm-lower-to-llvm` + LLVM tail | 设备 LLVM |
| `bisheng -O3` | 设备 object 优化 |

未使用及原因：

| Pass | 原因 |
| --- | --- |
| `pto-tile-cube-gemm` | 针对整矩阵 `linalg`→PTO 进行分块；桥接器发射显式 tile 链。仅当路线接受超过 L0 的形状时才需要 |
| `npu-swizzle-grid` | CUTLASS 风格 SPMD grid 重排；路线当前为单 block |
| `npu-lower-device-builtins` | 供读取 `core_id`/`block_dim` 的 SPMD kernel 使用 |
| `convert-linalg-to-pto`、`outline-linalg-to-npu` | linalg 前端；本桥接从 Buffer IR 开始 |
| `npu-embed-kernel-binary`、`convert-npu-kernel-to-func` | 备用构建/宿主仿真路径 |

## 性能

两条路线在 `tests/st/runtime/cce_mlir/test_cce_mlir_performance.py` 中比较，测量
对象是 `tests/st/runtime/cce_mlir/corpus.py` 中的 kernel 语料，
`tests/st/runtime/cce_mlir/bench_matrix.py` 定义路线与测量。每次测量编译同一个
`@pl.jit` kernel，在 `ChipWorker` 上注册一次，使用 device tensor 反复派发，并从
chip swimlane level 1 读取 AICore 任务时间。这样既避开了 `CompiledProgram`
每次派发重新组装产物的开销，也测量编译后的 kernel 本身而非 host 延迟。

| 路线 | 编译器 | 内存 planner |
| --- | --- | --- |
| `ptoas/PYPTO` | PTOAS | PyPTO 预烘焙 offset（`ptoas --pto-level=level3`） |
| `ptoas/PTOAS` | PTOAS | PTOAS `PlanMemory`（`--pto-level=level2`） |
| `cce_mlir/PYPTO` | cce-mlir | 桥接层唯一能消费的 planner |

`best_per_route` 把 PTOAS 的两种 planner 收敛为该编译器更快的结果，即每个编译器
的“最快基线”；测试要求两者保持在 2 倍以内。语料覆盖单 tile MMAD、累加循环、
分块 K 归约、同一归约的 `pl.pipeline(stage=2)` 版本、vector add、标量
elementwise 链，以及 cube 后接 vector 的双 kernel 程序。

Ascend 910B2 实测（AICore 任务时间，微秒，单次运行）：

| Kernel | PTOAS 最优 | cce-mlir |
| --- | ---: | ---: |
| 单个 128³ MMAD | 5.60 | 5.56 |
| 65 次累加 MMAD | 26.08 | 23.94 |
| 分块 K 归约（K=512） | 13.22 | 13.04 |
| 分块 K 归约，`stage=2` | 12.42 | 13.86 |
| vector add（128²） | 7.68 | 7.40 |
| vector `x * 1.5 + 1.0` | 5.60 | 6.18 |
| cube MMAD 后接 vector scale | 8.72 | 9.52 |

该主机逐 kernel 的运行间波动有数个百分点，因此小于约 10% 的差距属于噪声，优劣
方向可能在两次运行间翻转；分块 K 归约一行在连续两次运行中就发生了翻转。两个
编译器生成相同的硬件 intrinsic（GM→L1 ND2NZ、L1→L0A/L0B、MAD、L0C→GM
fixpipe）；每个任务的固定启动与 DMA 成本有数微秒，因此只有计算链足够长时
codegen 差异才显现。测试使用 2 倍 parity 上限，而不是固定目标数值，并输出机器
可读报告（`PYPTO_CCE_MLIR_BENCH_JSON`）。

在硬件上通过队列运行：

```bash
docker exec <queue-container> task-submit --device auto --device-num 1 \
  --max-time 3600 --timeout 0 \
  --run 'cd /opt/pypto && PTOAS_ROOT=/opt/pypto/.venv \
    PYPTO_CCE_MLIR_BENCH_JSON=/tmp/bench.json \
    .venv/bin/python -m pytest tests/st/runtime/cce_mlir/test_cce_mlir_performance.py -v \
    --platform a2a3 --device $TASK_DEVICE'
```

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

## 查看生成的代码

`python -m pypto.tools.trace_kernel` 编译一个 `@pl.jit` kernel，导出每个 PyPTO
pass 的 IR 与全部后端产物，并打印“阶段 → 文件”的对照表：

```bash
python -m pypto.tools.trace_kernel \
    --file tests/st/runtime/cce_mlir/corpus.py --function matmul_kloop
```

文件中定义多个 kernel 时用 `--function` 选择；参数是裸 `pl.Tensor` 注解（没有
静态形状）时用 `--samples f16:128x512,f16:512x128,f32:128x128` 指定样本张量。
`--compiler ptoas` 改为追踪另一条路线；`--out DIR` 指定产物根目录（默认
`build_output/trace_<stem>`）。

加 `--html` 会额外渲染 `<out>/trace.html`：一页按流水线顺序列出全部阶段，每个
阶段与**同一表示**中的上一阶段逐行 diff（PyPTO IR、PTO MLIR、cce-mlir
pipeline、device LLVM），并带阶段索引、每阶段 `+n/−n` 计数和“只看有变化的阶段”
过滤。diff 不跨表示边界，跨边界处页面会显式标注。

cce-mlir 路线下，若找到 `cce-opt`，工具还会运行 cce-mlir 自己的 pass manager：
`--mlir-print-ir-tree-dir` 会在 `<out>/cce_passes/NN_<pass>.mlir` 下为每个 pass
单独写出一个文件，`<out>/cce_level.mlir` 是 PTO→CCE 之后的 module
（PTO→CCE 之后的 module）。建议按以下顺序阅读：

| 阶段 | 文件 |
| --- | --- |
| 你写的 Python，对应的 PyPTO IR | `passes_dump/00_frontend.py` |
| tensor 算子变为 tile 算子 | `passes_dump/*after_ConvertTensorToTileOps.py` |
| tile 内存空间 / layout | `passes_dump/*after_InferTileMemorySpace.py`、`*after_ResolveBackendOpLayouts.py` |
| Buffer IR 选择门 | `passes_dump/*after_SelectBufferIRFunctions.py` |
| Buffer IR（后端交接点） | `passes_dump/*after_LowerTileToBuffer.py` |
| PyPTO 发射的 PTO MLIR | `cce_mlir/*.mlir` |
| cce-mlir 各 pass / CCE 层 | `cce_passes/NN_<pass>.mlir`、`cce_level.mlir` |
| device LLVM | `cce_mlir/*.lowered.mlir` |

若 `*after_SelectBufferIRFunctions.py` 中记录 `"buffer_ir_selected": False`，说明
kernel 停在 Tile IR，该路线不会产生 `cce_mlir/` 产物：拒绝它的是 Buffer IR
选择门，而不是桥接层。

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
- 只有二维静态形状能进入 Buffer IR，因此 rank-3 batch kernel 不会进入该路线。
- `buffer.full` 被 cce-mlir 缺陷阻塞：对 rank-2 目标，`pto.expands` 会错误地
  lower 成只带一个 index 的 `tensor.extract`（cce-mlir#272，与 #253 重复）。
- 以循环变量为 offset、写入更大 GM tensor 窗口的 store（`pl.store(tile, [i * 128, 0], out)`）
  无法 lower：`cce-lower-to-hivm` 会留下未转换的 `unrealized_conversion_cast`
  （cce-mlir#271，相关 #64、#102）。静态 offset 可以 lower 且结果精确。
- `buffer.recip` 映射到近似的 `vrec`，sigmoid/softmax 类数值与 PTOAS 相差约
  3e-3 相对误差（cce-mlir#273）。
- 原地 elementwise 更新（`dest == src`）在上游会算错（cce-mlir#274，与 #147
  重复）；桥接层改为写入 scratch，在循环内再把结果拷回目标 buffer。
- `pl.pipeline` 在两条路线下都使用 PyPTO planner 比较。PTOAS planner 的 slot
  形式（`MemRef.slot_count_ > 1`、`iv % F` 索引）被 Buffer IR 选择 pass 排除，
  因此该路线无法消费。
- `npu-lower-pto-to-llvm-pipeline` 需要独立 host driver 才能保留 device module，
  因此 device object 路径目前也会发射它。
- cce-mlir 由 Docker 镜像提供；缺失时跳过 MLIR 与 pipeline 测试，有类型前端仍可运行。
