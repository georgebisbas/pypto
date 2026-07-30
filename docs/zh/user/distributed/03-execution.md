# 执行

一次性的编译并派发调用足以应付快速测试。生产代码则会将设置成本——
fork chip 进程、组装 kernel——分摊到可复用的 `DistributedWorker` 上的多次
派发中。

## DistributedWorker

通过 `compiled.prepare()` 获得。设置（fork、通信引导、kernel 组装）仅执行一次；
分发可执行多次。

```python
from pypto.runtime import DistributedWorker

with DistributedWorker(compiled) as rt:
    rt(host_x, host_out)
```

### 方法

| 方法 | 描述 |
| ---- | ---- |
| `compiled.prepare(config=None, callbacks=None)` | 创建 worker、fork 芯片进程，返回 `DistributedWorker`。 |
| `rt(x, y, z)` | 单次分发。 |
| `rt.run(compiled, x, y, z)` | 多程序分发。 |
| `rt.alloc_tensor(shape, dtype, *, init=None)` | 分配设备驻留 `DeviceTensor`。 |
| `rt.free_tensor(tensor)` | 释放 `DeviceTensor`。 |
| `rt.alloc_stacked_tensor(host_w)` | 沿 dim 0 分片 host_w。 |
| `rt.free_stacked_tensor(stacked)` | 释放所有分片。 |
| `rt.copy_stacked_from(stacked, host_out)` | D2H 读回每个分片。 |
| `rt.close()` | 释放 buffer，关闭芯片 worker。 |

## DeviceTensor

设备驻留 buffer，跨分发存活。

```python
with compiled.prepare() as rt:
    weight = rt.alloc_tensor((1024, 4096), torch.float16, init=host_weight)
    rt(x, weight, out)   # 无 H2D/D2H
```

## One-Shot vs 持久 Worker

### One-Shot

```python
from pypto.ir.distributed_compiled_program import DistributedConfig
from pypto import ir

dc = DistributedConfig(device_ids=[0, 1, 2, 3])
compiled = ir.compile(HelloAllReduce, platform="a2a3", distributed_config=dc)

inputs = torch.randn(4, 1, 256)
outputs = torch.zeros_like(inputs)
compiled(inputs, outputs)
```

### 持久 Worker

```python
host_x = torch.zeros((4, 1, 256), dtype=torch.float32).share_memory_()
host_out = torch.zeros_like(host_x).share_memory_()

with DistributedWorker(compiled) as rt:
    for step in steps:
        host_x.copy_(next_input(step))
        rt(host_x, host_out)
        consume(host_out)
```

> **致命陷阱：** 传入 `DistributedWorker` 的 IO buffer 在 `prepare()` 前必须调用
> `.share_memory_()`。若忘记，运行时在分发时拒绝该 buffer。

## CLI 启动

分布式程序的启动方式与单设备程序完全一样——直接 `python script.py`。
`DistributedConfig(device_ids=[...])` 决定 rank 数量和使用的设备；运行时
会从这一个 Python 进程中为每个设备 fork 出一个 worker 进程，因此不需要
调用单独的多进程启动器。

```bash
python script.py
```

## 环境变量

### 编译时宏

这些是 C 预处理器 `#define` 宏，**不是环境变量**。通过 CMake 标记设置。

| 宏 | 默认值 | 效果 |
| -- | ------ | ---- |
| `SIMPLER_HOST_STRACE` | `1`（开） | `benchmark()` 计时标记必需。 |
| `SIMPLER_DFX` | `1`（开） | 设备端分析总开关。 |

### 运行时环境变量

| 变量 | 默认值 | 效果 |
| ---- | ------ | ---- |
| `SIMPLER_DEVICE_STRACE_ENABLE` | 开 | 运行时切换设备域 `[STRACE]` 标记。 |

## 相关链接

- [00-model](00-model.md) — 快速开始和模型词汇
- [04-debugging](04-debugging.md) — 常见故障模式
