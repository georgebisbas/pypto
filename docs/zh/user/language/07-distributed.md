# 分布式

`pld` 在 `pl` 之上添加了什么语言层面的东西：它自己的类型、它自己的命名空间划分，
以及选择它的规则。本页不重复那七个 `pl` 页面——`pld` 是在 `pl` 之上组合而成的，
你已经了解的类型与控制流原样适用。本页也不重新讲解模型或集合通信——见
[分布式编程](../distributed/index.md)。

> **前置知识：** [类型](00-types.md)、[函数与程序](01-functions.md)。

## 何时使用 `pld`

`pld` 的选择标准是**数据做什么，而不是数据在哪里**：如果操作跨越 rank，就用
`pld`，否则用 `pl`。命名空间的切分与 `pl` 一致——`pld.tile.*` 用于 InCore
函数内部，`pld.tensor.*` 用于编排层对整张张量的操作，再加上第三个家族
`pld.system.*`，用于 `pl` 没有对应物的 rank/信号原语。完整规则与带 2 段式简写
的完整名字列表见[选择命名空间](../ops/00-dispatch.md)。

## `DistributedTensor`

`pld.DistributedTensor` 是大多数 `pld` 代码会接触到的唯一新类型。它继承自
`pl.Tensor`——同样的 DSL 表面，同样的 `memref`/layout/`tensor_view` 下标位，
`pl.load`/`pl.store` 等操作对它同样透明地生效（作用在你自己 rank 的那一份切片
上）。不同的是它背后的存储：一个对称 HCCL **window buffer** 中由
`CommDomainScopeStmt` 划出的 per-rank 切片。在 IR 层面，
`ir.DistributedTensorType` 是 `TensorType` 的一个 precise-`ObjectKind` 子类，
因此每个跨 rank 算子的 verifier 都能直接拒绝一个普通的 `Tensor` 参数——传错
类型是编译期错误，而不是运行期的意外。

`DistributedTensor` 由 `pld.tensor.window(...)` 产生，它是对
`pld.tensor.alloc_window_buffer(...)` 所分配缓冲区的一个视图。分配 → 视图化
的完整走查，以及"对称"到底保证了什么（每个 rank 的大小与布局相同，而非共享
地址），见 [Window Buffer](../distributed/08-window_buffer.md)。

## `CommCtx` 与 rank/world 查询

`pld.CommCtx` 是一个通信上下文句柄，遵循与 `pl.Ptr` 相同的双重角色模式：既可
作为类型标注（`ctx: pld.CommCtx`），也可作为 `pld.get_comm_ctx(...)` 的返回
值，包装底层的 IR 调用。查询面由四个函数构成：

| 调用 | 接受 | 作用域 |
| ---- | ---- | ------ |
| `pld.world_size()` | 无 | 仅 Host |
| `pld.get_comm_ctx(dist_tensor)` | 一个窗口绑定的 `DistributedTensor` | Host 与 InCore |
| `pld.rank(ctx)` | 一个 `CommCtx` | Host 与 InCore |
| `pld.nranks(ctx)` | 一个 `CommCtx` | Host 与 InCore |

注意 `get_comm_ctx` 接受的是一个 `DistributedTensor`，不是凭空而来——它通过
传入的窗口绑定张量去查找上下文。而 `rank`/`nranks` 接受的是那个 `CommCtx`，
不是张量。

## 三个命名空间与简写形式

`pld.tensor.*`、`pld.tile.*` 与 `pld.system.*` 是三个命名空间（对应
`pl.tensor.*`/`pl.tile.*`/`pl.system.*` 的切分）。少数几个名字还有按名字派发
的 2 段式简写，就像 `pl.add` 会派发到 `pl.tile.add` 或 `pl.tensor.add` 一样：
`world_size`、`alloc_window_buffer`、`window`、`get_comm_ctx`、`rank`、
`nranks`、`remote_load`、`remote_store`——正好这八个，不多不少。其余每个
`pld` 操作（集合通信、`put`/`get`、信号原语）都需要完整的 3 段式写法。

## 枚举

`pld` 携带五个 ABI 枚举：`NotifyOp`、`WaitCmp`、`ReduceOp`、`AtomicType`，
以及 `CommEngine`。它们是**只能追加的 ABI**——新取值只能加在末尾，不能插入或
重新编号，因为它们底层的 `int` 值被序列化进了编译后的 IR。每个枚举的取值及含义见
[原语](../distributed/02-primitives.md)页的"类型与枚举"一节；本页只是把它们
列为类型面的一部分，不重复那张表。

## 接下来看什么

- [分布式编程](../distributed/index.md) —— 模型、集合通信，以及 16 步教程阶梯。
- [Distributed Operators](../../dev/distributed_ops.md) —— 算子语义、ABI
  枚举的确切编号，以及 barrier 信号协议。
- [算子目录](../ops/01-catalog.md) —— 每个 `pld` 算子及其签名页。

## 相关链接

- [选择命名空间](../ops/00-dispatch.md) —— `pl` 与 `pld` 的完整取舍规则。
- [编程模型](../03-programming-model.md) —— `Level.HOST` + `Role.Orchestrator`，
  标记分布式程序的 host 编排器。
