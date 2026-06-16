# ResolveDistributedShapeVars

> **编号说明**：该 pass 运行于 Default 策略中 `CtrlFlowTransform`（03）和
> `ConvertToSSA`（04）之间。从 04 开始的重编号已推迟 —— 待办：将
> `convert_to_ssa.md` → 05、`simplify.md` → 06 等，并将本文档插入为
> `04-resolve_distributed_shape_vars.md`。

## 目的

在 SSA 转换之前，解析分布式 InCore 函数中的 `pl.nranks_dim` 形状维度。

## 工作方式

1. **门控**：仅处理至少有一个 `DistributedTensor` 参数的 InCore 函数 —
   其他函数原样保留。

2. **查找 nranks**：扫描函数体找到 `nranks = pld.nranks(ctx)`。该
   `nranks` Var 是函数体内部定义的、持有运行时 rank 数量的唯一变量。

3. **重写形状**：遍历每个参数和返回类型。对于每个形状维度中
   `is_nranks_dim_ = true` 的 `Var`（由 `pl.nranks_dim` 设置），替换为
   函数内部定义的 `nranks` Var。

4. **误用检查**：在重写类型形状后，遍历函数体查找表达式位置中残留的
   `is_nranks_dim_` Var。若发现，则抛出面向用户的错误提示：该维度被
   用于 tile 操作而非类型注解。

## 为何要在 SSA 之前运行

替换在 `ConvertToSSA` **之前**运行。否则，类型注解中的 Var 将作为未定
义值进入 SSA 作用域，导致 SSA 转换失败或生成错误的 IR。替换后，
`nranks` Var 成为正常的函数体定义的 SSA 值。

## 复杂度

O(N) 单遍扫描 —— `FindNranksVar` 一次、形状重写一次、误用检查一次，
均为线性于函数体大小。
