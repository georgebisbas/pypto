# ResolveDistributedShapeVars

> **Numbering note**: this pass runs between `CtrlFlowTransform` (03) and
> `ConvertToSSA` (04) in the Default strategy.  Renumbering from 04 onward
> is deferred — TODO: renumber `convert_to_ssa.md` → 05, `simplify.md` →
> 06, etc., and insert this file as `04-resolve_distributed_shape_vars.md`.

## Purpose

Resolves `pl.nranks_dim` shape dimensions in distributed InCore functions
before SSA conversion.

## How it works

1. **Gate**: only processes InCore functions with at least one
   `DistributedTensor` parameter — all other functions are untouched.

2. **Find nranks**: scans the function body for `nranks = pld.nranks(ctx)`.
   This `nranks` Var is the single body-defined variable that holds the
   runtime rank count.

3. **Rewrite shapes**: walks every parameter and return type.  For each
   shape dimension that is a `Var` with `is_nranks_dim_ = true` (set by
   `pl.nranks_dim`), replaces it with the function-local `nranks` Var.

4. **Misuse check**: after rewriting type shapes, walks the function body
   for any remaining `is_nranks_dim_` Var in expression position.  If found,
   emits a user-facing error: the dimension was used in a tile operation
   instead of a type annotation.

## Why pre-SSA

The replacement runs **before** `ConvertToSSA`.  Without it, the
type-annotation Vars would enter SSA scope as undefined values, causing SSA
conversion to fail or producing incorrect IR.  After replacement, the
`nranks` Var is a normal body-defined SSA value.

## Complexity

O(N) single-pass — one visitor for `FindNranksVar`, one for shape rewriting,
one for misuse checking.  All three are linear in the function body size.
