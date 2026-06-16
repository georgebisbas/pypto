/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <memory>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace pass {

namespace {

// ---------------------------------------------------------------------------
// Pre-SSA resolution of pl.nranks_dim shape Vars in distributed functions.
//
// Vars created by pl.nranks_dim carry is_nranks_dim_ = true.  This pass
// replaces them with the function-local ``nranks = pld.nranks(ctx)`` Var,
// so the dimension resolves to the runtime rank count before tile lowering.
//
// Runs BEFORE ConvertToSSA.  Only InCore functions with DistributedTensor
// params are processed.  ConstInt dims are never affected.
// ---------------------------------------------------------------------------

bool HasDistributedTensorParam(const FunctionPtr& func) {
  for (const auto& param : func->params_) {
    if (As<DistributedTensorType>(param->GetType())) return true;
  }
  return false;
}

VarPtr FindNranksVar(const FunctionPtr& func) {
  if (!func->body_) return nullptr;

  struct Finder : public IRVisitor {
    VarPtr found_;
    void VisitStmt_(const AssignStmtPtr& op) override {
      auto call = As<Call>(op->value_);
      if (call && call->op_ && call->op_->name_ == "pld.nranks") {
        found_ = op->var_;
        return;
      }
      IRVisitor::VisitStmt_(op);
    }
  };

  Finder finder;
  finder.VisitFunction(func);
  return finder.found_;
}

/// Replace any shape dim that carries is_nranks_dim_ with the nranks Var.
TypePtr RewriteShapedType(const std::shared_ptr<const ShapedType>& shaped, const VarPtr& nranks_var) {
  bool changed = false;
  std::vector<ExprPtr> new_shape;
  for (const auto& dim : shaped->shape_) {
    auto var = As<Var>(dim);
    if (var && var->is_nranks_dim_) {
      new_shape.push_back(nranks_var);
      changed = true;
    } else {
      new_shape.push_back(dim);
    }
  }
  if (!changed) return shaped;

  if (auto tt = As<TensorType>(shaped)) {
    return std::make_shared<TensorType>(new_shape, tt->dtype_, tt->memref_, tt->tensor_view_);
  }
  if (auto dt = As<DistributedTensorType>(shaped)) {
    return std::make_shared<DistributedTensorType>(new_shape, dt->dtype_);
  }
  return shaped;
}

TypePtr RewriteType(const TypePtr& type, const VarPtr& nranks_var) {
  if (auto shaped = As<ShapedType>(type)) return RewriteShapedType(shaped, nranks_var);
  return type;
}

/// Check that no is_nranks_dim_ Var survives in expression positions.
/// Type-shape Vars were already rewritten — any remaining in the body
/// must be user misuse (e.g. pl.load(data, [NR, 0], [1, 128])).
void CheckBodyForNranksDimMisuse(const FunctionPtr& func) {
  struct Checker : public IRVisitor {
    void VisitExpr_(const VarPtr& op) override {
      CHECK_SPAN(!op->is_nranks_dim_, op->span_) << "pl.nranks_dim must only appear in type annotations, not "
                                                 << "in expressions.  Use pl.range(pld.nranks(ctx)) for "
                                                 << "dynamic loop bounds.";
      IRVisitor::VisitExpr_(op);
    }
  };
  Checker c;
  c.VisitFunction(func);
}

}  // namespace

Pass ResolveDistributedShapeVars() {
  return CreateFunctionPass(
      [](const FunctionPtr& func) -> FunctionPtr {
        if (!func || !func->body_) return func;
        if (!IsInCoreType(func->func_type_)) return func;
        if (!HasDistributedTensorParam(func)) return func;

        auto nranks_var = FindNranksVar(func);
        if (!nranks_var) return func;

        bool params_changed = false;
        std::vector<VarPtr> new_params;
        for (const auto& param : func->params_) {
          auto new_type = RewriteType(param->GetType(), nranks_var);
          if (new_type.get() == param->GetType().get()) {
            new_params.push_back(param);
          } else {
            params_changed = true;
            new_params.push_back(std::make_shared<Var>(param->name_hint_, new_type, param->span_));
          }
        }

        bool returns_changed = false;
        std::vector<TypePtr> new_returns;
        for (const auto& rt : func->return_types_) {
          auto new_rt = RewriteType(rt, nranks_var);
          new_returns.push_back(new_rt);
          if (new_rt.get() != rt.get()) returns_changed = true;
        }

        CheckBodyForNranksDimMisuse(func);

        if (!params_changed && !returns_changed) return func;

        return std::make_shared<Function>(func->name_, new_params, func->param_directions_, new_returns,
                                          func->body_, func->span_, func->func_type_, func->level_,
                                          func->role_, func->attrs_);
      },
      "ResolveDistributedShapeVars", PassProperties{});
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
