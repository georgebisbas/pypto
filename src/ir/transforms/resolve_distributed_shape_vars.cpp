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
#include <set>
#include <string>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace pass {

namespace {

// ---------------------------------------------------------------------------
// Pre-SSA resolution of dynamic shape Vars in distributed functions.
//
// Runs BEFORE ConvertToSSA so type-shape Vars never enter SSA scope.
// The algorithm is structural — no DimExpr node, no name matching:
//
//   1. Collect every Var defined in the function body (AssignStmt LHS).
//   2. Find the nranks Var from ``nranks = pld.nranks(ctx)``.
//   3. Walk parameter/return type shapes; any shape dim that is a Var
//      NOT in the body-def set is a type-only placeholder → replace
//      with the nranks Var.
//
// Only InCore functions with DistributedTensor params are processed.
// Static tile shapes (ConstInt) are never affected.
// ---------------------------------------------------------------------------

/// Scan the function body and collect all Var names that appear as
/// the LHS of an AssignStmt (pre-SSA definitions).
std::set<std::string> CollectBodyDefNames(const FunctionPtr& func) {
  std::set<std::string> names;
  if (!func->body_) return names;

  struct DefCollector : public IRVisitor {
    std::set<std::string>& names_;
    explicit DefCollector(std::set<std::string>& n) : names_(n) {}

    void VisitStmt_(const AssignStmtPtr& op) override {
      if (op->var_) names_.insert(op->var_->name_hint_);
      IRVisitor::VisitStmt_(op);
    }
  };

  DefCollector collector(names);
  collector.VisitFunction(func);
  return names;
}

/// Check whether a function participates in distributed communication.
bool HasDistributedTensorParam(const FunctionPtr& func) {
  for (const auto& param : func->params_) {
    if (auto overridden = param->GetOverrideType()) {
      if (As<DistributedTensorType>(overridden)) return true;
    }
    if (As<DistributedTensorType>(param->GetType())) return true;
    if (auto ptr = As<PtrType>(param->GetType())) {
      if (As<DistributedTensorType>(ptr->pointee_type_)) return true;
    }
  }
  return false;
}

/// Find the Var that holds the result of ``pld.nranks(ctx)``.
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

/// Return the ShapedType's shape dims, unwrapping Ptr wrappers.
const std::vector<ExprPtr>* GetShapeDims(const TypePtr& type) {
  if (auto st = As<ShapedType>(type)) return &st->shape_;
  if (auto pt = As<PtrType>(type)) return GetShapeDims(pt->pointee_type_);
  return nullptr;
}

/// Rewrite a single type, replacing type-only Vars with nranks_var.
TypePtr RewriteType(const TypePtr& type, const std::set<std::string>& body_def_names,
                    const VarPtr& nranks_var) {
  const auto* shape = GetShapeDims(type);
  if (!shape) return type;

  bool changed = false;
  std::vector<ExprPtr> new_shape;
  for (const auto& dim : *shape) {
    auto var = As<Var>(dim);
    if (var && body_def_names.find(var->name_hint_) == body_def_names.end()) {
      // Var is NOT defined in the body → type-only placeholder → replace.
      new_shape.push_back(nranks_var);
      changed = true;
    } else {
      new_shape.push_back(dim);
    }
  }
  if (!changed) return type;

  // Reconstruct the appropriate concrete type.
  if (auto tt = As<TensorType>(type)) {
    return std::make_shared<TensorType>(new_shape, tt->dtype_, tt->memref_, tt->tensor_view_);
  }
  if (auto dt = As<DistributedTensorType>(type)) {
    return std::make_shared<DistributedTensorType>(new_shape, dt->dtype_, dt->distributed_comm_type_);
  }
  // Pointer wrapper (e.g. Ptr<TensorType> used by InOut/Out).
  if (auto pt = As<PtrType>(type)) {
    auto new_pointee = RewriteType(pt->pointee_type_, body_def_names, nranks_var);
    return std::make_shared<PtrType>(new_pointee, pt->span_);
  }
  return type;
}

/// Rewrite a Var's type, including overridden type (InOut/Out wrappers).
VarPtr RewriteParam(const VarPtr& param, const std::set<std::string>& body_def_names,
                    const VarPtr& nranks_var) {
  auto new_type = RewriteType(param->GetType(), body_def_names, nranks_var);
  auto overridden = param->GetOverrideType();
  auto new_overridden = overridden ? RewriteType(overridden, body_def_names, nranks_var) : nullptr;

  if (new_type.get() == param->GetType().get() &&
      (!overridden || new_overridden.get() == overridden.get())) {
    return param;
  }

  auto new_param = std::make_shared<Var>(param->name_hint_, new_type, param->span_);
  if (new_overridden) new_param->SetOverrideType(new_overridden);
  return new_param;
}

}  // namespace

Pass ResolveDistributedShapeVars() {
  auto pass_func = [](const FunctionPtr& func) -> FunctionPtr {
    if (!func || !func->body_) return func;
    if (!IsInCoreType(func->func_type_)) return func;
    if (!HasDistributedTensorParam(func)) return func;

    auto nranks_var = FindNranksVar(func);
    if (!nranks_var) return func;

    auto body_def_names = CollectBodyDefNames(func);

    // Rewrite parameters.
    bool params_changed = false;
    std::vector<VarPtr> new_params;
    for (const auto& param : func->params_) {
      auto new_param = RewriteParam(param, body_def_names, nranks_var);
      new_params.push_back(new_param);
      if (new_param.get() != param.get()) params_changed = true;
    }

    // Rewrite return types.
    bool returns_changed = false;
    std::vector<TypePtr> new_returns;
    for (const auto& rt : func->return_types_) {
      auto new_rt = RewriteType(rt, body_def_names, nranks_var);
      new_returns.push_back(new_rt);
      if (new_rt.get() != rt.get()) returns_changed = true;
    }

    if (!params_changed && !returns_changed) return func;

    return std::make_shared<Function>(func->name_, new_params, func->param_directions_, new_returns,
                                      func->body_, func->span_, func->func_type_, func->level_,
                                      func->role_, func->attrs_);
  };

  // Requires nothing — runs pre-SSA; produces and invalidates nothing.
  static const PassProperties kProperties{};
  return CreateFunctionPass(pass_func, "ResolveDistributedShapeVars", kProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
