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

/// Check whether a function participates in distributed communication by
/// inspecting its parameter types for DistributedTensor.
bool HasDistributedTensorParam(const FunctionPtr& func) {
  for (const auto& param : func->params_) {
    auto type = param->GetType();
    // Check the type directly (for DistributedTensor params) and also
    // check if it's a Ptr wrapping a DistributedTensor.
    if (As<DistributedTensorType>(type)) return true;
    if (auto ptr_type = As<PtrType>(type)) {
      if (As<DistributedTensorType>(ptr_type->pointee_type_)) return true;
    }
    // InOut/Out wrappers carry the type via overridden type on the Var.
    if (auto overridden = param->GetOverrideType()) {
      if (As<DistributedTensorType>(overridden)) return true;
    }
  }
  return false;
}

/// Find the Var that holds `pld.nranks(ctx)` by scanning for the pattern:
///     nranks = call(pld.nranks, ctx)
/// Returns the LHS Var if found, nullptr otherwise.
VarPtr FindNranksVar(const FunctionPtr& func) {
  if (!func->body_) return nullptr;

  // Walk the function body looking for: <var> = call("pld.nranks", ctx)
  struct NranksFinder : public IRVisitor {
    VarPtr found_;

    void VisitStmt_(const AssignStmtPtr& op) override {
      auto call = As<Call>(op->value_);
      if (!call || !call->op_) {
        IRVisitor::VisitStmt_(op);
        return;
      }
      if (call->op_->name_ == "pld.nranks") {
        // The LHS var holds the nranks value.
        found_ = op->var_;
        return;  // Found it — stop recursing.
      }
      IRVisitor::VisitStmt_(op);
    }
  };

  NranksFinder finder;
  finder.VisitFunction(func);
  return finder.found_;
}

/// Replace all DimExpr nodes in type shapes with the given replacement Var.
///
/// Walks all function params and return types.  For each ShapedType
/// (TensorType, DistributedTensorType), replaces every DimExpr dimension
/// with the replacement Var.  Leaves static ConstInt dims untouched.
///
/// This is structural — no name matching.  Every DimExpr in a distributed
/// function resolves to the rank count.
class DimExprResolver : public IRMutator {
 public:
  explicit DimExprResolver(VarPtr nranks_var) : nranks_var_(std::move(nranks_var)) {}

  FunctionPtr Resolve(const FunctionPtr& func) {
    // Rewrite param types.
    std::vector<VarPtr> new_params;
    bool params_changed = false;
    for (const auto& param : func->params_) {
      auto new_type = RewriteType(param->GetType());
      if (new_type.get() != param->GetType().get()) {
        params_changed = true;
        auto new_param = std::make_shared<Var>(param->name_hint_, new_type, param->span_);
        if (auto overridden = param->GetOverrideType()) {
          auto new_override = RewriteType(overridden);
          new_param->SetOverrideType(new_override);
        }
        new_params.push_back(std::move(new_param));
      } else {
        new_params.push_back(param);
      }
    }

    // Rewrite return types.
    std::vector<TypePtr> new_return_types;
    bool returns_changed = false;
    for (const auto& ret_type : func->return_types_) {
      auto new_type = RewriteType(ret_type);
      new_return_types.push_back(new_type);
      if (new_type.get() != ret_type.get()) returns_changed = true;
    }

    // Rewrite body (handles type annotations nested in stmts/exprs).
    auto new_body = VisitStmt(func->body_);
    bool body_changed = (new_body.get() != func->body_.get());

    if (!params_changed && !returns_changed && !body_changed) return func;

    return std::make_shared<Function>(func->name_, new_params, func->param_directions_,
                                      new_return_types, new_body, func->span_, func->func_type_,
                                      func->level_, func->role_, func->attrs_);
  }

 protected:
  /// Rewrite a single type, replacing DimExpr dims.
  TypePtr RewriteType(const TypePtr& type) {
    if (!type) return type;

    // Handle TensorType
    if (auto tensor_type = As<TensorType>(type)) {
      return RewriteShapedType(tensor_type);
    }
    // Handle DistributedTensorType
    if (auto dist_type = As<DistributedTensorType>(type)) {
      return RewriteShapedType(dist_type);
    }
    // Handle PtrType (wraps another type)
    if (auto ptr_type = As<PtrType>(type)) {
      auto new_pointee = RewriteType(ptr_type->pointee_type_);
      if (new_pointee.get() != ptr_type->pointee_type_.get()) {
        return std::make_shared<PtrType>(new_pointee, ptr_type->span_);
      }
      return type;
    }
    // Handle TupleType (for multi-return)
    if (auto tuple_type = As<TupleType>(type)) {
      std::vector<TypePtr> new_elements;
      bool changed = false;
      for (const auto& elem : tuple_type->elements_) {
        auto new_elem = RewriteType(elem);
        new_elements.push_back(new_elem);
        if (new_elem.get() != elem.get()) changed = true;
      }
      if (changed) {
        return std::make_shared<TupleType>(new_elements, tuple_type->span_);
      }
      return type;
    }
    return type;
  }

  /// Rewrite a ShapedType (TensorType or DistributedTensorType).
  TypePtr RewriteShapedType(const std::shared_ptr<const ShapedType>& shaped) {
    // Walk shape dims and replace DimExpr nodes.
    bool changed = false;
    std::vector<ExprPtr> new_shape;
    for (const auto& dim : shaped->shape_) {
      auto new_dim = RewriteDim(dim);
      new_shape.push_back(new_dim);
      if (new_dim.get() != dim.get()) changed = true;
    }
    if (!changed) return shaped;

    // Reconstruct the appropriate type.
    if (auto tensor_type = As<TensorType>(shaped)) {
      return std::make_shared<TensorType>(new_shape, tensor_type->dtype_, tensor_type->memref_,
                                           tensor_type->tensor_view_);
    }
    if (auto dist_type = As<DistributedTensorType>(shaped)) {
      return std::make_shared<DistributedTensorType>(new_shape, dist_type->dtype_,
                                                      dist_type->distributed_comm_type_);
    }
    // Fallback: should not happen for ShapedType subclasses.
    return shaped;
  }

  /// Rewrite a single shape dimension.
  /// If it's a DimExpr, replace with nranks_var_.
  /// If it's a DimExpr wrapping a composite (e.g. Mul), replace the whole
  /// DimExpr node with nranks_var_.
  ExprPtr RewriteDim(const ExprPtr& dim) {
    if (auto dim_expr = As<DimExpr>(dim)) {
      // Structural inference: every DimExpr in a distributed function
      // resolves to the rank count.  The name inside (if any) is ignored.
      return nranks_var_;
    }
    return dim;
  }

  VarPtr nranks_var_;
};

}  // namespace

Pass InferDistributedDimBindings() {
  auto pass_func = [](const FunctionPtr& func) -> FunctionPtr {
    if (!func || !func->body_) return func;
    if (!IsInCoreType(func->func_type_)) return func;

    // Only process functions that participate in distributed communication.
    if (!HasDistributedTensorParam(func)) return func;

    // Find the nranks variable from pld.nranks(ctx).
    auto nranks_var = FindNranksVar(func);
    if (!nranks_var) return func;  // No nranks call — nothing to resolve.

    DimExprResolver resolver(nranks_var);
    return resolver.Resolve(func);
  };

  // This pass needs the IR in SSA form (so Var identity is stable) and
  // runs before tile lowering (so types are still TensorType, not TileType).
  // It produces nothing new and invalidates nothing — it only replaces
  // DimExpr nodes with existing Vars.
  static const PassProperties kProperties{
      .required = {IRProperty::SSAForm},
      .produced = {},
      .invalidated = {},
  };

  return CreateFunctionPass(pass_func, "InferDistributedDimBindings", kProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
