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

/**
 * @file broadcast.cpp
 * @brief Distributed tensor-level broadcast — pld.tensor.broadcast.
 *
 * Composite collective op: broadcast root rank's data to every rank.
 * Expands in ``LowerCompositeOps`` (pass 14) to a notify-all / wait-all
 * barrier followed by a ``pld.tile.remote_load`` from the root rank.
 *
 * IR signature:
 *
 *     pld.tensor.broadcast(target, signal, *, root: int)  -> DistributedTensorType
 *
 * ``root`` is a static int kwarg (the root rank is known at compile time).
 * The result type is ``target``'s :class:`DistributedTensorType` (in-place
 * rebind — every rank's slot holds root's data after the call).
 */

#include <any>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

TypePtr DeduceTensorBroadcastType(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 2) << "pld.tensor.broadcast requires exactly 2 positional arguments "
                             "(target, signal), but got "
                          << args.size();
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << "pld.tensor.broadcast positional argument #" << i << " must not be null";
  }

  auto target_type = As<DistributedTensorType>(args[0]->GetType());
  CHECK(target_type) << "pld.tensor.broadcast target must be a DistributedTensor (window-bound), got "
                     << args[0]->GetType()->TypeName();

  auto signal_type = As<DistributedTensorType>(args[1]->GetType());
  CHECK(signal_type) << "pld.tensor.broadcast signal must be a DistributedTensor (window-bound), got "
                     << args[1]->GetType()->TypeName();
  CHECK(signal_type->dtype_ == DataType::INT32)
      << "pld.tensor.broadcast signal must have INT32 element type, got dtype "
      << signal_type->dtype_.ToString();

  // Validate root kwarg.
  auto root_value = GetRequiredKwarg<int>(kwargs, "root", "pld.tensor.broadcast");
  CHECK(root_value >= 0) << "pld.tensor.broadcast root rank must be non-negative, got " << root_value;

  // Result type: same as target (in-place rebind — every rank's slot now
  // holds root's data).
  return args[0]->GetType();
}

}  // namespace

// ============================================================================
// pld.tensor.broadcast — broadcast root rank's data to all ranks
// ============================================================================

REGISTER_OP("pld.tensor.broadcast")
    .set_description(
        "Broadcast: replicate root rank's window-bound data to every rank in the comm group. "
        "`target` is a window-bound DistributedTensor (each rank writes its own data before the "
        "call; root's data is read and replicated by all non-root ranks). `signal` is a "
        "window-bound INT32 matrix used as the cross-rank barrier. `root` (int kwarg) selects "
        "the source rank. Lowered to notify-all / wait-all + remote_load by LowerCompositeOps; "
        "this op never survives past that pass.")
    .set_op_category("DistributedOp")
    .add_argument("target", "Window-bound DistributedTensor (InOut)")
    .add_argument("signal", "Window-bound INT32 DistributedTensor used as cross-rank barrier (InOut)")
    .set_attr<int>("root")
    .no_memory_spec()
    .f_deduce_type(DeduceTensorBroadcastType);

}  // namespace ir
}  // namespace pypto
