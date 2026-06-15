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
 * @file allgather.cpp
 * @brief Distributed tensor-level allgather — pld.tensor.allgather.
 *
 * Composite collective op: gather data from all ranks into every rank's
 * window.  The target DistributedTensor has shape [NR, SIZE] (one row per
 * rank).  Each rank stages its data in its own row before the call; the
 * intrinsic remote_loads every other rank's row and stores it locally,
 * so every rank ends up with the full gathered dataset.
 *
 * IR signature:
 *
 *     pld.tensor.allgather(target, signal)  -> DistributedTensorType (rebind of target)
 *
 * LowerCompositeOps expands this into a notify-all / wait-all barrier
 * followed by per-peer remote_load + tile.store.  Single barrier — no
 * post-gather barrier needed (read-only after staging).
 */

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

TypePtr DeduceTensorAllGatherType(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs) {
  (void)kwargs;
  CHECK(args.size() == 2) << "pld.tensor.allgather requires exactly 2 positional arguments "
                             "(target, signal), but got "
                          << args.size();
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << "pld.tensor.allgather positional argument #" << i << " must not be null";
  }

  auto target_type = As<DistributedTensorType>(args[0]->GetType());
  CHECK(target_type) << "pld.tensor.allgather target must be a DistributedTensor (window-bound), got "
                     << args[0]->GetType()->TypeName();
  CHECK(target_type->shape_.size() == 2)
      << "pld.tensor.allgather target must be 2D [NR, SIZE], got " << target_type->shape_.size() << " dims";

  auto signal_type = As<DistributedTensorType>(args[1]->GetType());
  CHECK(signal_type) << "pld.tensor.allgather signal must be a DistributedTensor (window-bound), got "
                     << args[1]->GetType()->TypeName();
  CHECK(signal_type->dtype_ == DataType::INT32)
      << "pld.tensor.allgather signal must have INT32 element type, got dtype "
      << signal_type->dtype_.ToString();

  // Result type: same as target (in-place rebind — every rank's local copy
  // now holds all gathered rows).
  return args[0]->GetType();
}

}  // namespace

// ============================================================================
// pld.tensor.allgather — gather data from every rank into every rank's window
// ============================================================================

REGISTER_OP("pld.tensor.allgather")
    .set_description(
        "All-gather: gather data from all ranks into every rank's window-bound "
        "DistributedTensor. `target` has shape [NR, SIZE] — each rank stages its chunk in "
        "its own row before the call. After the call every rank's local copy of `target` "
        "holds the data from all ranks. `signal` is a window-bound INT32 matrix used as "
        "the cross-rank barrier. Lowered to notify-all / wait-all + per-peer remote_load "
        "+ tile.store by LowerCompositeOps; this op never survives past that pass.")
    .set_op_category("DistributedOp")
    .add_argument("target", "Window-bound DistributedTensor[NR, SIZE] (InOut)")
    .add_argument("signal", "Window-bound INT32 DistributedTensor used as cross-rank barrier (InOut)")
    .no_memory_spec()
    .f_deduce_type(DeduceTensorAllGatherType);

}  // namespace ir
}  // namespace pypto
