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
 * @file reduce_scatter.cpp
 * @brief Distributed tensor-level reduce-scatter — pld.tensor.reduce_scatter.
 *
 * Composite collective op: element-wise reduce chunks across ranks, then
 * scatter — each rank receives one reduced chunk.  The target
 * DistributedTensor has shape [NR, SIZE] (one row per chunk).  Each rank
 * stages all NR chunks before the call; after the call, rank r's row
 * [r, 0:SIZE] holds the reduced value of chunk r.
 *
 * IR signature:
 *
 *     pld.tensor.reduce_scatter(target, signal, *, op: int)  -> DistributedTensorType
 *
 * Uses the same ``ReduceOp`` enum and 5-phase decomposition
 * (notify/wait/accumulate/notify/wait/store) as allreduce.
 */

#include <any>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/comm.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

TypePtr DeduceTensorReduceScatterType(const std::vector<ExprPtr>& args,
                                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 2) << "pld.tensor.reduce_scatter requires exactly 2 positional arguments "
                             "(target, signal), but got "
                          << args.size();
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << "pld.tensor.reduce_scatter positional argument #" << i << " must not be null";
  }

  auto target_type = As<DistributedTensorType>(args[0]->GetType());
  CHECK(target_type) << "pld.tensor.reduce_scatter target must be a DistributedTensor (window-bound), got "
                     << args[0]->GetType()->TypeName();
  CHECK(target_type->shape_.size() == 2) << "pld.tensor.reduce_scatter target must be 2D [NR, SIZE], got "
                                         << target_type->shape_.size() << " dims";

  auto signal_type = As<DistributedTensorType>(args[1]->GetType());
  CHECK(signal_type) << "pld.tensor.reduce_scatter signal must be a DistributedTensor (window-bound), got "
                     << args[1]->GetType()->TypeName();
  CHECK(signal_type->dtype_ == DataType::INT32)
      << "pld.tensor.reduce_scatter signal must have INT32 element type, got dtype "
      << signal_type->dtype_.ToString();

  // Validate op kwarg — kSum only for first version (same as allreduce).
  auto op_value = GetRequiredKwarg<int>(kwargs, "op", "pld.tensor.reduce_scatter");
  CHECK(op_value == static_cast<int>(ReduceOp::kSum))
      << "pld.tensor.reduce_scatter op must be ReduceOp.Sum (got int " << op_value
      << "); Max / Min / Prod lowerings are not yet implemented";

  // Result type: same as target (in-place rebind — rank r's row now holds
  // the reduced chunk r).
  return args[0]->GetType();
}

}  // namespace

// ============================================================================
// pld.tensor.reduce_scatter — reduce + scatter chunks across ranks
// ============================================================================

REGISTER_OP("pld.tensor.reduce_scatter")
    .set_description(
        "Reduce-scatter: element-wise reduce chunks across all ranks, then scatter so each "
        "rank receives one reduced chunk. `target` has shape [NR, SIZE] — each rank stages "
        "all NR chunks before the call. After the call, rank r's row [r, 0:SIZE] holds the "
        "reduced value of chunk r. `signal` is a window-bound INT32 matrix for the cross-rank "
        "barrier. `op` selects the reduction operator (Sum only in first version). Lowered to "
        "a 5-phase decomposition by LowerCompositeOps; this op never survives past that pass.")
    .set_op_category("DistributedOp")
    .add_argument("target", "Window-bound DistributedTensor[NR, SIZE] (InOut)")
    .add_argument("signal", "Window-bound INT32 DistributedTensor used as cross-rank barrier (InOut)")
    .set_attr<int>("op")
    .no_memory_spec()
    .f_deduce_type(DeduceTensorReduceScatterType);

}  // namespace ir
}  // namespace pypto
