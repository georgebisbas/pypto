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
 * Composite collective op: gather data from all ranks and return the
 * concatenated result as a local Tile.  The intrinsic accepts the local
 * chunk (Tile), a DistributedTensor staging window [NR, SIZE], and a
 * signal window.  It stages the local chunk, synchronises, and assembles
 * the gathered data via remote_load + tile.concat.
 *
 * IR signature:
 *
 *     pld.tensor.allgather(local_data, target, signal)  -> TileType
 *
 * LowerCompositeOps expands this into:
 *   - tile.store(local_data, [my_rank, 0], target)
 *   - notify-all / wait-all
 *   - remote_load + tile.concat for every peer
 * This Call never survives past that pass.
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
  CHECK(args.size() == 3) << "pld.tensor.allgather requires exactly 3 positional arguments "
                             "(local_data, target, signal), but got "
                          << args.size();
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i]) << "pld.tensor.allgather positional argument #" << i << " must not be null";
  }

  // arg 0: local_data — Tile [1, SIZE] with this rank's chunk
  auto local_type = As<TileType>(args[0]->GetType());
  CHECK(local_type) << "pld.tensor.allgather local_data must be a Tile, got "
                    << args[0]->GetType()->TypeName();

  // arg 1: target — DistributedTensor [NR, SIZE] staging window
  auto target_type = As<DistributedTensorType>(args[1]->GetType());
  CHECK(target_type) << "pld.tensor.allgather target must be a DistributedTensor (window-bound), got "
                     << args[1]->GetType()->TypeName();
  CHECK(target_type->shape_.size() == 2)
      << "pld.tensor.allgather target must be 2D [NR, SIZE], got " << target_type->shape_.size() << " dims";

  // arg 2: signal — DistributedTensor INT32
  auto signal_type = As<DistributedTensorType>(args[2]->GetType());
  CHECK(signal_type) << "pld.tensor.allgather signal must be a DistributedTensor (window-bound), got "
                     << args[2]->GetType()->TypeName();
  CHECK(signal_type->dtype_ == DataType::INT32)
      << "pld.tensor.allgather signal must have INT32 element type, got dtype "
      << signal_type->dtype_.ToString();

  // Result: Tile with the same shape as target (NR rows × SIZE cols).
  // The lowering assembles gathered chunks into a Tile via tile.concat.
  return std::make_shared<TileType>(target_type->shape_, target_type->dtype_);
}

}  // namespace

// ============================================================================
// pld.tensor.allgather — gather data from every rank into every rank's window
// ============================================================================

REGISTER_OP("pld.tensor.allgather")
    .set_description(
        "All-gather: gather data from all ranks, returning the concatenated result as "
        "a local Tile. `local_data` is the rank's chunk (Tile [1, SIZE]). `target` is "
        "a window-bound DistributedTensor[NR, SIZE] used as the staging area. `signal` is "
        "a window-bound INT32 DistributedTensor used as the cross-rank barrier. Lowered to "
        "tile.store + notify-all / wait-all + remote_load + tile.concat by LowerCompositeOps; "
        "this op never survives past that pass.")
    .set_op_category("DistributedOp")
    .add_argument("local_data", "Local tile [1, SIZE] — this rank's data (Input)")
    .add_argument("target", "Window-bound DistributedTensor[NR, SIZE] (InOut)")
    .add_argument("signal", "Window-bound INT32 DistributedTensor used as cross-rank barrier (InOut)")
    .no_memory_spec()
    .f_deduce_type(DeduceTensorAllGatherType);

}  // namespace ir
}  // namespace pypto
