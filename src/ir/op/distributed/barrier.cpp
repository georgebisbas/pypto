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
 * @file barrier.cpp
 * @brief Distributed tensor-level barrier — pld.tensor.barrier.
 *
 * Composite collective op: cross-rank barrier using a window-bound INT32
 * ``signal`` matrix. Expands in ``LowerCompositeOps`` (pass 14) to a
 * notify-all / wait-all sequence. Pure synchronization — no data movement.
 *
 * IR signature:
 *
 *     pld.tensor.barrier(signal)  -> DistributedTensorType (rebind of signal)
 *
 * The returned type is ``signal``'s :class:`DistributedTensorType` so the
 * rebind idiom (``sig = pld.tensor.barrier(sig)``) is consistent with
 * ``pld.tensor.allreduce``.
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

TypePtr DeduceTensorBarrierType(const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs) {
  (void)kwargs;
  CHECK(args.size() == 1) << "pld.tensor.barrier requires exactly 1 positional argument (signal), but got "
                          << args.size();
  CHECK(args[0]) << "pld.tensor.barrier positional argument #0 must not be null";

  auto signal_type = As<DistributedTensorType>(args[0]->GetType());
  CHECK(signal_type) << "pld.tensor.barrier signal must be a DistributedTensor (window-bound), got "
                     << args[0]->GetType()->TypeName();
  CHECK(signal_type->dtype_ == DataType::INT32)
      << "pld.tensor.barrier signal must have INT32 element type (the barrier slot is an int counter), "
         "got dtype "
      << signal_type->dtype_.ToString();

  // Return signal's type — the rebind idiom lets users write
  // ``sig = pld.tensor.barrier(sig)``, matching allreduce.
  return args[0]->GetType();
}

}  // namespace

// ============================================================================
// pld.tensor.barrier — cross-rank barrier (notify-all + wait-all)
// ============================================================================

REGISTER_OP("pld.tensor.barrier")
    .set_description(
        "Cross-rank barrier: blocks until all ranks in the comm group have reached the barrier. "
        "`signal` is a window-bound INT32 matrix used as the cross-rank synchronisation (one slot "
        "per rank). Lowered to a notify-all / wait-all sequence by LowerCompositeOps; this op "
        "never survives past that pass.")
    .set_op_category("DistributedOp")
    .add_argument("signal", "Window-bound INT32 DistributedTensor used as cross-rank barrier (InOut)")
    .no_memory_spec()
    .f_deduce_type(DeduceTensorBarrierType);

}  // namespace ir
}  // namespace pypto
