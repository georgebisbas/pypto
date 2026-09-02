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
 * @file allreduce_core_num.h
 * @brief Phase-A auto `core_num` policy for the HOST mesh allreduce.
 *
 * `core_num` (the SPMD AIV grid width of the mesh allreduce builtin, #2160) is a
 * real on-device speedup — single-AIV does not saturate the HCCS/MTE path — but
 * it only pays off once the per-rank payload is large enough that the extra AIV
 * lanes stop competing with the fixed per-peer barrier/lane overhead. Below that
 * crossover multi-AIV *loses* (contention), so the width has to be selected from
 * the payload, not always-on.
 *
 * This is the compiler-side selector: when a HOST `pld.tensor.allreduce` carries
 * no explicit `core_num`, `LowerHostTensorCollectives` resolves one here from
 * (per-rank payload bytes, world_size); `SynthesizeAllReduceSignals` uses the
 * P-independent form to size synthesized signal lane counts (it runs before the
 * comm-domain device set is known).
 *
 * Table source: pypto-profiling `corenum-message-size-crossover-2026-08-31.md`
 * (monotone-stable crossover: first payload where cn8/cn1 >= 1.0 and holds for
 * every larger payload):
 *
 *   - P=2: cn8 stable from 256 KiB (1.71x @ 256 KiB; holds through 4 MiB).
 *   - P=4: cn8 stable from 128 KiB (1.15x @ 128 KiB; holds: 2.58x @ 256 KiB ...).
 *   - P=8: single measured row 1.5x @ 64 KiB (2026-08-28 sweep) -> crossover 64 KiB.
 *   - Higher P pulls the crossover down; P=5..7 / P=9..16 hold their range's most
 *     conservative bound. Unknown/dynamic world_size assumes the P=2 column (the
 *     highest measured crossover — safe for every real P).
 *   - cn16 at P=4 from 16 KiB is treated as noise (~250 us runs, ~2x spread).
 *
 * Band widths: 1 below crossover, 8 from crossover, 16 at >= 2 MiB (P=4 cn16
 * 3.47x @ 2 MiB / 5.84x @ 4 MiB vs cn8 3.18x/4.66x; cn16 is never < 1.0x above
 * crossover, so no band regresses vs single-AIV). Every value is <= 16 and must
 * still be capped at the backend's `CoreType::VECTOR` count (48 on 910B, 36 on
 * 950) — never CUBE — by the caller (`CheckAllReduceCoreCapacity`).
 *
 * Caveat: the crossover numbers predate PR #2591's barrier/dcci slimming; the
 * thresholds are FP32 measurements and must be re-baselined on current main
 * before final tuning.
 */

#ifndef PYPTO_IR_TRANSFORMS_UTILS_ALLREDUCE_CORE_NUM_H_
#define PYPTO_IR_TRANSFORMS_UTILS_ALLREDUCE_CORE_NUM_H_

#include <cstdint>
#include <cstdlib>
#include <string>

#include "pypto/ir/expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace allreduce_core_num {

/// Per-rank payload at which multi-AIV's stable gain begins, per P band
/// (bytes). See the file comment for the measured rows.
inline constexpr int64_t kCrossoverBytesP2 = 256 * 1024;  // 256 KiB — measured P=2 cn8 stable (1.71x)
inline constexpr int64_t kCrossoverBytesP4 = 128 * 1024;  // 128 KiB — measured P=4 cn8 stable (1.15x)
inline constexpr int64_t kCrossoverBytesP8 = 64 * 1024;   // 64 KiB  — P=8/64K 1.5x (2026-08-28 sweep)
/// Per-rank payload at which cn16 replaces cn8 (P=4 cn16 3.47x @ 2 MiB, 5.84x @ 4 MiB).
inline constexpr int64_t kLargePayloadBytes = 2 * 1024 * 1024;  // 2 MiB

/// Band widths selected above the crossover.
inline constexpr int64_t kMidCoreNum = 8;     // cn8: measured sufficient at every payload >= crossover
inline constexpr int64_t kLargeCoreNum = 16;  // cn16: top end only (>= 2 MiB)

/// Environment override: `PYPTO_ALLREDUCE_CORE_NUM` forces the width for calls
/// that do not carry an explicit `core_num=` (explicit DSL wins over the env).
inline constexpr const char* kCoreNumEnvVar = "PYPTO_ALLREDUCE_CORE_NUM";

/// Value of `PYPTO_ALLREDUCE_CORE_NUM`, or 0 when unset / empty / non-numeric /
/// not positive. Read at each use (lowering is compile-time and infrequent, so
/// no per-process cache — keeping it live lets unit tests drive it with
/// `monkeypatch`).
inline int64_t EnvCoreNumOverride() {
  const char* value = std::getenv(kCoreNumEnvVar);
  if (value == nullptr) return 0;
  std::string text(value);
  if (text.empty()) return 0;
  try {
    const std::size_t pos = text.find_first_not_of(" \t");
    if (pos != std::string::npos && pos != 0) text = text.substr(pos);
    const int64_t parsed = std::stoll(text);
    return parsed > 0 ? parsed : 0;
  } catch (...) {
    return 0;
  }
}

/// Compile-time per-rank payload bytes of a HOST allreduce `src`/`target`
/// (product of constant extents x element bytes), or -1 when any extent is not
/// a compile-time constant (the compiler cannot select a payload band then).
inline int64_t StaticPayloadBytes(const ExprPtr& src) {
  if (!src) return -1;
  auto src_type = As<DistributedTensorType>(src->GetType());
  if (!src_type) return -1;
  int64_t numel = 1;
  for (const auto& dim : src_type->shape_) {
    auto extent = As<ConstInt>(dim);
    if (!extent) return -1;
    numel *= extent->value_;
  }
  const int64_t element_bytes = static_cast<int64_t>(src_type->dtype_.GetByte());
  if (element_bytes <= 0) return -1;
  return numel * element_bytes;
}

/// `(payload, P) -> core_num` policy. `payload_bytes` < 0 (unknown extent) keeps
/// single-AIV. `world_size_known == false` (fully-dynamic all-device domain)
/// falls back to the P=2 column — the highest measured crossover, safe for every
/// P. Result is never capped here: the caller applies the backend
/// `CoreType::VECTOR` bound and, for purely-auto calls, the signal lane cap.
inline int64_t PolicyCoreNum(int64_t payload_bytes, bool world_size_known, int64_t world_size) {
  int64_t crossover = kCrossoverBytesP2;  // P <= 3 and unknown P
  if (world_size_known) {
    if (world_size >= 8) {
      crossover = kCrossoverBytesP8;
    } else if (world_size >= 4) {
      crossover = kCrossoverBytesP4;
    }
  }
  if (payload_bytes < 0 || payload_bytes < crossover) return 1;
  if (payload_bytes < kLargePayloadBytes) return kMidCoreNum;
  return kLargeCoreNum;
}

}  // namespace allreduce_core_num
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_ALLREDUCE_CORE_NUM_H_
