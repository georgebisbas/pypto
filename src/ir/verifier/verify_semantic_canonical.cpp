/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/simplify.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

class SemanticCanonicalPropertyVerifier final : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "SemanticCanonical"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;

    for (const auto& entry : program->functions_) {
      const auto& func = entry.second;
      if (!func) continue;

      try {
        // The first prototype intentionally defines SemanticCanonical as a
        // fixed point of the existing semantic transform. It proves ownership
        // without introducing a second canonicalization implementation.
        auto canonical = transform_utils::SimplifyFunction(func, /*constants_only=*/false);
        if (!structural_equal(func, canonical)) {
          diagnostics.emplace_back(
              DiagnosticSeverity::Error, "SemanticCanonical", /*error_code=*/1,
              "function '" + func->name_ + "' is not at Simplify's semantic fixed point; "
              "a consumer must not assume canonical scalar/shape expressions",
              func->span_);
        }
      } catch (const std::exception& error) {
        diagnostics.emplace_back(DiagnosticSeverity::Error, "SemanticCanonical", /*error_code=*/2,
                                 "function '" + func->name_ + "' could not be checked for semantic "
                                 "canonicality: " + error.what(),
                                 func->span_);
      }
    }
  }
};

}  // namespace

PropertyVerifierPtr CreateSemanticCanonicalPropertyVerifier() {
  return std::make_shared<SemanticCanonicalPropertyVerifier>();
}

}  // namespace ir
}  // namespace pypto
