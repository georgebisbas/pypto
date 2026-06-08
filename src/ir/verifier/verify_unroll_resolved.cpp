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

#include "pypto/core/error.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {
namespace {

/// Walks a function body, reporting any ForStmt that still carries the
/// ``ForKind::Unroll`` marker. By design this marker must be gone after
/// ``UnrollLoops`` — the pass expands every unrollable loop into a SeqStmts
/// of N body copies. Any leftover indicates the pass silently skipped a loop
/// (e.g. non-const bounds or chunk_config it cannot handle).
class UnrollKindLeftoverChecker : public IRVisitor {
 public:
  UnrollKindLeftoverChecker(std::vector<Diagnostic>& diagnostics, const std::string& func_name)
      : diagnostics_(diagnostics), func_name_(func_name) {}

  void VisitStmt_(const ForStmtPtr& op) override {
    // Skip chunked unroll loops — they are valid at this pipeline position and
    // will be transformed by SplitChunkedLoops later.
    if (op->kind_ == ForKind::Unroll && !op->chunk_config_.has_value()) {
      diagnostics_.emplace_back(DiagnosticSeverity::Error, "UnrollResolved", 0,
                                "ForKind::Unroll survived past UnrollLoops in function '" + func_name_ +
                                    "'. This kind is a compile-time marker — UnrollLoops must expand it "
                                    "into a SeqStmts of N body copies. Check for non-const bounds or "
                                    "chunk_config that the pass cannot handle.",
                                op->span_);
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
  const std::string& func_name_;
};

class UnrollResolvedPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "UnrollResolved"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      UnrollKindLeftoverChecker checker(diagnostics, func->name_);
      checker.VisitStmt(func->body_);
    }
  }
};

}  // namespace

PropertyVerifierPtr CreateUnrollResolvedPropertyVerifier() {
  return std::make_shared<UnrollResolvedPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
