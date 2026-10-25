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

#include "pypto/ir/core.h"

#include <cstdint>
#include <deque>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "pypto/core/logging.h"
#include "pypto/ir/span.h"

namespace pypto {
namespace ir {
namespace {

const std::string& EmptyFilename() {
  static const std::string kEmpty;
  return kEmpty;
}

/// Process-wide, append-only filename interner.
///
/// References to the stored strings are stable: `std::deque` does not
/// invalidate references on `emplace_back`, and the map keys are views into
/// that deque. Reads and writes share a mutex; a production version would use
/// a per-compilation or sharded interner to keep the hot path lock-free.
class FilenameInterner {
 public:
  static FilenameInterner& Instance() {
    // Intentionally leaked: spans can live in static objects, and destroying
    // the interner at exit would leave their filename lookups dangling.
    static auto* instance = new FilenameInterner();
    return *instance;
  }

  std::uint32_t Intern(std::string_view name) {
    if (name.empty()) {
      return 0;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (auto it = index_.find(name); it != index_.end()) {
      return it->second;
    }
    const auto id = static_cast<std::uint32_t>(names_.size() + 1);
    names_.emplace_back(name.data(), name.size());
    index_.emplace(names_.back(), id);
    return id;
  }

  const std::string& Lookup(std::uint32_t id) const {
    if (id == 0) {
      return EmptyFilename();
    }
    std::lock_guard<std::mutex> lock(mutex_);
    // No public constructor takes a raw id, so an out-of-range handle can only
    // come from corrupted state inside PyPTO: an internal bug, not user input.
    INTERNAL_CHECK(id <= names_.size()) << "Internal error: invalid interned filename id " << id;
    return names_[id - 1];
  }

 private:
  mutable std::mutex mutex_;
  std::deque<std::string> names_;
  std::unordered_map<std::string_view, std::uint32_t> index_;
};

}  // namespace

Filename::Filename(std::string_view name) : id_(FilenameInterner::Instance().Intern(name)) {}

const std::string& Filename::str() const { return FilenameInterner::Instance().Lookup(id_); }

std::string_view Filename::view() const { return str(); }

std::ostream& operator<<(std::ostream& os, Filename filename) { return os << filename.str(); }

Span::Span(std::string filename, int begin_line, int begin_column, int end_line, int end_column)
    : Span(Filename(filename), begin_line, begin_column, end_line, end_column) {}

Span::Span(Filename filename, int begin_line, int begin_column, int end_line, int end_column)
    : filename_(filename),
      begin_line_(begin_line),
      begin_column_(begin_column),
      end_line_(end_line),
      end_column_(end_column) {}

std::string Span::to_string() const {
  std::ostringstream oss;
  oss << filename_ << ":" << begin_line_ << ":" << begin_column_;
  return oss.str();
}

bool Span::is_valid() const {
  if (begin_line_ <= 0 || (begin_column_ <= 0 && begin_column_ != -1)) {
    return false;
  }
  if (end_line_ == -1 || end_column_ == -1) {
    return true;
  }
  if (end_line_ <= 0 || (end_column_ <= 0 && end_column_ != -1)) {
    return false;
  }
  if (begin_column_ == -1 || end_column_ == -1) {
    return end_line_ >= begin_line_;
  }
  return end_line_ >= begin_line_ && (end_line_ > begin_line_ || end_column_ >= begin_column_);
}

Span Span::unknown() { return Span(Filename(), -1, -1, -1, -1); }

}  // namespace ir
}  // namespace pypto
