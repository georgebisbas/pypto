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

#include <cassert>
#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "pypto/ir/span.h"

namespace {

using namespace pypto::ir;  // NOLINT(build/namespaces)

// A `Filename` is a handle into a process-wide interner. These checks pin the
// contract that makes packing `Span` safe: equal names must produce the same
// handle, the empty name must be id 0 (so `Span::unknown()` allocates nothing),
// and lookups must stay name-based. A regression here would silently make two
// spans with the same filename compare unequal, or make an unknown span look
// like a real source location.
void CheckFilenameInterning() {
  Filename alpha("alpha.py");
  Filename alpha_again("alpha.py");
  Filename beta("beta.py");

  assert(!alpha.empty());
  assert(alpha.id() != 0);
  assert(alpha == alpha_again);
  assert(alpha.id() == alpha_again.id());
  assert(alpha != beta);
  assert(alpha.view() == "alpha.py");
  assert(alpha.str() == "alpha.py");

  Filename empty_default;
  Filename empty_interned("");
  assert(empty_default.empty());
  assert(empty_default.id() == 0);
  assert(empty_interned.empty());
  assert(empty_interned.id() == 0);
  assert(empty_default == empty_interned);

  std::ostringstream os;
  os << alpha;
  assert(os.str() == "alpha.py");
}

// Interning more names must never invalidate handles already handed out: the
// backing store is a deque (references are stable on growth) and the index keys
// are views into it. A `std::vector<std::string>` store would reallocate and
// leave every outstanding `string_view` dangling while the handle still looked
// valid — the exact failure this test is here to catch.
void CheckHandleStability() {
  Filename first("stable.py");
  const std::string_view first_view = first.view();
  const std::string* first_str = &first.str();

  std::vector<Filename> later;
  later.reserve(4096);
  for (int i = 0; i < 4096; ++i) {
    later.emplace_back("generated_" + std::to_string(i) + ".py");
  }

  assert(first.view() == first_view);
  assert(&first.str() == first_str);  // the same stored string, not a copy
  assert(first.str() == "stable.py");
}

void CheckSpanPacking() {
  static_assert(sizeof(Filename) == sizeof(std::uint32_t), "Filename must stay a 32-bit handle");
  static_assert(sizeof(Span) <= 24, "Span must stay packed: interned filename plus four ints");

  Span span("file.py", 3, 4, 3, 9);
  assert(span.filename_.str() == "file.py");
  assert(span.to_string() == "file.py:3:4");
  assert(span.is_valid());

  // Combining spans reuses the handle instead of interning the name again.
  Span combined(span.filename_, span.begin_line_, span.begin_column_, span.end_line_, span.end_column_);
  assert(combined.filename_ == span.filename_);
  assert(combined.filename_.id() == span.filename_.id());
  assert(combined.to_string() == span.to_string());

  Span unknown = Span::unknown();
  assert(unknown.filename_.empty());
  assert(unknown.filename_.id() == 0);
  assert(!unknown.is_valid());
  assert(unknown.to_string() == ":-1:-1");
}

}  // namespace

int main() {
  CheckFilenameInterning();
  CheckHandleStability();
  CheckSpanPacking();
  return 0;
}
