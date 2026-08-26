// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MALDOCA_JS_BABEL_SCOPE_H_
#define MALDOCA_JS_BABEL_SCOPE_H_

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/babel/babel.pb.h"

namespace maldoca {

template <typename H>
H AbslHashValue(H h, const JsSymbolId& s) {
  return H::combine(std::move(h), s.name(), s.binding_uid());
}

inline bool operator==(const JsSymbolId& lhs, const JsSymbolId& rhs) {
  return lhs.name() == rhs.name() && lhs.binding_uid() == rhs.binding_uid();
}

inline bool operator<(const JsSymbolId& lhs, const JsSymbolId& rhs) {
  return std::forward_as_tuple(lhs.binding_uid(), lhs.name()) <
         std::forward_as_tuple(rhs.binding_uid(), rhs.name());
}

template <typename Sink>
void AbslStringify(Sink& sink, const JsSymbolId& s) {
  std::string id = s.binding_uid().has_value()
                       ? absl::StrCat(*s.binding_uid())
                       : "undeclared";
  absl::Format(&sink, "%s#%s", s.name(), id);
}

inline std::ostream& operator<<(std::ostream& os, const JsSymbolId& s) {
  return os << absl::StrCat(s);
}

// Searches all scopes from `scope_uid` to the global scope for a symbol.
// Returns the binding_uid of the symbol if found, otherwise std::nullopt.
std::optional<int64_t> FindSymbolBindingUid(const BabelScopes &scopes,
                                            int64_t scope_uid,
                                            absl::string_view name);

// Turns a symbol name into a JsSymbolId, by searching all scopes from
// `scope_uid` to the global scope. If the symbol is not found, assume it has
// `scope_uid` 0.
JsSymbolId GetSymbolId(const BabelScopes& scopes, int64_t scope_uid,
                       absl::string_view name);

// Returns the definition scope uids of a symbol. If `symbol.binding_uid()` is
// set, searches `scopes` for the scopes containing that binding uid.
absl::InlinedVector<int64_t, 4> FindDefScopeUids(const BabelScopes& scopes,
                                                const JsSymbolId& symbol);

}  // namespace maldoca

#endif  // MALDOCA_JS_BABEL_SCOPE_H_
