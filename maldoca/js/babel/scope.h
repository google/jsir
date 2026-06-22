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

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/babel/babel.pb.h"

namespace maldoca {

template <typename H>
H AbslHashValue(H h, const JsSymbolId& s) {
  return H::combine(std::move(h), s.name(), s.def_scope_uid(),
                    s.binding_uid().value_or(-1));
}

inline bool operator==(const JsSymbolId& lhs, const JsSymbolId& rhs) {
  return lhs.name() == rhs.name() &&
         lhs.def_scope_uid() == rhs.def_scope_uid() &&
         lhs.binding_uid().value_or(-1) == rhs.binding_uid().value_or(-1);
}

inline bool operator<(const JsSymbolId& lhs, const JsSymbolId& rhs) {
  return std::make_tuple(lhs.binding_uid().value_or(-1), lhs.def_scope_uid(),
                         lhs.name()) <
         std::make_tuple(rhs.binding_uid().value_or(-1), rhs.def_scope_uid(),
                         rhs.name());
}

template <typename Sink>
void AbslStringify(Sink& sink, const JsSymbolId& s) {
  std::string id = s.binding_uid().has_value()
                       ? absl::StrCat("b", *s.binding_uid())
                       : (s.def_scope_uid().has_value()
                              ? absl::StrCat("s", *s.def_scope_uid())
                              : "undeclared");
  absl::Format(&sink, "%s#%s", s.name(), id);
}

inline std::ostream& operator<<(std::ostream& os, const JsSymbolId& s) {
  return os << absl::StrCat(s);
}

// Searches all scopes from `scope_uid` to the global scope for a symbol.
// Returns the uid of the scope where the symbol is defined.
std::optional<int64_t> FindSymbol(const BabelScopes& scopes, int64_t scope_uid,
                                  absl::string_view name,
                                  bool is_var_declaration = false);

// Turns a symbol name into a JsSymbolId, by searching all scopes from
// `scope_uid` to the global scope. If the symbol is not found, assume it has
// `scope_uid` 0.
JsSymbolId GetSymbolId(const BabelScopes& scopes, int64_t scope_uid,
                       absl::string_view name, bool is_var_declaration = false);

}  // namespace maldoca

#endif  // MALDOCA_JS_BABEL_SCOPE_H_
