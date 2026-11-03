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
//
// `declaration_kind` is the kind of the binding that Babel registers for the
// declaration being resolved, e.g. `KIND_VAR` for a `var` declarator or
// `KIND_PARAM` for a function parameter. It is std::nullopt for references and
// assignments, which do not declare the symbol.
//
// For references, assignments, and declarations of any kind other than
// `KIND_VAR`, returns the nearest binding of any kind, since Babel registers
// the bindings of those declarations in the scope where they are written.
//
// For `KIND_VAR`, skips block-scoped bindings (`KIND_LET` and `KIND_CONST`,
// which include catch clause parameters and class declarations) and returns
// the nearest binding of any other kind. This is how a `var` declaration finds
// its binding: the lookup starts in the scope where the `var` is written, but
// Babel registers the binding in the enclosing function or program scope. For
// example, in `try { ... } catch (e) { var e = 1; }`, the lookup starts in the
// catch clause scope, whose `e` is the catch parameter (`KIND_LET`), and has to
// continue to the enclosing scope to find the `var` binding.
//
// That binding is not necessarily `KIND_VAR`: if the name is already bound in
// the scope that the `var` is hoisted to, Babel reuses that binding and keeps
// its kind, e.g. `KIND_HOISTED` in `function a() {} var a;`, `KIND_PARAM` in
// `function f(a) { var a; }`, or `KIND_LOCAL` in `(function a() { var a; })`.
// Skipping only block-scoped bindings is enough because, in valid JavaScript,
// a catch clause parameter is the only binding of the same name that can be in
// a scope between a `var` and the scope that it is hoisted to; any other would
// be a redeclaration error.
std::optional<int64_t> FindSymbolBindingUid(
    const BabelScopes &scopes, int64_t scope_uid, absl::string_view name,
    std::optional<BabelBinding::Kind> declaration_kind = std::nullopt);

// Turns a symbol name into a JsSymbolId, by searching all scopes from
// `scope_uid` to the global scope. If the symbol is not found, assume it has
// `scope_uid` 0.
// See `FindSymbolBindingUid` for how `declaration_kind` affects the search.
JsSymbolId GetSymbolId(
    const BabelScopes& scopes, int64_t scope_uid, absl::string_view name,
    std::optional<BabelBinding::Kind> declaration_kind = std::nullopt);

// Returns the definition scope uids of a symbol. If `symbol.binding_uid()` is
// set, searches `scopes` for the scopes containing that binding uid.
absl::InlinedVector<int64_t, 4> FindDefScopeUids(const BabelScopes& scopes,
                                                const JsSymbolId& symbol);

}  // namespace maldoca

#endif  // MALDOCA_JS_BABEL_SCOPE_H_
