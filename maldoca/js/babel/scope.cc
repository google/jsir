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

#include "maldoca/js/babel/scope.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "maldoca/js/ast/ast.generated.h"

namespace maldoca {

std::optional<int64_t> FindBindingUid(const BabelScopes &scopes,
                                  int64_t use_scope_uid,
                                  absl::string_view name) {
  auto scope_it = scopes.scopes().find(use_scope_uid);
  if (scope_it == scopes.scopes().end()) {
    return std::nullopt;
  }
  const auto &scope = scope_it->second;
  auto binding_it = scope.binding_uids().find(name);
  if (binding_it != scope.binding_uids().end()) {
    return binding_it->second;
  }

  // If this scope has no parent, then this is the root, stop searching.
  if (!scope.has_parent_uid()) {
    return std::nullopt;
  }
  int64_t parent_scope_uid = scope.parent_uid();

  // Stop-gap: If parent_uid() defaults to 0, then we will be stuck in an
  // infinite loop, so also check whether this scope is the root (0).
  if (use_scope_uid == 0 && parent_scope_uid == use_scope_uid) {
    return std::nullopt;
  }

  return FindBindingUid(scopes, parent_scope_uid, name);
}

JsSymbolId GetSymbolId(const BabelScopes &scopes, int64_t use_scope_uid,
                       absl::string_view name) {
  std::optional<int64_t> binding_uid =
      FindBindingUid(scopes, use_scope_uid, name);
  return JsSymbolId{std::string(name), binding_uid};
}

std::optional<int64_t> FindDefScopeUid(const BabelScopes &scopes,
                                       const JsSymbolId &symbol) {
  if (!symbol.binding_uid().has_value()) {
    return std::nullopt;
  }
  for (const auto &[scope_uid, scope] : scopes.scopes()) {
    for (const auto &[name, binding_uid] : scope.binding_uids()) {
      if (binding_uid == *symbol.binding_uid()) {
        return scope_uid;
      }
    }
  }
  return std::nullopt;
}

}  // namespace maldoca
