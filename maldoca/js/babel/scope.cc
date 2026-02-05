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

#include "absl/strings/string_view.h"
#include "maldoca/js/ast/ast.generated.h"

namespace maldoca {

std::optional<int64_t> FindSymbol(const BabelScopes& scopes,
                                  int64_t use_scope_uid, absl::string_view name,
                                  bool is_var_declaration) {
  auto scope_it = scopes.scopes().find(use_scope_uid);
  if (scope_it == scopes.scopes().end()) {
    return std::nullopt;
  }
  const auto& scope = scope_it->second;

  bool found_matching_binding = false;

  auto binding_it = scope.binding_uids().find(std::string(name));
  if (binding_it != scope.binding_uids().end()) {
    found_matching_binding = true;
    if (is_var_declaration) {
      int32_t binding_uid = binding_it->second;
      auto b_it = scopes.bindings().find(binding_uid);
      if (b_it != scopes.bindings().end()) {
        const auto& binding = b_it->second;
        if (binding.kind() == BabelBinding::KIND_LET ||
            binding.kind() == BabelBinding::KIND_CONST ||
            binding.kind() == BabelBinding::KIND_LOCAL) {
          found_matching_binding = false;  // skip
        }
      }
    }
  }

  if (!found_matching_binding) {
    auto old_binding_it = scope.bindings().find(std::string(name));
    if (old_binding_it != scope.bindings().end()) {
      found_matching_binding = true;
      if (is_var_declaration) {
        const auto& binding = old_binding_it->second;
        if (binding.kind() == BabelBinding::KIND_LET ||
            binding.kind() == BabelBinding::KIND_CONST ||
            binding.kind() == BabelBinding::KIND_LOCAL) {
          found_matching_binding = false;  // skip
        }
      }
    }
  }

  if (found_matching_binding) {
    return use_scope_uid;
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

  return FindSymbol(scopes, parent_scope_uid, name, is_var_declaration);
}

JsSymbolId GetSymbolId(const BabelScopes& scopes, int64_t use_scope_uid,
                       absl::string_view name, bool is_var_declaration) {
  auto def_scope_uid =
      FindSymbol(scopes, use_scope_uid, name, is_var_declaration);
  std::optional<int64_t> binding_uid = std::nullopt;
  if (def_scope_uid.has_value()) {
    auto scope_it = scopes.scopes().find(*def_scope_uid);
    if (scope_it != scopes.scopes().end()) {
      auto binding_it = scope_it->second.binding_uids().find(std::string(name));
      if (binding_it != scope_it->second.binding_uids().end()) {
        binding_uid = binding_it->second;
      } else {
        auto old_binding_it =
            scope_it->second.bindings().find(std::string(name));
        if (old_binding_it != scope_it->second.bindings().end() &&
            old_binding_it->second.has_uid()) {
          binding_uid = old_binding_it->second.uid();
        }
      }
    }
  }
  return JsSymbolId{std::string(name), def_scope_uid, binding_uid};
}

}  // namespace maldoca
