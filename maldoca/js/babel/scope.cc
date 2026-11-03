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

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/babel/babel.pb.h"

namespace maldoca {
namespace {

// Returns whether the binding with `binding_uid` is block-scoped, i.e. declared
// with `let` or `const`. Babel also registers catch clause parameters and class
// declarations as `let`.
bool IsBlockScopedBinding(const BabelScopes &scopes, int64_t binding_uid) {
  auto it = scopes.bindings().find(binding_uid);
  if (it == scopes.bindings().end()) {
    return false;
  }
  return it->second.kind() == BabelBinding::KIND_LET ||
         it->second.kind() == BabelBinding::KIND_CONST;
}

// Returns whether the lookup for a declaration of `declaration_kind` skips the
// binding with `binding_uid` and continues in the parent scope. See
// `FindSymbolBindingUid`.
bool ShouldSkipBinding(const BabelScopes &scopes, int64_t binding_uid,
                       BabelBinding::Kind declaration_kind) {
  switch (declaration_kind) {
    case BabelBinding::KIND_VAR:
      // Babel registers `var` bindings in the enclosing function or program
      // scope, so skip block-scoped bindings (e.g. a catch clause parameter)
      // between the scope where the `var` is written and that scope.
      return IsBlockScopedBinding(scopes, binding_uid);
    case BabelBinding::KIND_UNKNOWN:
    case BabelBinding::KIND_LET:
    case BabelBinding::KIND_CONST:
    case BabelBinding::KIND_MODULE:
    case BabelBinding::KIND_HOISTED:
    case BabelBinding::KIND_PARAM:
    case BabelBinding::KIND_LOCAL:
      // Babel registers these bindings in the scope where they are written, so
      // the nearest binding is the one.
      return false;
  }
  ABSL_LOG(FATAL) << "Unknown BabelBinding::Kind: "
             << static_cast<int>(declaration_kind);
}

}  // namespace

std::optional<int64_t> FindSymbolBindingUid(
    const BabelScopes &scopes, int64_t use_scope_uid, absl::string_view name,
    std::optional<BabelBinding::Kind> declaration_kind) {
  auto scope_it = scopes.scopes().find(use_scope_uid);
  if (scope_it == scopes.scopes().end()) {
    return std::nullopt;
  }
  const auto &scope = scope_it->second;
  auto binding_it = scope.binding_uids().find(name);
  if (binding_it != scope.binding_uids().end()) {
    int64_t binding_uid = binding_it->second;
    // References and assignments resolve to the nearest binding, whereas a
    // declaration may skip it and keep searching up the scope tree, e.g. a
    // `var` skips a catch clause parameter to reach the scope that the `var`
    // is hoisted to.
    if (!declaration_kind.has_value() ||
        !ShouldSkipBinding(scopes, binding_uid, *declaration_kind)) {
      return binding_uid;
    }
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

  return FindSymbolBindingUid(scopes, parent_scope_uid, name, declaration_kind);
}

JsSymbolId GetSymbolId(const BabelScopes &scopes, int64_t use_scope_uid,
                       absl::string_view name,
                       std::optional<BabelBinding::Kind> declaration_kind) {
  std::optional<int64_t> binding_uid =
      FindSymbolBindingUid(scopes, use_scope_uid, name, declaration_kind);
  return JsSymbolId{std::string(name), binding_uid};
}

absl::InlinedVector<int64_t, 4> FindDefScopeUids(const BabelScopes &scopes,
                                                const JsSymbolId &symbol) {
  if (!symbol.binding_uid().has_value()) {
    return {};
  }
  absl::InlinedVector<int64_t, 4> def_scope_uids;
  for (const auto &[scope_uid, scope] : scopes.scopes()) {
    for (const auto &[name, binding_uid] : scope.binding_uids()) {
      if (binding_uid == *symbol.binding_uid()) {
        def_scope_uids.push_back(scope_uid);
        break;
      }
    }
  }
  absl::c_sort(def_scope_uids);
  return def_scope_uids;
}

}  // namespace maldoca
