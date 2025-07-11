// Copyright 2024 Google LLC
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

#include "maldoca/js/ir/analyses/scope.h"

#include <cstdint>
#include <optional>
#include <string>

#include "llvm/Support/Casting.h"
#include "mlir/IR/Operation.h"
#include "absl/strings/string_view.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/babel/babel.pb.h"
#include "maldoca/js/babel/scope.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {

std::optional<int64_t> FindSymbol(const BabelScopes &scopes,
                                  mlir::Operation *op, absl::string_view name) {
  auto trivia = llvm::dyn_cast<JsirTriviaAttr>(op->getLoc());
  if (trivia == nullptr) {
    return std::nullopt;
  }

  std::optional<int64_t> use_scope_uid = trivia.getLoc().getScopeUid();
  if (!use_scope_uid.has_value()) {
    return std::nullopt;
  }

  return FindSymbol(scopes, *use_scope_uid, name);
}

JsSymbolId GetSymbolId(const BabelScopes &scopes, mlir::Operation *op,
                       absl::string_view name) {
  return JsSymbolId{std::string(name), FindSymbol(scopes, op, name)};
}

JsSymbolId GetSymbolId(const BabelScopes &scopes, JsirIdentifierOp op) {
  return GetSymbolId(scopes, op, op.getName());
}

JsSymbolId GetSymbolId(const BabelScopes &scopes, JsirIdentifierRefOp op) {
  return GetSymbolId(scopes, op, op.getName());
}

JsSymbolId GetSymbolId(const BabelScopes &scopes, JsirIdentifierAttr attr) {
  absl::string_view name = attr.getName().strref();

  std::optional<int64_t> use_scope_uid = [&]() -> std::optional<int64_t> {
    JsirTriviaAttr trivia = attr.getLoc();
    if (trivia == nullptr) {
      return std::nullopt;
    }
    return trivia.getLoc().getScopeUid();
  }();

  if (!use_scope_uid.has_value()) {
    return JsSymbolId{std::string(name), std::nullopt};
  }
  return GetSymbolId(scopes, *use_scope_uid, name);
}

}  // namespace maldoca
