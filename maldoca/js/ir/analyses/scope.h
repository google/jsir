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

#ifndef MALDOCA_JS_IR_ANALYSES_SCOPE_H_
#define MALDOCA_JS_IR_ANALYSES_SCOPE_H_

#include <cstdint>
#include <optional>

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Operation.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/babel/babel.pb.h"
#include "maldoca/js/babel/scope.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const JsSymbolId &s) {
  return os << absl::StrCat(s);
}

// Searches all scopes from the one `op` is in to the global scope for a symbol.
// Returns the uid of the scope where the symbol is defined.
std::optional<int64_t> FindSymbol(const BabelScopes &scopes,
                                  mlir::Operation *op, absl::string_view name);

// Turns a symbol name into a JsSymbolId, by searching all scopes from
// the one `op` is in to the global scope. If the symbol is not found, assume it
// has `scope_uid` 0.
JsSymbolId GetSymbolId(const BabelScopes &scopes, mlir::Operation *op,
                       absl::string_view name);

// Turns an op / attr into a JsSymbolId, by searching all scopes from
// the one op / attr is in to the global scope. If the symbol is not found,
// assume it has `scope_uid` 0.
JsSymbolId GetSymbolId(const BabelScopes &scopes, JsirIdentifierOp op);
JsSymbolId GetSymbolId(const BabelScopes &scopes, JsirIdentifierRefOp op);
JsSymbolId GetSymbolId(const BabelScopes &scopes, JsirIdentifierAttr attr);

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_ANALYSES_SCOPE_H_
