// Copyright 2026 Google LLC
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

#include "maldoca/js/ir/analyses/scope_table/jsir_scope_table.h"

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Operation.h"

namespace maldoca {

const JsirSymbol &JsirScopeTable::Declare(llvm::StringRef name,
                                          JsirSymbolKind kind,
                                          mlir::Operation *owner) {
  auto [it, inserted] = symbols_.try_emplace(name, JsirSymbol{
                                                       .name = name.str(),
                                                       .kind = kind,
                                                       .scope = scope_,
                                                       .owner = owner,
                                                   });
  if (inserted) {
    order_.push_back(&it->second);
  }
  return it->second;
}

const JsirSymbol *JsirScopeTable::Lookup(llvm::StringRef name) const {
  auto it = symbols_.find(name);
  return it == symbols_.end() ? nullptr : &it->second;
}

void JsirScopeTable::SetScope(mlir::Operation *scope) {
  scope_ = scope;
  for (auto &entry : symbols_) {
    entry.second.scope = scope;
  }
}

void JsirScopeTable::ReplaceOwner(mlir::Operation *old_owner,
                                  mlir::Operation *new_owner) {
  // A null owner means "unknown", not a particular function.
  if (old_owner == nullptr) {
    return;
  }
  for (auto &entry : symbols_) {
    if (entry.second.owner == old_owner) {
      entry.second.owner = new_owner;
    }
  }
}

}  // namespace maldoca
