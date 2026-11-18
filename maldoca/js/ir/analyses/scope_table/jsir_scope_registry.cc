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

#include "maldoca/js/ir/analyses/scope_table/jsir_scope_registry.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/WalkResult.h"
#include "maldoca/js/ir/analyses/scope_table/jsir_scope_table.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {
namespace {

// If `op` is a `jsir.identifier` or `jsir.identifier_ref`, returns its name.
std::optional<llvm::StringRef> GetIdentifierName(mlir::Operation* op) {
  if (auto identifier = llvm::dyn_cast<JsirIdentifierOp>(op)) {
    return identifier.getName();
  }
  if (auto identifier_ref = llvm::dyn_cast<JsirIdentifierRefOp>(op)) {
    return identifier_ref.getName();
  }
  return std::nullopt;
}

}  // namespace

// -----------------------------------------------------------------------------
// Tables.
// -----------------------------------------------------------------------------

JsirScopeTable& JsirScopeRegistry::AddScope(mlir::Operation* op) {
  std::unique_ptr<JsirScopeTable>& table = scopes_[op];
  if (table == nullptr) {
    table = std::make_unique<JsirScopeTable>(op);
  }
  return *table;
}

JsirScopeTable* JsirScopeRegistry::GetScopeTable(mlir::Operation* op) const {
  auto it = scopes_.find(op);
  return it == scopes_.end() ? nullptr : it->second.get();
}

JsirScopeTable* JsirScopeRegistry::GetScopeTableFor(mlir::Operation* op) const {
  for (; op != nullptr; op = op->getParentOp()) {
    if (JsirScopeTable* table = GetScopeTable(op)) {
      return table;
    }
  }
  return nullptr;
}

void JsirScopeRegistry::TransferScope(mlir::Operation* old_op,
                                      mlir::Operation* new_op) {
  if (old_op == new_op) {
    return;
  }
  auto it = scopes_.find(old_op);
  if (it == scopes_.end()) {
    return;
  }
  std::unique_ptr<JsirScopeTable> table = std::move(it->second);
  scopes_.erase(it);
  table->SetScope(new_op);
  scopes_[new_op] = std::move(table);
  program_.ReplaceOwner(old_op, new_op);
}

void JsirScopeRegistry::RemoveScope(mlir::Operation* op) {
  if (scopes_.erase(op) == 0) {
    return;
  }
  // Do not leave dangling owners behind.
  program_.ReplaceOwner(op, nullptr);
}

void JsirScopeRegistry::MergeScopeInto(mlir::Operation* callee_op,
                                       mlir::Operation* caller_op) {
  JsirScopeTable* from = GetScopeTable(callee_op);
  JsirScopeTable* to = GetScopeTable(caller_op);
  if (from == nullptr || to == nullptr || from == to) {
    return;
  }
  for (const JsirSymbol* symbol : from->symbols()) {
    to->Declare(symbol->name, symbol->kind, symbol->owner);
  }
  // The callee's body now runs in the caller's activation, so the caller owns
  // the callee's environment variables. Re-own them before RemoveScope, which
  // would otherwise clear them.
  program_.ReplaceOwner(callee_op, caller_op);
  RemoveScope(callee_op);
}

// -----------------------------------------------------------------------------
// On-demand queries.
// -----------------------------------------------------------------------------

namespace {

// Returns true if `op` introduces a scope: a JsirFunctionOpInterface op, or an
// op with a table.
bool IsScopeOp(mlir::Operation* op, const JsirScopeRegistry& registry) {
  return llvm::isa<JsirFunctionOpInterface>(op) ||
         registry.GetScopeTable(op) != nullptr;
}

// Returns true if `function`'s parameters bind `name`. Every
// `jsir.identifier_ref` in the params region is a binding, including those
// nested in destructuring, rest and default-value patterns. Reads inside
// default-value expressions are `jsir.identifier`s and are not included.
bool IsParam(JsirFunctionOpInterface function, llvm::StringRef name) {
  return function.getParams()
      .walk([&](JsirIdentifierRefOp ref) {
        return ref.getName() == name ? mlir::WalkResult::interrupt()
                                     : mlir::WalkResult::advance();
      })
      .wasInterrupted();
}

// Returns the name in a function declaration's `id`, or "" if it has none.
llvm::StringRef GetFunctionName(JsirFunctionDeclarationOp declaration) {
  JsirIdentifierAttr id = declaration.getIdAttr();
  if (id == nullptr || id.getName() == nullptr) {
    return "";
  }
  return id.getName().getValue();
}

// Returns true if `scope` directly declares a nested scope named `name`: a
// `jsir.function_declaration` nested in `scope` but not in another scope op.
bool DeclaresScope(mlir::Operation* scope, llvm::StringRef name,
                   const JsirScopeRegistry& registry) {
  bool found = false;
  for (mlir::Region& region : scope->getRegions()) {
    region.walk<mlir::WalkOrder::PreOrder>(
        [&](mlir::Operation* op) -> mlir::WalkResult {
          if (auto declaration =
                  llvm::dyn_cast<JsirFunctionDeclarationOp>(op)) {
            if (GetFunctionName(declaration) == name) {
              found = true;
              return mlir::WalkResult::interrupt();
            }
            return mlir::WalkResult::skip();
          }
          if (IsScopeOp(op, registry)) {
            return mlir::WalkResult::skip();
          }
          return mlir::WalkResult::advance();
        });
    if (found) {
      return true;
    }
  }
  return false;
}

}  // namespace

std::optional<JsirSymbol> JsirScopeRegistry::Resolve(
    mlir::Operation* identifier) const {
  std::optional<llvm::StringRef> name = GetIdentifierName(identifier);
  if (!name.has_value()) {
    return std::nullopt;
  }
  return Resolve(identifier, *name);
}

std::optional<JsirSymbol> JsirScopeRegistry::Resolve(
    mlir::Operation* site, llvm::StringRef name) const {
  // Only the table of the nearest op with a table is consulted.
  bool found_table = false;

  for (mlir::Operation* op = site->getParentOp(); op != nullptr;
       op = op->getParentOp()) {
    const JsirScopeTable* table = found_table ? nullptr : GetScopeTable(op);
    if (table != nullptr) {
      found_table = true;
    }
    bool is_program = llvm::isa<JsirProgramOp>(op);
    auto function = llvm::dyn_cast<JsirFunctionOpInterface>(op);
    if (table == nullptr && !is_program && !function) {
      continue;
    }

    // The scope `op` stands for: null for the program.
    mlir::Operation* declaring_op = is_program ? nullptr : op;

    // 1. / 3. Parameters, then functions declared directly in `op`.
    if (function && IsParam(function, name)) {
      return JsirSymbol{.name = name.str(),
                        .kind = JsirSymbolKind::kParam,
                        .scope = declaring_op};
    }
    if (DeclaresScope(op, name, *this)) {
      return JsirSymbol{.name = name.str(),
                        .kind = JsirSymbolKind::kFunction,
                        .scope = declaring_op};
    }

    // 2. The nearest table.
    if (table != nullptr) {
      if (const JsirSymbol* symbol = table->Lookup(name)) {
        return *symbol;
      }
    }
  }

  // 3. The program table.
  if (const JsirSymbol* symbol = program_.Lookup(name)) {
    return *symbol;
  }

  // 4. A global.
  return std::nullopt;
}

std::vector<mlir::Operation*> JsirScopeRegistry::GetUses(
    const JsirSymbol& symbol) const {
  // Only the IR of the declaring scope (all of it for the program) can refer
  // to `symbol`.
  mlir::Operation* root = symbol.scope != nullptr ? symbol.scope : anchor_;

  // For parameters, the region whose identifiers declare them.
  mlir::Region* params = nullptr;
  if (symbol.kind == JsirSymbolKind::kParam) {
    if (auto function =
            llvm::dyn_cast_or_null<JsirFunctionOpInterface>(symbol.scope)) {
      params = &function.getParams();
    }
  }

  std::vector<mlir::Operation*> uses;
  root->walk([&](mlir::Operation* op) {
    std::optional<llvm::StringRef> name = GetIdentifierName(op);
    if (!name.has_value() || *name != symbol.name) {
      return;
    }
    if (params != nullptr && llvm::isa<JsirIdentifierRefOp>(op) &&
        params->isAncestor(op->getParentRegion())) {
      return;
    }
    std::optional<JsirSymbol> resolved = Resolve(op, *name);
    if (resolved.has_value() && *resolved == symbol) {
      uses.push_back(op);
    }
  });
  return uses;
}

bool JsirScopeRegistry::IsCaptured(const JsirSymbol& symbol) const {
  if (symbol.kind == JsirSymbolKind::kFunction) {
    return false;
  }

  // The function op the symbol belongs to; null for the program.
  mlir::Operation* home =
      symbol.kind == JsirSymbolKind::kEnv ? symbol.owner : symbol.scope;

  for (mlir::Operation* use : GetUses(symbol)) {
    // The innermost scope op (see IsScopeOp) enclosing the use: the function
    // the use runs in.
    mlir::Operation* use_scope = use->getParentOp();
    while (use_scope != nullptr && !IsScopeOp(use_scope, *this)) {
      use_scope = use_scope->getParentOp();
    }
    if (use_scope != home) {
      return true;
    }
  }
  return false;
}

}  // namespace maldoca
