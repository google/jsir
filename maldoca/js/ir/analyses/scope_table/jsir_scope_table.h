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

// JsirScopeTable: the declarations of ONE scope (a function, or the program).
//
// A table only stores what its producer registers: name -> symbol. It knows
// the op of its scope (null for the program), but does not store uses and
// never looks at the IR. JsirScopeRegistry (jsir_scope_registry.h) owns the
// tables of a program and answers name-resolution queries with them.

#ifndef MALDOCA_JS_IR_ANALYSES_SCOPE_TABLE_JSIR_SCOPE_TABLE_H_
#define MALDOCA_JS_IR_ANALYSES_SCOPE_TABLE_JSIR_SCOPE_TABLE_H_

#include <cstddef>
#include <string>
#include <vector>

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Operation.h"

namespace maldoca {

// What kind of declaration introduces a symbol, in JavaScript terms. Where the
// symbol is visible is given by JsirSymbol::scope.
enum class JsirSymbolKind {
  // A function-scoped variable, with the semantics of `var` (e.g. a temporary
  // or a register created by the Hermes decompiler).
  kVar,
  // A `var` of the function `owner` that lives in a closure environment, so
  // closures created by `owner` can read and write it. Declared at program
  // level, because producers such as the Hermes decompiler flatten closures
  // into program-level functions that reach their parents' environments
  // directly. `owner` is null if the owning function is unknown.
  kClosureVar,
  // A function parameter. Never stored in a table: read from the function op's
  // params region (JsirFunctionOpInterface::getParams()).
  kParam,
  // A function name. Never stored in a table: read from the `id` of the
  // `jsir.function_declaration`.
  kFunction,
};

// A declared name: both what a table stores, and what a name resolves to
// (JsirScopeRegistry::Resolve).
struct JsirSymbol {
  std::string name;
  JsirSymbolKind kind;
  // The op of the declaring scope; nullptr for the program:
  // - kVar / kClosureVar: the scope of the table that stores the symbol.
  // - kParam: the function op that declares the parameter.
  // - kFunction: the function op (or nullptr for the program) in whose body
  //   the function is declared.
  mlir::Operation *scope = nullptr;
  // kClosureVar only: the function op that owns the environment holding the
  // variable; nullptr if unknown. Always nullptr for other kinds.
  mlir::Operation *owner = nullptr;

  // Returns true if `a` and `b` are the same variable: the same name,
  // declared in the same scope.
  friend bool operator==(const JsirSymbol &a, const JsirSymbol &b) {
    return a.name == b.name && a.scope == b.scope;
  }
};

class JsirScopeTable {
 public:
  // `scope` is the op of the scope; nullptr for the program.
  explicit JsirScopeTable(mlir::Operation *scope = nullptr) : scope_(scope) {}

  JsirScopeTable(const JsirScopeTable &) = delete;
  JsirScopeTable &operator=(const JsirScopeTable &) = delete;

  // The op of the scope; nullptr for the program.
  mlir::Operation *scope() const { return scope_; }

  // Declares `name` in this scope (the symbol's `scope` is scope()). Declaring
  // a name that is already declared has no effect (the first declaration
  // wins). Returns the symbol for `name`, which stays valid until the table is
  // destroyed.
  const JsirSymbol &Declare(llvm::StringRef name, JsirSymbolKind kind,
                            mlir::Operation *owner = nullptr);

  // Returns the symbol declared for `name`, or null.
  const JsirSymbol *Lookup(llvm::StringRef name) const;

  bool Contains(llvm::StringRef name) const { return symbols_.contains(name); }

  // All symbols, in declaration order.
  const std::vector<const JsirSymbol *> &symbols() const { return order_; }

  size_t size() const { return order_.size(); }
  bool empty() const { return order_.empty(); }

 private:
  friend class JsirScopeRegistry;

  // Makes `scope` the op of this scope, and of every symbol in it.
  void SetScope(mlir::Operation *scope);

  // Sets the `owner` of every symbol owned by `old_owner` to `new_owner`.
  void ReplaceOwner(mlir::Operation *old_owner, mlir::Operation *new_owner);

  mlir::Operation *scope_;
  // StringMap entries are allocated individually, so pointers to them stay
  // valid as the map grows.
  llvm::StringMap<JsirSymbol> symbols_;
  std::vector<const JsirSymbol *> order_;
};

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_ANALYSES_SCOPE_TABLE_JSIR_SCOPE_TABLE_H_
