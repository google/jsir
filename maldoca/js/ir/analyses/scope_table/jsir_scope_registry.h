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

// JsirScopeRegistry: the scopes and declarations of a program, populated by
// the code that creates them, as an MLIR analysis.
//
// Model
// -----
//
// The registry starts out empty (just an empty program table). Whoever creates
// a function creates its table; whoever declares a variable registers it in
// the table of the scope it belongs to. By the time the producer (e.g. the
// Hermes decompiler) has finished, the registry holds every scope and every
// declaration in the program. Nothing is discovered by scanning the IR.
//
//   JsirScopeRegistry (module analysis)
//   ├── program():  program-level declarations (e.g. closure variables);
//   │               its scope op is nullptr.
//   └── per scope op (a function op): a JsirScopeTable with the function's
//         local declarations.
//
// Tables are keyed by the op of their scope. When an op is replaced by
// another that takes over its body (e.g. a `hermescir.func` lowered to a
// `jsir.function_declaration`), the code doing the replacement calls
// TransferScope, which re-keys the table and repoints the `scope` of its
// symbols, and the `owner` of the closure variables the function owns.
//
// What is NOT stored:
//
// - Names that an op declares itself are read from the op when needed:
//   parameters (the `jsir.identifier_ref`s in a JsirFunctionOpInterface op's
//   params region) and function names (the `id` of
//   `jsir.function_declaration`s).
// - Scopes nested in a function (e.g. the pseudo-functions created by the
//   Hermes decompiler's liftJumps) get no table: their identifiers resolve to
//   the enclosing function's table, except for their own parameters and the
//   functions they declare, which are read from the IR.
// - Uses are never registered. A use is resolved on demand (Resolve), and the
//   uses of a symbol (GetUses) or whether it is captured (IsCaptured) are
//   computed on demand by walking the relevant IR.
//
// Resolution order for an identifier (Resolve). Walking outwards from the
// identifier, for each enclosing op:
//
// 1. If it is a function op: its parameters (kParam, scope = the op), then
//    the functions declared directly in its body (kFunction, scope = the op).
//    Parameters and function names are lexically scoped, so this applies to
//    every enclosing function, including those outside the nearest table.
// 2. If it is the nearest op with a table: that table (scope = its op).
//    Tables further out are not consulted: variables shared across functions
//    are kClosureVars in the program table.
// 3. If it is the `jsir.program`: the functions declared directly in it
//    (kFunction, scope = nullptr).
//
// Then the program table (scope = nullptr); otherwise a global
// (std::nullopt).
//
// Lifetime
// --------
//
// MLIR invalidates every analysis that a pass does not mark preserved, after
// each pass (mlir/Pass/AnalysisManager.h). The registry is not derived from
// the IR, so it could not be recomputed: keeping it correct is the job of the
// code that creates functions and declarations, and it must survive every
// pass, including passes that know nothing about it. Hence it is anchored on
// the module and isInvalidated() always returns false (AnalysisMap::invalidate
// asks that hook even when a pass preserves nothing). Passes never need to
// mark it preserved.
//
// Threading
// ---------
//
// Function-level passes run in parallel, one thread per function, and can
// only reach module analyses through
// `getCachedParentAnalysis<JsirScopeRegistry>(module)`; they cannot create
// them. So:
//
// - A module pass that runs before any function-level pass creates the
//   registry and every function's table up front (in the Hermes decompiler,
//   `jsirInitScopes`).
// - During function-level passes the set of tables is not modified, so
//   looking them up (GetScopeTable, GetScopeTableFor) is a read-only
//   operation that is safe to call concurrently.
// - A function-level pass writes only to its own function's table, so no two
//   threads write to the same object, and no lock is needed.
//
// Calls that add, remove or re-key tables (AddScope, TransferScope,
// RemoveScope, MergeScopeInto), writes to the program table, and the
// on-demand queries (which walk IR that other threads may be changing) are
// for module-level code only.

#ifndef MALDOCA_JS_IR_ANALYSES_SCOPE_TABLE_JSIR_SCOPE_REGISTRY_H_
#define MALDOCA_JS_IR_ANALYSES_SCOPE_TABLE_JSIR_SCOPE_REGISTRY_H_

#include <memory>
#include <optional>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/AnalysisManager.h"
#include "maldoca/js/ir/analyses/scope_table/jsir_scope_table.h"

namespace maldoca {

class JsirScopeRegistry {
 public:
  // MLIR analysis constructor. `anchor` is the op the analysis is attached to
  // (the module). The registry starts with an empty program table and no
  // other scopes.
  explicit JsirScopeRegistry(mlir::Operation *anchor) : anchor_(anchor) {}

  JsirScopeRegistry(const JsirScopeRegistry &) = delete;
  JsirScopeRegistry &operator=(const JsirScopeRegistry &) = delete;

  // MLIR analysis invalidation hook: never invalidated (see "Lifetime").
  bool isInvalidated(
      const mlir::AnalysisManager::PreservedAnalyses & /*preserved*/) const {
    return false;
  }

  mlir::Operation *anchor() const { return anchor_; }

  // ---------------------------------------------------------------------------
  // Tables.
  // ---------------------------------------------------------------------------

  // The program's table (its scope op is nullptr).
  JsirScopeTable &program() { return program_; }
  const JsirScopeTable &program() const { return program_; }

  // Creates the (empty) table of the scope `op`, or returns the existing one.
  // Module-level code only.
  JsirScopeTable &AddScope(mlir::Operation *op);

  // Returns the table of the scope `op`, or null.
  //
  // Lookups are const so that const code (e.g. the on-demand queries) can
  // make them. They return a mutable table because the tables are not part of
  // the registry's own state: the registry owns the set of tables, and a
  // lookup does not change it.
  JsirScopeTable *GetScopeTable(mlir::Operation *op) const;

  // Returns the table of the nearest op enclosing `op` (or `op` itself) that
  // has a table, or null.
  JsirScopeTable *GetScopeTableFor(mlir::Operation *op) const;

  // Moves the table of `old_op` to `new_op`, for when `new_op` replaces
  // `old_op` (e.g. takes over its body): the `scope` of every symbol in the
  // table becomes `new_op`, and so does the `owner` of every program-table
  // symbol owned by `old_op`. Replaces any table `new_op` already had. Does
  // nothing if `old_op` has no table. Call it before `old_op` is erased.
  // Module-level code only.
  void TransferScope(mlir::Operation *old_op, mlir::Operation *new_op);

  // Removes the table of `op`, e.g. when the function is deleted. Program
  // symbols owned by `op` get no owner (nullptr). Module-level code only.
  void RemoveScope(mlir::Operation *op);

  // Moves the declarations of `callee_op` into `caller_op`, then removes
  // `callee_op`'s table; for when the body of `callee_op` is inlined into
  // `caller_op`. A function's closure variables belong to its activation,
  // and the inlined body runs in the caller's activation, so every program
  // symbol owned by `callee_op` becomes owned by `caller_op` (unlike
  // RemoveScope, which clears the owner). Module-level code only.
  //
  // TODO(olatayo): Once a pass inlines functions: each inlined copy of the
  // callee (several call sites, or a call in a loop) needs its own variables,
  // i.e. distinct names; and names declared in both scopes must be handled
  // (the inliner then has to rename the callee's).
  void MergeScopeInto(mlir::Operation *callee_op, mlir::Operation *caller_op);

  // ---------------------------------------------------------------------------
  // On-demand queries. These walk the IR; module-level code only.
  // ---------------------------------------------------------------------------

  // Returns what the `jsir.identifier` / `jsir.identifier_ref` `identifier`
  // refers to, or std::nullopt for a global (or any other op). See
  // "Resolution order" above.
  std::optional<JsirSymbol> Resolve(mlir::Operation *identifier) const;

  // Resolves `name` as if it were used at `site`.
  std::optional<JsirSymbol> Resolve(mlir::Operation *site,
                                    llvm::StringRef name) const;

  // Returns every `jsir.identifier` / `jsir.identifier_ref` that resolves to
  // `symbol`, in IR walk order, excluding parameter declaration sites. Walks
  // only the declaring scope's IR (the whole anchor for program symbols).
  std::vector<mlir::Operation *> GetUses(const JsirSymbol &symbol) const;

  // Returns true if `symbol` is used from a function other than the one it
  // belongs to: its declaring scope op (for closure variables, the owner;
  // none for unowned ones). Nested function ops (e.g. pseudo-functions) count
  // as other functions. Always false for kFunction.
  bool IsCaptured(const JsirSymbol &symbol) const;

 private:
  mlir::Operation *anchor_;
  JsirScopeTable program_;
  // Tables are heap-allocated, so they never move.
  llvm::DenseMap<mlir::Operation *, std::unique_ptr<JsirScopeTable>> scopes_;
};

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_ANALYSES_SCOPE_TABLE_JSIR_SCOPE_REGISTRY_H_
