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

#ifndef MALDOCA_JS_IR_ANALYSES_DYNAMIC_CONSTANT_PROPAGATION_SYMBOL_MUTATION_INFO_H_
#define MALDOCA_JS_IR_ANALYSES_DYNAMIC_CONSTANT_PROPAGATION_SYMBOL_MUTATION_INFO_H_

// TODO(tzx): Investigate if we can use Babel's binding information instead of
// rolling our own.

// Collects symbols that are mutated by a given operation.
//
// Background:
//
// We only inline when a symbol is assigned only once. For example, we don't
// inline the following code because `dst` is assigned twice:
//
// ```
// let dst = ...;
// dst = ...;
// let src = dst;
// use(src);  // We don't inline this into `use(dst)`.
// ```
//
// Therefore, we need to count the number of assignments for each symbol.
//
// Note that we also avoid inline when a symbol is not re-assigned but the
// underlying object is mutated. For example, we don't inline the following code
// because `dst` is mutated:
//
// ```
// let dst = {...};
// dst.field = ...;  // We consider `dst` mutated.
// let src = dst;
// use(src);         // We don't inline this into `use(dst)`.
// ```
//
// Note that our current analysis misses the case when a symbol is aliased. For
// example, the following code is not inlined because `src` is aliased to `dst`:
//
// ```
// let dst = { field: value };
// let dst2 = dst;
// dst2.field = value2;  // `dst2` is detected as mutated, but `dst` is not.
// let src = dst;
// use(src.field);       // This will be inlined incorrectly into `use(value)`.
// ```

#include <cstddef>
#include <tuple>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "absl/container/flat_hash_map.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/babel/scope.h"

namespace maldoca {

struct LvalueRootSymbols {
  // `a = <value>`: `a` is assigned.
  std::vector<JsSymbolId> assignment_symbols;

  // `a.field = <value>`: `a` is mutated.
  std::vector<JsSymbolId> mutation_symbols;

  LvalueRootSymbols &operator+=(const LvalueRootSymbols &other) {
    llvm::append_range(assignment_symbols, other.assignment_symbols);
    llvm::append_range(mutation_symbols, other.mutation_symbols);
    return *this;
  }
};

LvalueRootSymbols GetLvalueRootSymbols(const BabelScopes &scopes,
                                       mlir::Value value);

struct SymbolMutationInfo {
  size_t num_assignments;
  size_t num_mutations;

  bool operator==(const SymbolMutationInfo &other) const {
    return std::forward_as_tuple(num_assignments, num_mutations) ==
           std::forward_as_tuple(other.num_assignments, other.num_mutations);
  }

  template <typename Sink>
  friend void AbslStringify(Sink &sink, const SymbolMutationInfo &s) {
    absl::Format(&sink, "assignments: %zu, mutations: %zu", s.num_assignments,
                 s.num_mutations);
  }
};

absl::flat_hash_map<JsSymbolId, SymbolMutationInfo> GetSymbolMutationInfos(
    const BabelScopes &scopes, mlir::Operation *root);

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_ANALYSES_DYNAMIC_CONSTANT_PROPAGATION_SYMBOL_MUTATION_INFO_H_
