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

#include "maldoca/js/ir/analyses/dynamic_constant_propagation/symbol_mutation_info.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/ir/analyses/scope.h"
#include "maldoca/js/ir/ir.h"
#include "maldoca/js/ir/jsir_utils.h"

namespace maldoca {

LvalueRootSymbols GetLvalueRootSymbols(const BabelScopes &scopes,
                                       mlir::Value value) {
  using Ret = LvalueRootSymbols;

  CHECK(value != nullptr);
  mlir::Operation *op = value.getDefiningOp();
  if (op == nullptr) {
    return Ret{};
  }

  return llvm::TypeSwitch<mlir::Operation *, Ret>(op)
      .Case([&](JsirIdentifierRefOp op) -> Ret {
        return LvalueRootSymbols{
            .assignment_symbols = {GetSymbolId(scopes, op)},
            .mutation_symbols = {},
        };
      })
      .Case([&](JsirMemberExpressionRefOp op) -> Ret {
        return GetLvalueRootSymbols(scopes, op.getObject());
      })
      .Case([&](JsirIdentifierOp op) -> Ret {
        return {
            .assignment_symbols = {},
            .mutation_symbols = {GetSymbolId(scopes, op)},
        };
      })
      .Case([&](JsirMemberExpressionOp op) -> Ret {
        return GetLvalueRootSymbols(scopes, op.getObject());
      })
      .Case([&](JsirObjectPatternRefOp op) -> Ret {
        absl::StatusOr<mlir::ValueRange> property_values =
            GetExprsRegionValues(op.getProperties_());
        if (!property_values.ok()) {
          return {};
        }

        LvalueRootSymbols lvalue_root_symbols;
        for (mlir::Value property_value : *property_values) {
          auto op = property_value.getDefiningOp<JsirObjectPropertyRefOp>();
          if (op == nullptr) {
            continue;
          }

          lvalue_root_symbols += GetLvalueRootSymbols(scopes, op.getValue());
        }
        return lvalue_root_symbols;
      })
      .Default(Ret{});
}

absl::flat_hash_map<JsSymbolId, SymbolMutationInfo> GetSymbolMutationInfos(
    const BabelScopes &scopes, mlir::Operation *root) {
  absl::flat_hash_map<JsSymbolId, SymbolMutationInfo> infos;
  root->walk([&](mlir::Operation *op) {
    llvm::TypeSwitch<mlir::Operation *, void>(op)
        .Case([&](JsirVariableDeclaratorOp op) {
          if (op.getInit() == nullptr) {
            return;
          }

          LvalueRootSymbols root_symbols =
              GetLvalueRootSymbols(scopes, op.getId());
          for (const JsSymbolId &symbol : root_symbols.assignment_symbols) {
            infos[symbol].num_assignments++;
          }
          for (const JsSymbolId &symbol : root_symbols.mutation_symbols) {
            infos[symbol].num_mutations++;
          }
        })
        .Case([&](JsirFunctionDeclarationOp op) {
          if (!op.getId().has_value()) {
            return;
          }

          JsSymbolId def_symbol = GetSymbolId(scopes, *op.getId());
          infos[def_symbol].num_assignments++;
        })
        .Case([&](JsirFunctionExpressionOp op) {
          if (!op.getId().has_value()) {
            return;
          }

          JsSymbolId def_symbol = GetSymbolId(scopes, *op.getId());
          infos[def_symbol].num_assignments++;
        })
        .Case([&](JsirAssignmentExpressionOp op) {
          LvalueRootSymbols root_symbols =
              GetLvalueRootSymbols(scopes, op.getLeft());
          for (const JsSymbolId &symbol : root_symbols.assignment_symbols) {
            infos[symbol].num_assignments++;
          }
          for (const JsSymbolId &symbol : root_symbols.mutation_symbols) {
            infos[symbol].num_mutations++;
          }
        })
        .Case([&](JsirUpdateExpressionOp op) {
          LvalueRootSymbols root_symbols =
              GetLvalueRootSymbols(scopes, op.getArgument());
          for (const JsSymbolId &symbol : root_symbols.assignment_symbols) {
            infos[symbol].num_assignments++;
          }
          for (const JsSymbolId &symbol : root_symbols.mutation_symbols) {
            infos[symbol].num_mutations++;
          }
        });
  });

  return infos;
}

}  // namespace maldoca
