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

#include "maldoca/js/ir/transforms/dead_code_elimination/pass.h"

#include <tuple>
#include <vector>

#include "absl/log/log.h"
#include "absl/container/flat_hash_map.h"
#include "llvm/ADT/STLExtras.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/ir/ir.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace maldoca
{

  void IfStatementElimination(mlir::Operation *root_op)
  {
    root_op->walk([&](JshirIfStatementOp op)
                  {
    auto condition = op.getTest().getDefiningOp<JsirBooleanLiteralOp>();
    if (condition == nullptr) {
      return;
    }

    if (condition.getValue()) {
      // Use the consequent block to replace the if statement.
      mlir::Region* consequent_region = &op.getConsequent();
      mlir::Block& consequent_block = consequent_region->front();

      for (mlir::Operation& op_to_move :
           llvm::make_early_inc_range(consequent_block.getOperations())) {
        op_to_move.moveBefore(op);
      }
      op.erase();
    } else {
      // Use the alternate block to replace the if statement.
      mlir::Region* alternate_region = &op.getAlternate();
      if (alternate_region->empty()) {
        op.erase();
        return;
      }

      mlir::Block& alternate_block = alternate_region->front();
      for (mlir::Operation& op_to_move :
           llvm::make_early_inc_range(alternate_block.getOperations())) {
        op_to_move.moveBefore(op);
      }
      op.erase();
    } });
  }

  void WhileStatementElimination(mlir::Operation *root_op)
  {
    root_op->walk([&](JshirWhileStatementOp op)
                  {
                    mlir::Region &test_region = op.getTest();
                    if (test_region.empty())
                    {
                      return;
                    }

                    auto expr_region_end_op =
                        llvm::dyn_cast<JsirExprRegionEndOp>(&test_region.front().back());
                    if (expr_region_end_op == nullptr)
                    {
                      return;
                    }

                    auto condition_op =
                        expr_region_end_op.getOperand().getDefiningOp<JsirBooleanLiteralOp>();
                    if (condition_op == nullptr)
                    {
                      return;
                    }

                    if (!condition_op.getValue())
                    {
                      // Condition is constantly false, eliminate the entire loop.
                      op.erase();
                    }

                    // If condition is true, it's an infinite loop. For now, we leave it.
                  });
  }

  struct SymbolInfo
  {
    std::vector<mlir::Operation *> definitions;
    std::vector<mlir::Operation *> references;
    std::vector<JsSymbolId> inner_definitions;
    std::vector<JsSymbolId> outer_definitions;
  };

  bool operator==(const JsSymbolId &lhs, const JsSymbolId &rhs)
  {
    return std::forward_as_tuple(lhs.name(), lhs.def_scope_uid()) ==
           std::forward_as_tuple(rhs.name(), rhs.def_scope_uid());
  }

  template <typename H>
  H AbslHashValue(H h, const JsSymbolId &m)
  {
    return H::combine(std::move(h), m.name(), m.def_scope_uid());
  }

  JsSymbolId GetSymbolIdFromAttr(JsirSymbolIdAttr symbol_attr)
  {
    std::string name = symbol_attr.getName().str();
    std::optional<int64_t> scope_uid = symbol_attr.getDefScopeId();
    return JsSymbolId{name, scope_uid};
  }

  void UnusedFunctionElimination(mlir::Operation *root_op)
  {
    absl::flat_hash_map<JsSymbolId, SymbolInfo> symbol_infos;

    root_op->walk([&](mlir::Operation *op)
                  {
    if (llvm::isa<JsirExprsRegionEndOp>(op) || llvm::isa<JsirExprRegionEndOp>(op) ) {
      return;
    } 
    auto trivia = llvm::dyn_cast<JsirTriviaAttr>(op->getLoc());
    if (trivia == nullptr) {
      return;
    }
    JsirSymbolIdAttr symbol = trivia.getReferencedSymbol();
    if (symbol != nullptr) {
      symbol_infos[GetSymbolIdFromAttr(symbol)].references.push_back(op);
    }

    llvm::ArrayRef<JsirSymbolIdAttr> mlir_defined_symbols =
        trivia.getDefinedSymbols();
    for (JsirSymbolIdAttr defined_symbol : mlir_defined_symbols) {
      symbol_infos[GetSymbolIdFromAttr(defined_symbol)].definitions.push_back(
          op);
    } });

    for (auto &[symbol, info] : symbol_infos)
    {
      if (info.definitions.empty())
      {
        LOG(INFO) << "Warning: Symbol has references but no definitions: "
                  << symbol.name() << std::endl;
        continue;
      }
      if (info.definitions.size() > 1)
      {
        LOG(INFO) << "Warning: Multiple definitions for symbol: "
                  << symbol.name() << std::endl;
        for (mlir::Operation *def_op : info.definitions)
        {
          LOG(INFO) << "  Defined in operation: "
                    << def_op->getName().getStringRef().str() << std::endl;
        }
        continue;
      }

      mlir::Operation *def_op = info.definitions[0];
      // Recursively traverse the parent chain of def_op by using getParentOp() and find the first op in the chain that defines symbols.
      // In order to check if an op defines symbols, we can use the trivia attribute like above in the same file.
      mlir::Operation *current_op = def_op->getParentOp();
      while (current_op != nullptr)
      {
        auto current_trivia = llvm::dyn_cast<JsirTriviaAttr>(current_op->getLoc());
        if (current_trivia != nullptr)
        {
          llvm::ArrayRef<JsirSymbolIdAttr> outer_defined_symbols =
              current_trivia.getDefinedSymbols();
          if (!outer_defined_symbols.empty())
          {
            for (JsirSymbolIdAttr outer_defined_symbol_attr : outer_defined_symbols)
            {
              JsSymbolId outer_symbol = GetSymbolIdFromAttr(outer_defined_symbol_attr);
              auto outer_symbol_info_it = symbol_infos.find(outer_symbol);
              if (outer_symbol_info_it != symbol_infos.end())
              {
                info.outer_definitions.push_back(outer_symbol);
                outer_symbol_info_it->second.inner_definitions.push_back(symbol);
              }
            }
            break; // Found the first op in the chain that defines symbols
          }
        }
        current_op = current_op->getParentOp();
      }
    }

    // Check if info.definitions contains duplicates across symbol_infos? If so, print warning
    absl::flat_hash_map<mlir::Operation *, size_t> defined_ops;
    for (const auto &[symbol, info] : symbol_infos)
    {
      for (mlir::Operation *def_op : info.definitions)
      {
        defined_ops[def_op]++;
      }
    }
    for (const auto &[def_op, count] : defined_ops)
    {
      if (count > 1)
      {
        LOG(INFO) << "Warning: Operation defines multiple symbols: "
                  << def_op->getName().getStringRef().str() << std::endl;
      }
    }

    for (auto &[symbol, info] : symbol_infos)
    {
      if (info.references.empty())
      {
        if (info.definitions.size() > 1)
        {
          LOG(INFO) << "Warning: Multiple definitions for symbol: "
                    << symbol.name() << std::endl;
          for (mlir::Operation *def_op : info.definitions)
          {
            LOG(INFO) << "  Defined in operation: "
                      << def_op->getName().getStringRef().str() << std::endl;
          }
        }

        for (mlir::Operation *def_op : info.definitions)
        {
          if (llvm::isa<JsirFunctionDeclarationOp>(def_op))
          {
            std::vector<JsSymbolId> worklist = info.inner_definitions;
            while (!worklist.empty())
            {
              JsSymbolId current_symbol_id = worklist.back();
              worklist.pop_back();

              auto it = symbol_infos.find(current_symbol_id);
              if (it != symbol_infos.end())
              {
                it->second.definitions.clear();
                worklist.insert(worklist.end(),
                                it->second.inner_definitions.begin(),
                                it->second.inner_definitions.end());
              }
            }
            def_op->erase();
            for (const auto &outer_symbol : info.outer_definitions)
            {
              auto outer_it = symbol_infos.find(outer_symbol);
              if (outer_it != symbol_infos.end())
              {
                auto &outer_info = outer_it->second;
                outer_info.inner_definitions.erase(
                    std::remove(outer_info.inner_definitions.begin(),
                                outer_info.inner_definitions.end(),
                                symbol),
                    outer_info.inner_definitions.end());
              }
            }
          }
        }
      }
    }
  }

  void DeadCodeElimination(mlir::Operation *root_op)
  {
    IfStatementElimination(root_op);
    WhileStatementElimination(root_op);
    // TODO: Iteratively eliminate unused functions until no more can be eliminated.
    // TODO: Update the AST after dynamic constant progropagation (before dce).
    UnusedFunctionElimination(root_op);
  }
} // namespace maldoca
