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

#include "maldoca/js/ir/ir.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"

namespace maldoca {
void IfStatementElimination(mlir::Operation *root_op) {
  root_op->walk([&](JshirIfStatementOp op) {
    auto condition = op.getTest().getDefiningOp<JsirBooleanLiteralOp>();
    if (condition == nullptr) {
      return;
    }

    if (condition.getValue()) {
      // Use the consequent block to replace the if statement.
      mlir::Region *consequent_region = &op.getConsequent();
      mlir::Block &consequent_block = consequent_region->front();

      for (mlir::Operation &op_to_move :
           llvm::make_early_inc_range(consequent_block.getOperations())) {
        op_to_move.moveBefore(op);
      }
      op.erase();
    } else {
      // Use the alternate block to replace the if statement.
      mlir::Region *alternate_region = &op.getAlternate();
      if (alternate_region->empty()) {
        op.erase();
        return;
      }

      mlir::Block &alternate_block = alternate_region->front();
      for (mlir::Operation &op_to_move :
           llvm::make_early_inc_range(alternate_block.getOperations())) {
        op_to_move.moveBefore(op);
      }
      op.erase();
    }
  });
}

void WhileStatementElimination(mlir::Operation *root_op) {
  root_op->walk([&](JshirWhileStatementOp op) {
    mlir::Region &test_region = op.getTest();
    if (test_region.empty()) {
      return;
    }

    auto expr_region_end_op =
        llvm::dyn_cast<JsirExprRegionEndOp>(&test_region.front().back());
    if (expr_region_end_op == nullptr) {
      return;
    }

    auto condition_op =
        expr_region_end_op.getOperand().getDefiningOp<JsirBooleanLiteralOp>();
    if (condition_op == nullptr) {
      return;
    }

    if (!condition_op.getValue()) {
      // Condition is constantly false, eliminate the entire loop.
      op.erase();
    }

    // If condition is true, it's an infinite loop. For now, we leave it.
  });
}

void DeadCodeElimination(mlir::Operation *root_op) {
  IfStatementElimination(root_op);
  WhileStatementElimination(root_op);
}
} // namespace maldoca
