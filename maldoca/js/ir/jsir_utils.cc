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

#include "maldoca/js/ir/jsir_utils.h"

#include <cassert>

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "maldoca/base/status_macros.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {

absl::StatusOr<mlir::Operation *> GetStmtRegionOperation(mlir::Region &region) {
  if (!region.hasOneBlock()) {
    return absl::InvalidArgumentError("Region should have exactly one block.");
  }
  mlir::Block &block = region.front();
  if (block.empty()) {
    return absl::InvalidArgumentError("Block cannot be empty.");
  }
  return &block.back();
}

absl::StatusOr<mlir::Block *> GetStmtsRegionBlock(mlir::Region &region) {
  if (!region.hasOneBlock()) {
    return absl::InvalidArgumentError("Region should have exactly one block.");
  }
  mlir::Block &block = region.front();
  return &block;
}

absl::StatusOr<mlir::Value> GetExprRegionValue(mlir::Region &region) {
  if (!region.hasOneBlock()) {
    return absl::InvalidArgumentError("Region should have exactly one block.");
  }
  mlir::Block &block = region.front();
  if (block.empty()) {
    return absl::InvalidArgumentError("Block cannot be empty.");
  }
  auto expr_region_end = llvm::dyn_cast<JsirExprRegionEndOp>(block.back());
  if (expr_region_end == nullptr) {
    return absl::InvalidArgumentError(
        "Block should end with JsirExprRegionEndOp.");
  }
  return expr_region_end.getArgument();
}

absl::StatusOr<JsirExprsRegionEndOp> GetExprsRegionEndOp(mlir::Region &region) {
  if (!region.hasOneBlock()) {
    return absl::InvalidArgumentError("Region should have exactly one block.");
  }
  mlir::Block &block = region.front();
  if (block.empty()) {
    return absl::InvalidArgumentError("Block cannot be empty.");
  }
  auto exprs_region_end = llvm::dyn_cast<JsirExprsRegionEndOp>(block.back());
  if (exprs_region_end == nullptr) {
    return absl::InvalidArgumentError(
        "Block should end with JsirExprsRegionEndOp.");
  }
  return exprs_region_end;
}

absl::StatusOr<mlir::ValueRange> GetExprsRegionValues(mlir::Region &region) {
  MALDOCA_ASSIGN_OR_RETURN(JsirExprsRegionEndOp exprs_region_end,
                           GetExprsRegionEndOp(region));
  return exprs_region_end.getArguments();
}

// ============================================================================
//  Block-manipulation functions
// ============================================================================

bool IsStatementBlock(mlir::Block &block) {
  mlir::Region *region = block.getParent();
  if (region == nullptr) {
    return false;
  }

  return llvm::TypeSwitch<mlir::Operation *, bool>(region->getParentOp())
      .Case([&](JshirWithStatementOp parent_op) {
        // interface WithStatement <: Statement {
        //   object: Expression;
        //   body: Statement;
        // }
        return region == &parent_op.getBody();
      })
      .Case([&](JshirLabeledStatementOp parent_op) {
        // interface LabeledStatement <: Statement {
        //   label: Identifier;
        //   body: Statement;
        // }
        return region == &parent_op.getBody();
      })
      .Case([&](JshirIfStatementOp parent_op) {
        // interface IfStatement <: Statement {
        //   test: Expression;
        //   consequent: Statement;
        //   alternate: Statement | null;
        // }
        return region == &parent_op.getConsequent() ||
               region == &parent_op.getAlternate();
      })
      .Case([&](JshirWhileStatementOp parent_op) {
        // interface WhileStatement <: Statement {
        //   test: Expression;
        //   body: Statement;
        // }
        return region == &parent_op.getBody();
      })
      .Case([&](JshirDoWhileStatementOp parent_op) {
        // interface DoWhileStatement <: Statement {
        //   body: Statement;
        //   test: Expression;
        // }
        return region == &parent_op.getBody();
      })
      .Case([&](JshirForStatementOp parent_op) {
        // interface ForStatement <: Statement {
        //   init: VariableDeclaration | Expression | null;
        //   test: Expression | null;
        //   update: Expression | null;
        //   body: Statement;
        // }
        return region == &parent_op.getBody();
      })
      .Case([&](JshirForInStatementOp parent_op) {
        // interface ForInStatement <: Statement {
        //   left: VariableDeclaration | LVal;
        //   right: Expression;
        //   body: Statement;
        // }
        return region == &parent_op.getBody();
      })
      .Case([&](JshirForOfStatementOp parent_op) {
        // interface ForOfStatement <: Statement {
        //   left: VariableDeclaration | LVal;
        //   right: Expression;
        //   body: Statement;
        //   await: boolean;
        // }
        return region == &parent_op.getBody();
      })
      .Default(false);
}

void WrapBlockContentWithBlockStatement(mlir::Block &block) {
  mlir::Region *region = block.getParent();
  mlir::MLIRContext *context = region->getContext();

  // +-------------+-------------+
  // | Before      | After       |
  // +-------------+-------------+
  // | ^block:     | ^block:     |
  // |   op1       |             |
  // |   op2       | ^new_block: |
  // |   ...       |   op1       |
  // |             |   op2       |
  // |             |   ...       |
  // +-------------+-------------+
  mlir::Block *new_block = block.splitBlock(block.begin());
  assert(block.empty());

  // After:
  //
  //  ^block:
  //    JshirBlockStatement(/*directives=*/{}, /*body=*/{})
  //
  //  ^new_block:
  //    op1
  //    op2
  //    ...
  mlir::OpBuilder builder{context};
  builder.setInsertionPointToStart(&block);
  auto block_stmt_op = JshirBlockStatementOp::create(builder, region->getLoc());

  // `directives` is empty, but we need to keep an empty block in the region.
  block_stmt_op.getDirectives().emplaceBlock();

  // After:
  //
  //  ^block:
  //    JshirBlockStatement(
  //      /*directives=*/{
  //      ^empty_block:
  //      },
  //      /*body=*/{
  //      ^new_block:
  //        op1
  //        op2
  //        ...
  //      })
  new_block->moveBefore(&block_stmt_op.getBody(),
                        block_stmt_op.getBody().end());
}

}  // namespace maldoca
