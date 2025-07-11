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

#include "maldoca/js/ir/transforms/split_sequence_expressions/pass.h"

#include <cassert>
#include <vector>

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "maldoca/js/ir/ir.h"
#include "maldoca/js/ir/jsir_utils.h"

namespace maldoca {

// From:
// %a = ...
// %b = ...
// %c = ...
// %expr = jsir.sequence_expression(%a, %b, %c)
// jsir.return_statement(%expr)
//
// To:
// %a = ...
// jsir.expression_statement(%a)
// %b = ...
// jsir.expression_statement(%b)
// %c = ...
// jsir.return_statement(%c)

void SplitSequenceExpressions(mlir::Operation *root) {
  mlir::MLIRContext *context = root->getContext();
  mlir::OpBuilder builder{context};

  std::vector<mlir::Block *> modified_blocks;
  root->walk([&](JsirSequenceExpressionOp op) {
    mlir::Block *parent_block = op->getBlock();

    for (mlir::Operation *user : op->getUsers()) {
      if (!(llvm::isa<JsirReturnStatementOp>(user) ||
            llvm::isa<JsirExpressionStatementOp>(user))) {
        return;
      }
    }

    for (mlir::Value expr : op.getExpressions().drop_back(1)) {
      builder.setInsertionPointAfterValue(expr);
      JsirExpressionStatementOp::create(builder, expr.getLoc(), expr);
    }

    mlir::Value last_expr = op.getExpressions().back();
    op.replaceAllUsesWith(last_expr);
    op.erase();

    if (parent_block != nullptr) {
      modified_blocks.push_back(parent_block);
    }
  });

  // If the statement is a child of another statement, then we can't split it
  // into two statements. For example, consider this `with` statement:
  //
  // ```
  // with (x)
  //   a, b;
  //   ~~~~~ body
  // ```
  //
  // The `body` is a single statement, and we can't replace it with two
  // statements.
  //
  // More specifically, the `with` statement looks like this in JSHIR:
  //
  // ```
  // %x = jsir.identifier {"x"}
  // jshir.with_statement (%x) {
  //   %a = jsir.identifier {"a"}
  //   %b = jsir.identifier {"b"}
  //   %expr = jsir.sequence_expression(%a, %b)
  //   jsir.expression_statement(%expr)
  // }
  // ```
  //
  // If we simplify split `body` into two statements like this:
  //
  // ```
  // %x = jsir.identifier {"x"}
  // jshir.with_statement (%x) {
  //   %a = jsir.identifier {"a"}
  //   jsir.expression_statement(%a)
  //   %b = jsir.identifier {"b"}
  //   jsir.expression_statement(%b)
  // }
  // ```
  //
  // Then we can't correctly convert `body` into a `JsStatement` node in the
  // AST. In the implementation, only one of the two statements gets kept.
  //
  // Therefore, we need to wrap the two statements in a block:
  //
  // ```
  // with (x) {
  //   a;
  //   b;
  // }
  // ```
  //
  // Or, in JSHIR, like this:
  //
  // ```
  // %x = jsir.identifier {"x"}
  // jshir.with_statement (%x) {
  //   jshir.block_statement {
  //     %a = jsir.identifier {"a"}
  //     jsir.expression_statement(%a)
  //     %b = jsir.identifier {"b"}
  //     jsir.expression_statement(%b)
  //   }
  // }
  // ```
  //
  for (mlir::Block *block : modified_blocks) {
    if (IsStatementBlock(*block)) {
      WrapBlockContentWithBlockStatement(*block);
    }
  }
}

}  // namespace maldoca
