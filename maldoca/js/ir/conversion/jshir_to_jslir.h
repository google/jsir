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

#ifndef MALDOCA_JS_IR_CONVERSION_JSHIR_TO_JSLIR_H_
#define MALDOCA_JS_IR_CONVERSION_JSHIR_TO_JSLIR_H_

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "maldoca/js/ir/conversion/jslir_jump_env.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {

class JshirToJslir {
 public:
  explicit JshirToJslir(mlir::OpBuilder &builder) : builder_(builder) {}

  // Converts the op and inserts the new op at the current builder insertion
  // point.
  void VisitOperation(mlir::Operation *hir_op);

  // Converts all ops in hir_ops and inserts the new ops at the current builder
  // insertion point.
  using OperationIteratorRange =
      llvm::iterator_range<llvm::iplist<mlir::Operation>::iterator>;
  void VisitOperations(OperationIteratorRange hir_ops);

  // Converts all ops in hir_block and inserts the new ops at the current
  // builder insertion point.
  void VisitBlock(mlir::Block *hir_block);

  // REQUIRES: hir_region has exactly 1 block, ending with JsirExprRegionEndOp.
  //
  // Input (hir_region):
  // {
  //   %0 = hir_op_1
  //   ...
  //   %n = hir_op_n
  //   "jsir.expr_region_end" (%n)
  // }
  //
  // Output (lir_block):
  //   %0 = lir_op_1
  //   ...
  //   %n = lir_op_n
  //
  // Return:
  //   %n
  void VisitExprRegion(mlir::Region *hir_region, mlir::Block *lir_block,
                       absl::FunctionRef<void(mlir::Value)> on_result) {
    // Save insertion point.
    // Will revert at the end.
    mlir::OpBuilder::InsertionGuard insertion_guard(builder_);
    builder_.setInsertionPointToEnd(lir_block);

    mlir::Block &hir_block = hir_region->front();

    // Lower all the ops except for the terminator.
    VisitOperations(hir_block.without_terminator());

    // Lower the JsirExprRegionEndOp.
    auto *hir_terminator = hir_block.getTerminator();
    auto hir_end_op = llvm::cast<JsirExprRegionEndOp>(hir_terminator);
    mlir::Value lir_result = mapping_.lookup(hir_end_op.getArgument());

    on_result(lir_result);
  }

  // REQUIRES: hir_region has exactly 1 block, ending with JsirExprRegionEndOp.
  //
  // Input (hir_region):
  // {
  //   %0 = hir_op_1
  //   ...
  //   %n = hir_op_n
  //   "jsir.expr_region_end" (%n)
  // }
  //
  // Output (lir_block):
  //   %0 = lir_op_1
  //   ...
  //   %n = lir_op_n
  //   "jslir.<end_op>" (%n) [^successor_1, ..., ^successor_m]
  template <typename EndOp, typename... Successors>
  void VisitExprRegion(mlir::Region *hir_region, mlir::Block *lir_block,
                       Successors... lir_successors) {
    VisitExprRegion(hir_region, lir_block, [&](mlir::Value lir_result) {
      builder_.create<EndOp>(builder_.getUnknownLoc(), lir_result,
                             lir_successors...);
    });
  }

  // REQUIRES: hir_region has exactly 1 block.
  //
  // Input (hir_region):
  // {
  //   %0 = hir_op_1
  //   ...
  //   %n = hir_op_n
  // }
  //
  // Output (lir_block):
  //   %0 = lir_op_1
  //   ...
  //   %n = lir_op_n
  //   "jslir.<end_op>" () [^successor_1, ..., ^successor_m]
  template <typename EndOp, typename... Successors>
  EndOp VisitStmtOrStmtsRegion(mlir::Region *hir_region, mlir::Block *lir_block,
                               Successors... lir_successors) {
    // Save insertion point.
    // Will revert at the end.
    mlir::OpBuilder::InsertionGuard insertion_guard(builder_);
    builder_.setInsertionPointToEnd(lir_block);

    CHECK(hir_region->hasOneBlock());
    mlir::Block &hir_block = hir_region->front();

    // Lower all the ops.
    VisitOperations(hir_block);

    return builder_.create<EndOp>(builder_.getUnknownLoc(), lir_successors...);
  }

  // REQUIRES: hir_region has exactly 1 block.
  //
  // Input (hir_region):
  // {
  //   %0 = hir_op_1
  //   ...
  //   %n = hir_op_n
  //   "jsir.expr_region_end" (%n)
  // }
  //
  // Output (lir_block):
  //   "jslir.control_flow_marker" {kind} (%token)
  //   %0 = lir_op_1
  //   ...
  //   %n = lir_op_n
  //   "cf.cond_br"(%n)[^successor]
  void VisitStmtOrStmtsRegionWithMarkerOp(JsirControlFlowMarkerKind kind,
                                          mlir::Value token,
                                          mlir::Region *hir_region,
                                          mlir::Block *lir_block,
                                          mlir::Block *lir_successor) {
    // Save insertion point.
    // Will revert at the end.
    mlir::OpBuilder::InsertionGuard insertion_guard(builder_);
    builder_.setInsertionPointToStart(lir_block);
    CreateStmt<JslirControlFlowMarkerOp>(hir_region->getParentOp(), kind,
                                         token);
    VisitStmtOrStmtsRegion<mlir::cf::BranchOp>(hir_region, lir_block,
                                               lir_successor);
  }

  // REQUIRES: hir_region has exactly 1 block, ending with JsirExprRegionEndOp.
  //
  // Input (hir_region):
  // {
  //   %0 = hir_op_1
  //   ...
  //   %n = hir_op_n
  // }
  //
  // Output (lir_block):
  //   "jslir.control_flow_marker" {kind} (%token)
  //   %0 = lir_op_1
  //   ...
  //   %n = lir_op_n
  //   "cf.br"[^successor]
  void VisitExprRegionWithMarkerOp(JsirControlFlowMarkerKind kind,
                                   mlir::Value token, mlir::Region *hir_region,
                                   mlir::Block *lir_block,
                                   mlir::Block *lir_successor_true,
                                   mlir::Block *lir_successor_false) {
    // Save insertion point.
    // Will revert at the end.
    mlir::OpBuilder::InsertionGuard insertion_guard(builder_);
    builder_.setInsertionPointToStart(lir_block);
    CreateStmt<JslirControlFlowMarkerOp>(hir_region->getParentOp(), kind,
                                         token);
    VisitExprRegion(hir_region, lir_block, [&](mlir::Value lir_value) {
      CreateCondBranch(builder_.getUnknownLoc(), lir_value, lir_successor_true,
                       {}, lir_successor_false, {});
    });
  }

  // REQUIRES: hir_region has 0 or 1 block.
  //
  // If hir_region is non-empty, same as VisitExprRegion().
  //
  // If hir_region is empty:
  // Input (hir_region):
  // {
  // }
  //
  // Output (lir_block):
  //   "jslir.<end_op>" () [^successor_1, ..., ^successor_m]
  template <typename EndOp, typename... Successors>
  EndOp VisitOptionalStmtOrStmtsRegion(mlir::Region *hir_region,
                                       mlir::Block *lir_block,
                                       Successors... lir_successors) {
    if (!hir_region->empty()) {
      return VisitStmtOrStmtsRegion<EndOp>(hir_region, lir_block,
                                           lir_successors...);
    }

    mlir::OpBuilder::InsertionGuard insertion_guard(builder_);
    builder_.setInsertionPointToEnd(lir_block);
    return builder_.create<EndOp>(builder_.getUnknownLoc(), lir_successors...);
  }

  // REQUIRES: hir_region has exactly 1 block. This block might end with
  // JsirExprRegionEndOp or JsirExprsRegionEndOp or nothing.
  //
  // Case 1:
  //   Input (hir_region):
  //   {
  //     %0 = hir_op_1
  //     ...
  //     %n = hir_op_n
  //     "jsir.expr_region_end" (%n)
  //   }
  //
  //   Output (lir_block):
  //     %0 = lir_op_1
  //     ...
  //     %n = lir_op_n
  //     "jslir.<end_op>" (%n) [^successor_1, ..., ^successor_m]
  //
  // Case 2:
  //   Input (hir_region):
  //   {
  //     %0 = hir_op_1
  //     ...
  //     %n = hir_op_n
  //     "jsir.exprs_region_end" (%0, ..., %n)
  //   }
  //
  //   Output (lir_block):
  //     %0 = lir_op_1
  //     ...
  //     %n = lir_op_n
  //     "jslir.<end_op>" (%0, ..., %n) [^successor_1, ..., ^successor_m]
  //
  // Case 3:
  //   Input (hir_region):
  //   {
  //     %0 = hir_op_1
  //     ...
  //     %n = hir_op_n
  //   }
  //
  //   Output (lir_block):
  //     %0 = lir_op_1
  //     ...
  //     %n = lir_op_n
  //     "jslir.<end_op>" () [^successor_1, ..., ^successor_m]
  template <typename EndOp, typename... Successors>
  EndOp VisitUnknownRegion(mlir::Region *hir_region, mlir::Block *lir_block,
                           Successors... lir_successors) {
    // Save insertion point.
    // Will revert at the end.
    mlir::OpBuilder::InsertionGuard insertion_guard(builder_);
    builder_.setInsertionPointToEnd(lir_block);

    mlir::Block &hir_block = hir_region->front();

    // Depending on what the terminator (if any) is, we know what ops to
    // traverse, and the result values.
    std::vector<mlir::Value> hir_results;
    if (!hir_block.empty()) {
      mlir::Operation *last_op = &hir_block.back();
      llvm::TypeSwitch<mlir::Operation *, void>(last_op)
          .Case([&](JsirExprRegionEndOp op) {
            // hir_region is an ExprRegion.

            // Lower all the ops except for the terminator.
            VisitOperations(hir_block.without_terminator());

            hir_results.push_back(op.getArgument());
          })
          .Case([&](JsirExprsRegionEndOp op) {
            // hir_region is an ExprsRegion.

            // Lower all the ops except for the terminator.
            VisitOperations(hir_block.without_terminator());

            for (mlir::Value hir_result : op.getArguments()) {
              hir_results.push_back(hir_result);
            }
          })
          .Default([&](mlir::Operation *op) {
            // hir_region is an StmtRegion or StmtsRegion.

            // Lower all the ops.
            VisitOperations(hir_block);
          });
    }

    std::vector<mlir::Value> lir_results;
    lir_results.reserve(hir_results.size());
    for (auto hir_result : hir_results) {
      lir_results.push_back(mapping_.lookup(hir_result));
    }

    return builder_.create<EndOp>(builder_.getUnknownLoc(), lir_results,
                                  lir_successors...);
  }

  // Converts all ops from hir_region to lir_region.
  void VisitRegion(mlir::Region *hir_region, mlir::Region *lir_region);

  void VisitBlockStatementOp(JshirBlockStatementOp hir_op);

  void VisitWithStatementOp(JshirWithStatementOp hir_op);

  void VisitLabeledStatementOp(JshirLabeledStatementOp hir_op);

  void VisitIfStatementOp(JshirIfStatementOp hir_op);

  struct CaseTest {
    mlir::Location loc;
    mlir::Block *hir_block;
    mlir::Block *lir_block;
    mlir::Block *lir_unmatch_target_block;
  };

  struct Case {
    size_t idx;
    mlir::Location loc;

    std::optional<CaseTest> test;

    mlir::Block *hir_body_block;
    mlir::Location body_loc;
    mlir::Block *lir_body_block;
    mlir::Block *lir_fall_through_block;

    mlir::Block *lir_first_block() const {
      if (test.has_value()) {
        return test->lir_block;
      } else {
        return lir_body_block;
      }
    }
  };

  mlir::Value VisitSwitchCaseTest(mlir::Value switch_token,
                                  mlir::Value lir_discriminant,
                                  mlir::IntegerAttr case_idx,
                                  const CaseTest &test,
                                  mlir::Block *lir_body_block);

  void VisitSwitchCaseOp(mlir::Value switch_token, mlir::Value lir_discriminant,
                         const Case &kase);

  void VisitSwitchStatementOp(JshirSwitchStatementOp hir_op);

  void VisitCatchClauseOp(JshirCatchClauseOp hir_op);

  void VisitTryStatementOp(JshirTryStatementOp hir_op);

  void VisitWhileStatementOp(JshirWhileStatementOp hir_op);

  void VisitDoWhileStatementOp(JshirDoWhileStatementOp hir_op);

  void VisitForStatementOp(JshirForStatementOp hir_op);

  void VisitForInOfStatementOp(mlir::Operation *hir_op, JsirForInOfKind kind,
                               JsirForInOfDeclarationAttr left_declaration,
                               mlir::Value hir_left, mlir::Value hir_right,
                               std::optional<bool> await,
                               mlir::Region &hir_body);

  void VisitForInStatementOp(JshirForInStatementOp hir_op);

  void VisitForOfStatementOp(JshirForOfStatementOp hir_op);

  void VisitLogicalExpressionOp(JshirLogicalExpressionOp hir_op);

  void VisitConditionalExpressionOp(JshirConditionalExpressionOp hir_op);

  void VisitBreakStatementOp(JshirBreakStatementOp hir_op);

  void VisitContinueStatementOp(JshirContinueStatementOp hir_op);

 private:
  void CreateCondBranch(mlir::Location loc, mlir::Value test,
                        mlir::Block *true_dest,
                        mlir::ValueRange true_dest_operands,
                        mlir::Block *false_dest,
                        mlir::ValueRange false_dest_operands);

  template <typename T, typename... Args>
  T CreateExpr(mlir::Operation *hir_op, Args &&...args) {
    CHECK(hir_op != nullptr) << "hir_op cannot be null.";
    return builder_.create<T>(hir_op->getLoc(),
                              // TODO(tzx): Properly handle type.
                              JsirAnyType::get(builder_.getContext()),
                              std::forward<Args>(args)...);
  }

  template <typename T, typename... Args>
  T CreateStmt(mlir::Operation *hir_op, Args &&...args) {
    CHECK(hir_op != nullptr) << "hir_op cannot be null.";
    return builder_.create<T>(hir_op->getLoc(), std::nullopt,
                              std::forward<Args>(args)...);
  }

  mlir::Block *CreateBlockBefore(mlir::Block *current) {
    auto *new_block = new mlir::Block();
    current->getParent()->getBlocks().insert(mlir::Region::iterator(current),
                                             new_block);
    return new_block;
  }

  mlir::Block *CreateBlockAfter(mlir::Block *current) {
    return current->splitBlock(current->end());
  }

  // A mapping from JSHIR values to JSLIR values.
  //
  // We lower one op at a time. When lowering jshir.some_op (%arg1, %arg2) into
  // jslir.some_op (%arg1', %arg2'), we need to query the map so that from %arg1
  // and %arg2 we get the corresponding %arg1' and %arg2' (%arg1' and %arg2' are
  // created when we lower previous ops).
  //
  // mlir::IRMapping also supports mapping between mlir::Block's, but
  // we are not using that.
  mlir::IRMapping mapping_;

  // Keeps track of labels and jump targets.
  // This guides the creation of conditional/unconditional branches.
  JslirJumpEnv env_;

  mlir::OpBuilder &builder_;
};

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_CONVERSION_JSHIR_TO_JSLIR_H_
