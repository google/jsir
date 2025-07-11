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

#include "maldoca/js/ir/conversion/jshir_to_jslir.h"

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "absl/log/check.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {

void JshirToJslir::VisitOperation(mlir::Operation *hir_op) {
  llvm::TypeSwitch<mlir::Operation *, void>(hir_op)
      .Case(
          [&](JshirBlockStatementOp hir_op) { VisitBlockStatementOp(hir_op); })
      .Case([&](JshirWithStatementOp hir_op) { VisitWithStatementOp(hir_op); })
      .Case([&](JshirLabeledStatementOp hir_op) {
        VisitLabeledStatementOp(hir_op);
      })
      .Case([&](JshirIfStatementOp hir_op) { VisitIfStatementOp(hir_op); })
      .Case([&](JshirSwitchStatementOp hir_op) {
        VisitSwitchStatementOp(hir_op);
      })
      .Case([&](JshirCatchClauseOp hir_op) { VisitCatchClauseOp(hir_op); })
      .Case([&](JshirTryStatementOp hir_op) { VisitTryStatementOp(hir_op); })
      .Case(
          [&](JshirWhileStatementOp hir_op) { VisitWhileStatementOp(hir_op); })
      .Case([&](JshirDoWhileStatementOp hir_op) {
        VisitDoWhileStatementOp(hir_op);
      })
      .Case([&](JshirForStatementOp hir_op) { VisitForStatementOp(hir_op); })
      .Case(
          [&](JshirForInStatementOp hir_op) { VisitForInStatementOp(hir_op); })
      .Case(
          [&](JshirForOfStatementOp hir_op) { VisitForOfStatementOp(hir_op); })
      .Case([&](JshirLogicalExpressionOp hir_op) {
        VisitLogicalExpressionOp(hir_op);
      })
      .Case([&](JshirConditionalExpressionOp hir_op) {
        VisitConditionalExpressionOp(hir_op);
      })
      .Case(
          [&](JshirBreakStatementOp hir_op) { VisitBreakStatementOp(hir_op); })
      .Case([&](JshirContinueStatementOp hir_op) {
        VisitContinueStatementOp(hir_op);
      })
      .Default([&](mlir::Operation *hir_op) {
        auto *lir_op = builder_.cloneWithoutRegions(*hir_op, mapping_);
        for (unsigned int i = 0, e = hir_op->getNumRegions(); i != e; ++i) {
          auto with_env_scope = env_.WithScope();
          auto &lir_region = lir_op->getRegion(i);
          VisitRegion(&hir_op->getRegion(i), &lir_region);
        }
      });
}

void JshirToJslir::VisitOperations(
    llvm::iterator_range<llvm::iplist<mlir::Operation>::iterator> hir_ops) {
  for (auto &hir_op : hir_ops) {
    VisitOperation(&hir_op);
  }
}

void JshirToJslir::VisitBlock(mlir::Block *hir_block) {
  VisitOperations(*hir_block);
}

void JshirToJslir::VisitRegion(mlir::Region *hir_region,
                               mlir::Region *lir_region) {
  // Save insertion point.
  // Will revert at the end.
  mlir::OpBuilder::InsertionGuard insertion_guard(builder_);

  for (mlir::Block &hir_block : *hir_region) {
    mlir::Block &lir_block = lir_region->emplaceBlock();
    builder_.setInsertionPointToStart(&lir_block);

    for (auto &hir_op : hir_block) {
      VisitOperation(&hir_op);
    }
  }
}

void JshirToJslir::VisitBlockStatementOp(JshirBlockStatementOp hir_op) {
  mlir::Value token = CreateExpr<JslirControlFlowStarterOp>(
      hir_op, JsirControlFlowKind::BlockStatement);

  // The JshirBlockStatementOp can be lowered without introducing any new block,
  // if the block does not contain labeled breaks:
  //
  // ```
  //   %token = jslir.control_flow_starter {BlockStatement}
  //   jslir.control_flow_marker {BlockStatementDirectives} (%token)
  //   ...
  //   jslir.control_flow_marker {BlockStatementBody} (%token)
  //   ...
  //   jslir.control_flow_marker {BlockStatementEnd} (%token)
  // ```
  //
  // However, a labeled break would jump to the BlockStatementEnd marker, which
  // requires it to be the start of a block:
  //
  // ```
  //   jslir.control_flow_starter {BlockStatement}
  //   jslir.control_flow_marker {BlockStatementDirectives}
  //   ...
  //   jslir.control_flow_marker {BlockStatementBody}
  //   ...
  //   cf.br [^end_bb]
  //
  // ^end_bb:  <- A labeled break jumps here.
  //   jslir.control_flow_marker {BlockStatementEnd}
  // ```

  auto generate_directives_and_body = [&] {
    {
      // Directives
      CreateStmt<JslirControlFlowMarkerOp>(
          hir_op, JsirControlFlowMarkerKind::BlockStatementDirectives, token);

      VisitOperations(hir_op.getDirectives().front());
    }

    {
      // Body
      CreateStmt<JslirControlFlowMarkerOp>(
          hir_op, JsirControlFlowMarkerKind::BlockStatementBody, token);

      VisitOperations(hir_op.getBody().front());
    }
  };

  if (!env_.unmatched_labels().empty()) {
    mlir::Block *lir_end_block = CreateBlockAfter(builder_.getBlock());

    auto with_break_continue_info = env_.WithJumpTargets({
        .labeled_break_target = lir_end_block,
        .unlabeled_break_target = std::nullopt,
        .continue_target = std::nullopt,
    });

    generate_directives_and_body();

    CreateStmt<mlir::cf::BranchOp>(hir_op, lir_end_block);
    builder_.setInsertionPointToStart(lir_end_block);

  } else {
    generate_directives_and_body();
  }

  CreateStmt<JslirControlFlowMarkerOp>(
      hir_op, JsirControlFlowMarkerKind::BlockStatementEnd, token);
}

void JshirToJslir::VisitWithStatementOp(JshirWithStatementOp hir_op) {
  mlir::Block *lir_current_block = builder_.getBlock();
  mlir::Block *lir_body_block = CreateBlockAfter(lir_current_block);
  mlir::Block *lir_end_block = CreateBlockAfter(lir_body_block);

  auto with_break_continue_info = env_.WithJumpTargets({
      .labeled_break_target = lir_end_block,
      .unlabeled_break_target = std::nullopt,
      .continue_target = std::nullopt,
  });

  mlir::Value lir_object = mapping_.lookup(hir_op.getObject());
  mlir::Value token = CreateExpr<JslirWithStatementStartOp>(hir_op, lir_object);
  CreateStmt<mlir::cf::BranchOp>(hir_op, lir_body_block);

  VisitStmtOrStmtsRegionWithMarkerOp(
      JsirControlFlowMarkerKind::WithStatementBody, token, &hir_op.getBody(),
      lir_body_block, lir_end_block);

  builder_.setInsertionPointToStart(lir_end_block);
  CreateStmt<JslirControlFlowMarkerOp>(
      hir_op, JsirControlFlowMarkerKind::WithStatementEnd, token);
}

void JshirToJslir::VisitLabeledStatementOp(JshirLabeledStatementOp hir_op) {
  auto with_label = env_.WithLabel(hir_op.getLabel().getName());

  mlir::Value token =
      CreateExpr<JslirLabeledStatementStartOp>(hir_op, hir_op.getLabel());

  VisitOperations(hir_op.getBody().front());

  CreateStmt<JslirControlFlowMarkerOp>(
      hir_op, JsirControlFlowMarkerKind::LabeledStatementEnd, token);
}

void JshirToJslir::VisitIfStatementOp(JshirIfStatementOp hir_op) {
  if (hir_op.getConsequent().empty()) {
    hir_op->emitOpError("consequent region is empty");
    return;
  }

  mlir::Block *lir_current_block = builder_.getBlock();
  mlir::Block *lir_consequent_block = CreateBlockAfter(lir_current_block);
  mlir::Block *lir_end_block = CreateBlockAfter(lir_consequent_block);
  mlir::Block *lir_alternate_block = nullptr;
  if (!hir_op.getAlternate().empty()) {
    lir_alternate_block = CreateBlockAfter(lir_consequent_block);
  }

  auto with_break_continue_info = env_.WithJumpTargets({
      .labeled_break_target = lir_end_block,
      .unlabeled_break_target = std::nullopt,
      .continue_target = std::nullopt,
  });

  mlir::Value token = CreateExpr<JslirControlFlowStarterOp>(
      hir_op, JsirControlFlowKind::IfStatement);

  mlir::Value lir_test = mapping_.lookup(hir_op.getTest());
  if (lir_alternate_block != nullptr) {
    CreateCondBranch(hir_op->getLoc(), lir_test,
                     /*true_dest=*/lir_consequent_block, {},
                     /*false_dest=*/lir_alternate_block, {});
  } else {
    CreateCondBranch(hir_op->getLoc(), lir_test,
                     /*true_dest=*/lir_consequent_block, {},
                     /*false_dest=*/lir_end_block, {});
  }

  VisitStmtOrStmtsRegionWithMarkerOp(
      JsirControlFlowMarkerKind::IfStatementConsequent, token,
      &hir_op.getConsequent(), lir_consequent_block, lir_end_block);

  if (lir_alternate_block != nullptr) {
    builder_.setInsertionPointToStart(lir_alternate_block);
    CreateStmt<JslirControlFlowMarkerOp>(
        hir_op, JsirControlFlowMarkerKind::IfStatementAlternate, token);
    VisitOptionalStmtOrStmtsRegion<mlir::cf::BranchOp>(
        &hir_op.getAlternate(), lir_alternate_block, lir_end_block);
  }

  builder_.setInsertionPointToStart(lir_end_block);
  CreateStmt<JslirControlFlowMarkerOp>(
      hir_op, JsirControlFlowMarkerKind::IfStatementEnd, token);
}

mlir::Value JshirToJslir::VisitSwitchCaseTest(mlir::Value switch_token,
                                              mlir::Value lir_discriminant,
                                              mlir::IntegerAttr case_idx,
                                              const CaseTest &test,
                                              mlir::Block *lir_body_block) {
  mlir::OpBuilder::InsertionGuard insertion_guard(builder_);
  builder_.setInsertionPointToStart(test.lir_block);

  mlir::Value case_token = JslirSwitchStatementCaseStartOp::create(
      builder_, test.loc, switch_token, case_idx);

  // Lower all the ops except for the terminator.
  VisitOperations(test.hir_block->without_terminator());

  // Get the `test` value.
  auto *hir_test_terminator = test.hir_block->getTerminator();
  auto hir_test_end_op = llvm::cast<JsirExprRegionEndOp>(hir_test_terminator);
  mlir::Value lir_test = mapping_.lookup(hir_test_end_op.getArgument());

  // An op that marks the `lir_test` value.
  JslirSwitchStatementCaseTestOp::create(builder_, test.loc, lir_test);

  // Perform the actual comparison.
  //
  // switch (<discriminant>) {
  //   case <test>:
  //     ...
  // }
  mlir::Value case_matched = JsirBinaryExpressionOp::create(
      builder_, test.loc, "===", lir_discriminant, lir_test);

  // Jump to the body on match, or a specified block on unmatch.
  CreateCondBranch(test.loc, case_matched, /*true_dest=*/lir_body_block, {},
                   /*false_dest=*/test.lir_unmatch_target_block, {});

  return case_token;
}

void JshirToJslir::VisitSwitchCaseOp(mlir::Value switch_token,
                                     mlir::Value lir_discriminant,
                                     const Case &kase) {
  mlir::Value case_token;
  if (kase.test.has_value()) {
    // Test
    mlir::IntegerAttr case_idx = builder_.getUI32IntegerAttr(kase.idx);
    case_token = VisitSwitchCaseTest(switch_token, lir_discriminant, case_idx,
                                     *kase.test, kase.lir_body_block);
  }

  {
    // Body

    builder_.setInsertionPointToStart(kase.lir_body_block);

    if (kase.test.has_value()) {
      JslirControlFlowMarkerOp::create(
          builder_, kase.body_loc,
          JsirControlFlowMarkerKind::SwitchStatementCaseBody, case_token);
    } else {
      JslirSwitchStatementDefaultStartOp::create(builder_, kase.loc,
                                                 switch_token, kase.idx);
    }

    VisitOperations(*kase.hir_body_block);

    // Jump to next body block.
    mlir::cf::BranchOp::create(builder_, kase.body_loc,
                               kase.lir_fall_through_block);
  }
}

void JshirToJslir::VisitSwitchStatementOp(JshirSwitchStatementOp hir_op) {
  mlir::Block *lir_current_block = builder_.getBlock();
  auto &hir_switch_case_ops = hir_op.getCases().front().getOperations();
  size_t num_cases = hir_switch_case_ops.size();

  // Verify HIR and collect information of each case.
  std::vector<Case> cases;
  for (auto [case_idx, hir_op] : llvm::enumerate(hir_switch_case_ops)) {
    auto hir_switch_case_op = llvm::dyn_cast<JshirSwitchCaseOp>(hir_op);
    if (hir_switch_case_op == nullptr) {
      hir_op.emitError("cases must all be `JshirSwitchCaseOp`s");
      return;
    }

    std::optional<CaseTest> test;
    mlir::Region &hir_test_region = hir_switch_case_op.getTest();
    if (!hir_test_region.empty()) {
      test = CaseTest{
          .loc = hir_test_region.getLoc(),
          .hir_block = &hir_test_region.front(),
          .lir_block = nullptr,                 // To be created later.
          .lir_unmatch_target_block = nullptr,  // To be computed later.
      };
    }

    mlir::Region &hir_body_region = hir_switch_case_op.getConsequent();
    mlir::Block *hir_body_block = &hir_body_region.front();

    cases.push_back(Case{
        .idx = case_idx,
        .loc = hir_switch_case_op.getLoc(),

        .test = std::move(test),

        .hir_body_block = hir_body_block,
        .body_loc = hir_body_region.getLoc(),
        .lir_body_block = nullptr,          // To be created later.
        .lir_fall_through_block = nullptr,  // To be computed later.
    });
  }

  // Find the index of the default case.
  std::optional<size_t> default_case_idx;
  for (const auto &kase : cases) {
    if (kase.test.has_value()) {
      continue;
    }

    if (default_case_idx.has_value()) {
      hir_op.emitError("contains more than 1 default case");
      return;
    }

    default_case_idx = kase.idx;
  }

  // Create LIR blocks in the original order of cases in HIR.
  // This step computes CaseTest::lir_block and Case::lir_body_block.
  mlir::Block *lir_end_block = CreateBlockAfter(lir_current_block);
  for (auto &kase : cases) {
    if (kase.test.has_value()) {
      kase.test->lir_block = CreateBlockBefore(lir_end_block);
    }
    kase.lir_body_block = CreateBlockBefore(lir_end_block);
  }

  // test_idx_to_case_idx[test_idx] == case_idx
  //
  // In a switch-statement, the order of test matches and the order of body
  // fallthroughs are different.
  //
  // switch (<condition>) {
  //   case <test_a>:
  //     <body_a>;
  //   default:
  //     <body_default>;
  //   case <test_b>:
  //     <body_b>;
  // }
  //
  // Test match order:
  //   test_a -> test_b -> default
  // Body fallthrough order:
  //   body_a -> body_default -> body_b
  //
  // We use `test_idx` to indicate the order of test matches, and `case_idx` to
  // indicate body fallthrough order (i.e. original order).
  //
  // The only difference is that in test match order, default comes at last.
  std::vector<size_t> test_idx_to_case_idx;
  test_idx_to_case_idx.reserve(cases.size());
  for (size_t case_idx = 0; case_idx < num_cases; ++case_idx) {
    if (default_case_idx != case_idx) {
      test_idx_to_case_idx.push_back(case_idx);
    }
  }
  if (default_case_idx.has_value()) {
    test_idx_to_case_idx.push_back(*default_case_idx);
  }

  auto with_break_continue_info = env_.WithJumpTargets({
      .labeled_break_target = lir_end_block,
      .unlabeled_break_target = lir_end_block,
      .continue_target = std::nullopt,
  });

  // Set up the switch statement start and branch into first test
  mlir::Value lir_discriminant = mapping_.lookup(hir_op.getDiscriminant());
  mlir::Value token =
      CreateExpr<JslirSwitchStatementStartOp>(hir_op, lir_discriminant);
  if (num_cases == 0) {
    CreateStmt<mlir::cf::BranchOp>(hir_op, lir_end_block);
  } else {
    CreateStmt<mlir::cf::BranchOp>(hir_op, cases[0].lir_first_block());
  }

  // Generate code for each switch case.
  for (size_t test_idx = 0; test_idx < num_cases; ++test_idx) {
    const size_t case_idx = test_idx_to_case_idx[test_idx];
    Case &kase = cases[case_idx];

    if (kase.test.has_value()) {
      // If the case does not match, which block should we jump to?
      kase.test->lir_unmatch_target_block = [&] {
        if (test_idx + 1 == cases.size()) {
          return lir_end_block;
        } else {
          size_t next_case_idx = test_idx_to_case_idx[test_idx + 1];
          return cases[next_case_idx].lir_first_block();
        }
      }();
    }

    // We always fall through to the body of the next case, regardless of the
    // location of the default case.
    kase.lir_fall_through_block = [&] {
      if (case_idx + 1 == num_cases) {
        return lir_end_block;
      } else {
        return cases[case_idx + 1].lir_body_block;
      }
    }();

    VisitSwitchCaseOp(token, lir_discriminant, kase);
  }

  // mark the end block with a nullop
  builder_.setInsertionPointToStart(lir_end_block);
  CreateStmt<JslirControlFlowMarkerOp>(
      hir_op, JsirControlFlowMarkerKind::SwitchStatementEnd, token);
}

void JshirToJslir::VisitCatchClauseOp(JshirCatchClauseOp hir_op) {
  mlir::Value lir_param = nullptr;
  if (hir_op.getParam() != nullptr) {
    lir_param = mapping_.lookup(hir_op.getParam());
  }

  CreateStmt<JslirCatchClauseStartOp>(hir_op, lir_param);
  CHECK(hir_op.getBody().hasOneBlock());
  mlir::Block &hir_body_block = hir_op.getBody().front();
  VisitOperations(hir_body_block);
}

void JshirToJslir::VisitTryStatementOp(JshirTryStatementOp hir_op) {
  mlir::Block *lir_current_block = builder_.getBlock();
  mlir::Block *lir_end_block = CreateBlockAfter(lir_current_block);

  mlir::Block *lir_body_block = CreateBlockBefore(lir_end_block);

  mlir::Block *lir_handler_block = nullptr;
  if (!hir_op.getHandler().empty()) {
    lir_handler_block = CreateBlockBefore(lir_end_block);
  }

  mlir::Block *lir_finalizer_block = nullptr;
  if (!hir_op.getFinalizer().empty()) {
    lir_finalizer_block = CreateBlockBefore(lir_end_block);
  }

  mlir::Value token = CreateExpr<JslirControlFlowStarterOp>(
      hir_op, JsirControlFlowKind::TryStatement);
  CreateStmt<mlir::cf::BranchOp>(hir_op, lir_body_block);

  mlir::Block *lir_body_successor =
      (lir_finalizer_block != nullptr) ? lir_finalizer_block : lir_end_block;
  VisitStmtOrStmtsRegionWithMarkerOp(
      JsirControlFlowMarkerKind::TryStatementBody, token, &hir_op.getBlock(),
      lir_body_block, /*lir_successor=*/lir_body_successor);

  if (lir_handler_block != nullptr) {
    VisitStmtOrStmtsRegionWithMarkerOp(
        JsirControlFlowMarkerKind::TryStatementHandler, token,
        &hir_op.getHandler(), lir_handler_block,
        /*lir_successor=*/lir_body_successor);
  }

  if (lir_finalizer_block != nullptr) {
    VisitStmtOrStmtsRegionWithMarkerOp(
        JsirControlFlowMarkerKind::TryStatementFinalizer, token,
        &hir_op.getFinalizer(), lir_finalizer_block,
        /*lir_successor=*/lir_end_block);
  }

  builder_.setInsertionPointToStart(lir_end_block);
  CreateStmt<JslirControlFlowMarkerOp>(
      hir_op, JsirControlFlowMarkerKind::TryStatementEnd, token);
}

void JshirToJslir::VisitWhileStatementOp(JshirWhileStatementOp hir_op) {
  mlir::Block *lir_current_block = builder_.getBlock();
  mlir::Block *lir_test_block = CreateBlockAfter(lir_current_block);
  mlir::Block *lir_body_block = CreateBlockAfter(lir_test_block);
  mlir::Block *lir_end_block = CreateBlockAfter(lir_body_block);

  auto with_break_continue_info = env_.WithJumpTargets({
      .labeled_break_target = lir_end_block,
      .unlabeled_break_target = lir_end_block,
      .continue_target = lir_test_block,
  });

  mlir::Value token = CreateExpr<JslirControlFlowStarterOp>(
      hir_op, JsirControlFlowKind::WhileStatement);
  CreateStmt<mlir::cf::BranchOp>(hir_op, lir_test_block);

  VisitExprRegionWithMarkerOp(JsirControlFlowMarkerKind::WhileStatementTest,
                              token, &hir_op.getTest(), lir_test_block,
                              lir_body_block, lir_end_block);

  VisitStmtOrStmtsRegionWithMarkerOp(
      JsirControlFlowMarkerKind::WhileStatementBody, token, &hir_op.getBody(),
      lir_body_block, lir_test_block);

  builder_.setInsertionPointToStart(lir_end_block);
  CreateStmt<JslirControlFlowMarkerOp>(
      hir_op, JsirControlFlowMarkerKind::WhileStatementEnd, token);
}

void JshirToJslir::VisitDoWhileStatementOp(JshirDoWhileStatementOp hir_op) {
  mlir::Block *lir_current_block = builder_.getBlock();
  mlir::Block *lir_body_block = CreateBlockAfter(lir_current_block);
  mlir::Block *lir_test_block = CreateBlockAfter(lir_body_block);
  mlir::Block *lir_end_block = CreateBlockAfter(lir_test_block);

  auto with_break_continue_info = env_.WithJumpTargets({
      .labeled_break_target = lir_end_block,
      .unlabeled_break_target = lir_end_block,
      .continue_target = lir_test_block,
  });

  mlir::Value token = CreateExpr<JslirControlFlowStarterOp>(
      hir_op, JsirControlFlowKind::DoWhileStatement);
  CreateStmt<mlir::cf::BranchOp>(hir_op, lir_body_block);

  VisitStmtOrStmtsRegionWithMarkerOp(
      JsirControlFlowMarkerKind::DoWhileStatementBody, token, &hir_op.getBody(),
      lir_body_block, lir_test_block);

  VisitExprRegionWithMarkerOp(JsirControlFlowMarkerKind::DoWhileStatementTest,
                              token, &hir_op.getTest(), lir_test_block,
                              lir_body_block, lir_end_block);

  builder_.setInsertionPointToStart(lir_end_block);
  CreateStmt<JslirControlFlowMarkerOp>(
      hir_op, JsirControlFlowMarkerKind::DoWhileStatementEnd, token);
}

void JshirToJslir::VisitForStatementOp(JshirForStatementOp hir_op) {
  // A for-statement has the following structure. "?" means that this block is
  // optional and might not exist.
  //           ...
  //            |
  //     lir_current_block
  //            |
  //            v
  //      lir_init_block?
  //            |
  //            v
  // +--- lir_test_block? <--+
  // |          |            |
  // |          v            |
  // |    lir_body_block     |
  // |          |            |
  // |          v            |
  // |   lir_update_block? --+
  // |
  // +--> lir_end_block
  //            |
  //           ...
  //
  // Specifically, if `lir_test_block` doesn't exist:
  //           ...
  //            |
  //     lir_current_block
  //            |
  //            v
  //      lir_init_block?
  //            |
  //            v
  //      lir_body_block <---+
  //            |            |
  //            v            |
  //     lir_update_block? --+
  //
  //      lir_end_block  // no predecessor
  //            |
  //           ...

  mlir::Block *lir_current_block = builder_.getBlock();
  mlir::Block *lir_end_block = CreateBlockAfter(lir_current_block);

  mlir::Block *lir_init_block = nullptr;
  if (!hir_op.getInit().empty()) {
    lir_init_block = CreateBlockBefore(lir_end_block);
  }

  mlir::Block *lir_test_block = nullptr;
  if (!hir_op.getTest().empty()) {
    lir_test_block = CreateBlockBefore(lir_end_block);
  }

  mlir::Block *lir_body_block = CreateBlockBefore(lir_end_block);

  mlir::Block *lir_update_block = nullptr;
  if (!hir_op.getUpdate().empty()) {
    lir_update_block = CreateBlockBefore(lir_end_block);
  }

  // The target of `continue;` is the successor of `lir_body_block`.
  mlir::Block *lir_continue_target =
      lir_update_block ? lir_update_block
                       : (lir_test_block ? lir_test_block : lir_body_block);

  auto with_break_continue_info = env_.WithJumpTargets({
      .labeled_break_target = lir_end_block,
      .unlabeled_break_target = lir_end_block,
      .continue_target = lir_continue_target,
  });

  mlir::Value token = CreateExpr<JslirControlFlowStarterOp>(
      hir_op, JsirControlFlowKind::ForStatement);

  // Branch into the first block of the for-statement.
  mlir::Block *lir_first_block =
      lir_init_block ? lir_init_block
                     : (lir_test_block ? lir_test_block : lir_body_block);

  CreateStmt<mlir::cf::BranchOp>(hir_op, lir_first_block);

  if (lir_init_block != nullptr) {
    // Init

    builder_.setInsertionPointToEnd(lir_init_block);
    CreateStmt<JslirControlFlowMarkerOp>(
        hir_op, JsirControlFlowMarkerKind::ForStatementInit, token);

    mlir::Block *lir_branch_target =
        lir_test_block ? lir_test_block : lir_body_block;

    VisitUnknownRegion<mlir::cf::BranchOp>(&hir_op.getInit(), lir_init_block,
                                           lir_branch_target);
  }

  if (lir_test_block != nullptr) {
    // Test

    VisitExprRegionWithMarkerOp(JsirControlFlowMarkerKind::ForStatementTest,
                                token, /*hir_region=*/&hir_op.getTest(),
                                /*lir_block=*/lir_test_block,
                                /*lir_successor_true=*/lir_body_block,
                                /*lir_successor_false=*/lir_end_block);
  }

  {
    // Body

    VisitStmtOrStmtsRegionWithMarkerOp(
        JsirControlFlowMarkerKind::ForStatementBody, token, &hir_op.getBody(),
        lir_body_block, lir_continue_target);
  }

  if (lir_update_block != nullptr) {
    // Update

    mlir::Block *lir_successor =
        lir_test_block ? lir_test_block : lir_body_block;

    VisitStmtOrStmtsRegionWithMarkerOp(
        JsirControlFlowMarkerKind::ForStatementUpdate, token,
        /*hir_region=*/&hir_op.getUpdate(), /*lir_block=*/lir_update_block,
        lir_successor);
  }

  builder_.setInsertionPointToStart(lir_end_block);
  CreateStmt<JslirControlFlowMarkerOp>(
      hir_op, JsirControlFlowMarkerKind::ForStatementEnd, token);
}

void JshirToJslir::VisitForInOfStatementOp(
    mlir::Operation *hir_op, JsirForInOfKind kind,
    JsirForInOfDeclarationAttr left_declaration, mlir::Value hir_left_lval,
    mlir::Value hir_right, std::optional<bool> await, mlir::Region &hir_body) {
  mlir::Block *lir_start_block = builder_.getBlock();
  mlir::Block *lir_next_block = CreateBlockAfter(lir_start_block);
  mlir::Block *lir_body_block = CreateBlockAfter(lir_next_block);
  mlir::Block *lir_end_block = CreateBlockAfter(lir_body_block);

  auto with_break_continue_info = env_.WithJumpTargets({
      .labeled_break_target = lir_end_block,
      .unlabeled_break_target = lir_end_block,
      .continue_target = lir_next_block,
  });

  mlir::Value lir_left_lval = mapping_.lookup(hir_left_lval);
  mlir::Value lir_right = mapping_.lookup(hir_right);

  mlir::Value iterator;
  switch (kind) {
    case JsirForInOfKind::ForIn: {
      iterator = CreateExpr<JslirForInStatementStartOp>(
          hir_op, left_declaration, lir_left_lval, lir_right);
      break;
    }
    case JsirForInOfKind::ForOf: {
      if (!await.has_value()) {
        hir_op->emitOpError("expected await to be defined, got nullopt");
        return;
      }

      iterator = CreateExpr<JslirForOfStatementStartOp>(
          hir_op, left_declaration, lir_left_lval, lir_right, await.value());
      break;
    }
  }

  mlir::cf::BranchOp::create(builder_, hir_op->getLoc(), lir_next_block);

  {
    builder_.setInsertionPointToStart(lir_body_block);
    CreateStmt<JslirForInOfStatementGetNextOp>(hir_op, iterator);

    VisitStmtOrStmtsRegion<mlir::cf::BranchOp>(&hir_body, lir_body_block,
                                               /*dest=*/lir_next_block);
  }

  {
    builder_.setInsertionPointToStart(lir_next_block);
    mlir::Value has_next =
        CreateExpr<JslirForInOfStatementHasNextOp>(hir_op, iterator);

    CreateCondBranch(hir_op->getLoc(), has_next, /*true_dest=*/lir_body_block,
                     {},
                     /*false_dest=*/lir_end_block, {});
  }

  builder_.setInsertionPointToStart(lir_end_block);
  CreateStmt<JslirForInOfStatementEndOp>(hir_op, iterator);
}

void JshirToJslir::VisitForInStatementOp(JshirForInStatementOp hir_op) {
  return VisitForInOfStatementOp(hir_op, JsirForInOfKind::ForIn,
                                 hir_op.getLeftDeclarationAttr(),
                                 hir_op.getLeftLval(), hir_op.getRight(),
                                 /*await=*/std::nullopt, hir_op.getBody());
}

void JshirToJslir::VisitForOfStatementOp(JshirForOfStatementOp hir_op) {
  return VisitForInOfStatementOp(hir_op, JsirForInOfKind::ForOf,
                                 hir_op.getLeftDeclarationAttr(),
                                 hir_op.getLeftLval(), hir_op.getRight(),
                                 hir_op.getAwait(), hir_op.getBody());
}

void JshirToJslir::VisitLogicalExpressionOp(JshirLogicalExpressionOp hir_op) {
  mlir::Block *lir_current_block = builder_.getBlock();
  mlir::Block *lir_right_block = CreateBlockAfter(lir_current_block);
  mlir::Block *lir_end_block = CreateBlockAfter(lir_right_block);

  mlir::Value lir_left = mapping_.lookup(hir_op.getLeft());

  mlir::Value token = CreateExpr<JslirLogicalExpressionStartOp>(
      hir_op, hir_op.getOperator_(), lir_left);

  struct LirBranchTarget {
    mlir::Block *dest;
    mlir::ValueRange operands;
  };
  struct CondBrArgs {
    mlir::Value cond;
    LirBranchTarget true_branch_target;
    LirBranchTarget false_branch_target;
  };

  auto logical_operator = StringToJsLogicalOperator(hir_op.getOperator_());
  if (!logical_operator.ok()) {
    hir_op->emitError("invalid logical operator: ")
        << logical_operator.status().ToString();
    return;
  }

  auto [cond, true_branch_target, false_branch_target] = [&]() -> CondBrArgs {
    switch (*logical_operator) {
      case JsLogicalOperator::kAnd: {
        // left && right => left ? right : left
        //
        //   %left = ...
        //   if %left goto [^right_bb] else goto [^end_bb(%left)]
        // ^right_bb:
        //   %right = ...
        //   goto [^end_bb(%right)]
        // ^end_bb(%result):
        //   ...

        return {
            .cond = lir_left,
            .true_branch_target = {.dest = lir_right_block, .operands = {}},
            .false_branch_target = {.dest = lir_end_block,
                                    .operands = lir_left},
        };
      }
      case JsLogicalOperator::kOr: {
        // left || right => left ? left : right
        //
        //   %left = ...
        //   if %left goto [^end_bb(%left)] else goto [^right_bb]
        // ^right_bb:
        //   %right = ...
        //   goto [^end_bb(%right)]
        // ^end_bb(%result):
        //   ...

        return {
            .cond = lir_left,
            .true_branch_target = {.dest = lir_end_block, .operands = lir_left},
            .false_branch_target = {.dest = lir_right_block, .operands = {}},
        };
      }
      case JsLogicalOperator::kNullishCoalesce: {
        // left ?? right => (left == null) ? right : left
        //
        //   %left = ...
        //   %left_is_null = (%left == null)
        //   if %left_is_null goto [^right_bb] else goto [^end_bb(%left)]
        // ^right_bb:
        //   %right = ...
        //   goto [^end_bb(%right)]
        // ^end_bb(%result):
        //   ...

        mlir::Value lir_null = CreateExpr<JsirNullLiteralOp>(hir_op);
        mlir::StringAttr mlir_equal_operator = builder_.getStringAttr(
            JsBinaryOperatorToString(JsBinaryOperator::kEqual));
        mlir::Value lir_left_is_null = CreateExpr<JsirBinaryExpressionOp>(
            hir_op, mlir_equal_operator, lir_left, lir_null);

        return {
            .cond = lir_left_is_null,
            .true_branch_target = {.dest = lir_right_block, .operands = {}},
            .false_branch_target = {.dest = lir_end_block,
                                    .operands = lir_left},
        };
      }
    }
  }();

  CreateCondBranch(hir_op->getLoc(), cond, true_branch_target.dest,
                   true_branch_target.operands, false_branch_target.dest,
                   false_branch_target.operands);

  builder_.setInsertionPointToStart(lir_right_block);
  CreateStmt<JslirControlFlowMarkerOp>(
      hir_op, JsirControlFlowMarkerKind::LogicalExpressionRight, token);
  VisitExprRegion<mlir::cf::BranchOp>(&hir_op.getRight(), lir_right_block,
                                      lir_end_block);

  builder_.setInsertionPointToStart(lir_end_block);
  mlir::Value lir_result = lir_end_block->addArgument(
      JsirAnyType::get(builder_.getContext()), builder_.getUnknownLoc());
  mapping_.map(hir_op, lir_result);
  CreateStmt<JslirControlFlowMarkerOp>(
      hir_op, JsirControlFlowMarkerKind::LogicalExpressionEnd, token);
}

void JshirToJslir::VisitConditionalExpressionOp(
    JshirConditionalExpressionOp hir_op) {
  mlir::Block *lir_current_block = builder_.getBlock();
  mlir::Block *lir_alternate_block = CreateBlockAfter(lir_current_block);
  mlir::Block *lir_consequent_block = CreateBlockAfter(lir_alternate_block);
  mlir::Block *lir_end_block = CreateBlockAfter(lir_consequent_block);

  mlir::Value lir_test = mapping_.lookup(hir_op.getTest());
  mlir::Value token = CreateExpr<JslirControlFlowStarterOp>(
      hir_op, JsirControlFlowKind::ConditionalExpression);
  CreateCondBranch(hir_op->getLoc(), lir_test,
                   /*true_dest=*/lir_consequent_block, {},
                   /*false_dest=*/lir_alternate_block, {});

  builder_.setInsertionPointToStart(lir_alternate_block);
  CreateStmt<JslirControlFlowMarkerOp>(
      hir_op, JsirControlFlowMarkerKind::ConditionalExpressionAlternate, token);
  VisitExprRegion<mlir::cf::BranchOp>(&hir_op.getAlternate(),
                                      lir_alternate_block, lir_end_block);

  builder_.setInsertionPointToStart(lir_consequent_block);
  CreateStmt<JslirControlFlowMarkerOp>(
      hir_op, JsirControlFlowMarkerKind::ConditionalExpressionConsequent,
      token);
  VisitExprRegion<mlir::cf::BranchOp>(&hir_op.getConsequent(),
                                      lir_consequent_block, lir_end_block);

  builder_.setInsertionPointToStart(lir_end_block);
  mlir::Value lir_result = lir_end_block->addArgument(
      JsirAnyType::get(builder_.getContext()), builder_.getUnknownLoc());
  mapping_.map(hir_op, lir_result);
  CreateStmt<JslirControlFlowMarkerOp>(
      hir_op, JsirControlFlowMarkerKind::ConditionalExpressionEnd, token);
}

void JshirToJslir::VisitContinueStatementOp(JshirContinueStatementOp hir_op) {
  JsirIdentifierAttr label = hir_op.getLabelAttr();  // May be nullptr.
  auto continue_target = (label == nullptr)
                             ? env_.continue_target()
                             : env_.continue_target(label.getName());
  if (!continue_target.ok()) {
    hir_op->emitError("unknown continue target, will not generate branch op: ")
        << continue_target.status().ToString();
    return;
  }

  CreateStmt<JslirContinueStatementOp>(hir_op, label);
  CreateStmt<mlir::cf::BranchOp>(hir_op, *continue_target);

  mlir::Block *dead_block_after_continue =
      CreateBlockAfter(builder_.getBlock());
  builder_.setInsertionPointToStart(dead_block_after_continue);
}

void JshirToJslir::VisitBreakStatementOp(JshirBreakStatementOp hir_op) {
  absl::StatusOr<mlir::Block *> break_target;

  JsirIdentifierAttr label = hir_op.getLabelAttr();  // May be nullptr.
  if (label == nullptr) {
    break_target = env_.break_target();
  } else {
    // ```
    // break <label>;
    // ```

    // Case 1: break immediately after label:
    // ```
    // label:
    //   break label;
    // ```
    // At this point, `label` is not associated with any control flow structure.
    if (env_.unmatched_labels().contains(label.getName())) {
      // In this case, the break is a nop.

      CreateStmt<JslirBreakStatementOp>(hir_op, label);
      return;
    }

    // Case 2: break is part of a control flow structure:
    // ```
    // label: while (...) {
    //   ...
    //   break;
    //   ...
    // }
    break_target = env_.break_target(label.getName());
  }

  if (!break_target.ok()) {
    hir_op->emitError("unknown break target, will not generate branch op: ")
        << break_target.status().ToString();
    return;
  }

  CreateStmt<JslirBreakStatementOp>(hir_op, label);
  CreateStmt<mlir::cf::BranchOp>(hir_op, *break_target);

  mlir::Block *dead_block_after_break = CreateBlockAfter(builder_.getBlock());
  builder_.setInsertionPointToStart(dead_block_after_break);
}

void JshirToJslir::CreateCondBranch(mlir::Location loc, mlir::Value test,
                                    mlir::Block *true_dest,
                                    mlir::ValueRange true_dest_operands,
                                    mlir::Block *false_dest,
                                    mlir::ValueRange false_dest_operands) {
  auto test_i1 = mlir::UnrealizedConversionCastOp::create(
      builder_, loc, mlir::TypeRange{builder_.getI1Type()},
      mlir::ValueRange{test});

  mlir::cf::CondBranchOp::create(builder_, loc, test_i1.getResult(0), true_dest,
                                 true_dest_operands, false_dest,
                                 false_dest_operands);
}

}  // namespace maldoca
