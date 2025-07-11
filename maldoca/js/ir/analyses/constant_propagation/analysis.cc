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

#include "maldoca/js/ir/analyses/constant_propagation/analysis.h"

#include <cassert>
#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/ir/analyses/dataflow_analysis.h"
#include "maldoca/js/ir/analyses/scope.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {

// =============================================================================
// JsirConstantPropagationValue
// =============================================================================

mlir::ChangeResult JsirConstantPropagationValue::Join(
    const JsirConstantPropagationValue &other) {
  // Case 1: Join(Uninitialized, A) = A
  if (IsUninitialized()) {
    if (other.IsUninitialized()) {
      return mlir::ChangeResult::NoChange;
    } else {
      value_ = other.value_;
      return mlir::ChangeResult::Change;
    }
  }

  // Case 2: Join(A, Uninitialized) = A
  if (other.IsUninitialized()) {
    return mlir::ChangeResult::NoChange;
  }

  // Case 3: Join(A, B), when A != Uninitialized & B != Uninitialized.
  if (*value_ == *other.value_) {
    return mlir::ChangeResult::NoChange;
  }

  // Case 4. Join(Unknown, A) = Unknown
  if (IsUnknown()) {
    return mlir::ChangeResult::NoChange;
  }

  value_ = mlir::Attribute();
  return mlir::ChangeResult::Change;
}

void JsirConstantPropagationValue::print(llvm::raw_ostream &os) const {
  if (IsUninitialized()) {
    os << "<uninitialized>";
  } else if (IsUnknown()) {
    os << "<unknown>";
  } else {
    value_->print(os);
  }
}

// =============================================================================
// JsirConstantPropagationAnalysis
// =============================================================================

void JsirConstantPropagationAnalysis::VisitOpCommon(
    mlir::Operation *op,
    llvm::ArrayRef<const JsirConstantPropagationValue *> operands,
    const JsirConstantPropagationState *before,
    llvm::MutableArrayRef<JsirStateRef<JsirConstantPropagationValue>> results,
    JsirStateRef<JsirConstantPropagationState> after) {
  llvm::TypeSwitch<mlir::Operation *, void>(op)
      .Case([&](JsirAssignmentExpressionOp op) {
        VisitAssignmentExpression(op, operands, before, results, after);
      })
      .Case([&](JsirUpdateExpressionOp op) {
        VisitUpdateExpression(op, operands, before, results, after);
      })
      .Case([&](JsirIdentifierOp op) {
        assert(results.size() == 1);
        auto &result = results[0];
        VisitIdentifier(op, operands, before, result);
        after.Join(*before);
      })
      .Case([&](JsirVariableDeclaratorOp op) {
        VisitVariableDeclarator(op, operands, before, results, after);
      })
      .Case([&](JsirVariableDeclarationOp op) {
        VisitVariableDeclaration(op, operands, before, results, after);
      })
      .Case([&](mlir::UnrealizedConversionCastOp op) {
        for (auto [operand, result] : llvm::zip_equal(operands, results)) {
          result.Join(*operand);
        }
        after.Join(*before);
      })
      .Default([&](mlir::Operation *op) {
        VisitOpDefault(op, operands, before, results, after);
      });
}

void JsirConstantPropagationAnalysis::VisitOpDefault(
    mlir::Operation *op,
    llvm::ArrayRef<const JsirConstantPropagationValue *> operands,
    const JsirConstantPropagationState *before,
    llvm::MutableArrayRef<JsirStateRef<JsirConstantPropagationValue>> results,
    JsirStateRef<JsirConstantPropagationState> after) {
  after.Join(*before);

  // Don't try to simulate the results of a region operation as we can't
  // guarantee that folding will be out-of-place. We don't allow in-place
  // folds as the desire here is for simulated execution, and not general
  // folding.
  if (op->getNumRegions()) {
    for (auto &result : results) {
      result.Write(JsirConstantPropagationValue::Unknown());
    }
    return;
  }

  llvm::SmallVector<mlir::Attribute, 8> operand_attributes;
  operand_attributes.reserve(op->getNumOperands());
  for (auto *operand : operands) {
    if (operand->IsUninitialized()) {
      // If any operand is uninitialized, bail out.
      return;
    }
    operand_attributes.push_back(***operand);
  }

  // Save the original operands and attributes just in case the operation
  // folds in-place. The constant passed in may not correspond to the real
  // runtime value, so in-place updates are not allowed.
  llvm::SmallVector<mlir::Value, 8> original_operands(op->getOperands());
  mlir::DictionaryAttr original_attrs = op->getAttrDictionary();

  // Simulate the result of folding this operation to a constant. If folding
  // fails or was an in-place fold, mark the results as overdefined.
  llvm::SmallVector<mlir::OpFoldResult, 8> fold_results;
  fold_results.reserve(op->getNumResults());
  if (mlir::failed(op->fold(operand_attributes, fold_results))) {
    for (auto &result : results) {
      result.Join(JsirConstantPropagationValue::Unknown());
    }
    return;
  }

  // If the folding was in-place, mark the results as overdefined and reset
  // the operation. We don't allow in-place folds as the desire here is for
  // simulated execution, and not general folding.
  if (fold_results.empty()) {
    op->setOperands(original_operands);
    op->setAttrs(original_attrs);
    for (auto &result : results) {
      result.Join(JsirConstantPropagationValue::Unknown());
    }
    return;
  }

  // Merge the fold results into the lattice for this operation.
  assert(fold_results.size() == op->getNumResults() && "invalid result size");
  for (const auto it : llvm::zip(results, fold_results)) {
    JsirStateRef<JsirConstantPropagationValue> &result = std::get<0>(it);

    // Merge in the result of the fold, either a constant or a value.
    mlir::OpFoldResult fold_result = std::get<1>(it);
    if (auto attr = fold_result.dyn_cast<mlir::Attribute>()) {
      result.Join(JsirConstantPropagationValue{attr});
    } else {
      auto result_value = fold_result.get<mlir::Value>();
      auto result_state_ref = GetStateAt(result_value);
      result.Join(result_state_ref.value());
    }
  }
}

void JsirConstantPropagationAnalysis::VisitAssignmentExpression(
    JsirAssignmentExpressionOp op,
    OperandStates<JsirAssignmentExpressionOp> operands,
    const JsirConstantPropagationState *before,
    llvm::MutableArrayRef<JsirStateRef<JsirConstantPropagationValue>> results,
    JsirStateRef<JsirConstantPropagationState> after) {
  auto left = llvm::dyn_cast<JsirIdentifierRefOp>(op.getLeft().getDefiningOp());
  if (left == nullptr) {
    return after.Join(*before);
  }

  results[0].Join(JsirConstantPropagationValue::Unknown());
  const JsirConstantPropagationValue &right = *operands.getRight();
  WriteDenseAfterState(op, left.getName(), right, before, after);
}

void JsirConstantPropagationAnalysis::VisitUpdateExpression(
    JsirUpdateExpressionOp op, OperandStates<JsirUpdateExpressionOp> operands,
    const JsirConstantPropagationState *before,
    llvm::MutableArrayRef<JsirStateRef<JsirConstantPropagationValue>> results,
    JsirStateRef<JsirConstantPropagationState> after) {
  auto id =
      llvm::dyn_cast<JsirIdentifierRefOp>(op.getArgument().getDefiningOp());
  if (id == nullptr) {
    return after.Join(*before);
  }

  JsSymbolId symbol_id = GetSymbolId(scopes_, op, id.getName());
  const auto &value_before = before->Get(symbol_id);
  if (value_before.IsUninitialized() || value_before.IsUnknown()) {
    results[0].Join(value_before);
    return after.Join(*before);
  }

  mlir::MLIRContext *context = op.getContext();
  mlir::OpBuilder builder(context);

  auto one = JsirNumericLiteralAttr::get(context, /*loc=*/nullptr,
                                         builder.getF64FloatAttr(1.0),
                                         /*extra=*/nullptr);
  std::optional<mlir::Attribute> value_after;
  if (op.getOperator_() == "++") {
    value_after = EmulateBinOp("+", context, **value_before, one);
  } else if (op.getOperator_() == "--") {
    value_after = EmulateBinOp("-", context, **value_before, one);
  }

  if (!value_after.has_value()) {
    results[0].Join(JsirConstantPropagationValue::Unknown());
    return after.Join(*before);
  }

  if (op.getPrefix()) {
    // ++a
    results[0].Join(JsirConstantPropagationValue{*value_after});
  } else {
    // a++
    results[0].Join(value_before);
  }

  return WriteDenseAfterState(op, id.getName(),
                              JsirConstantPropagationValue{*value_after},
                              before, after);
}

bool JsirConstantPropagationAnalysis::IsCfgEdgeExecutable(
    JsirGeneralCfgEdge *edge) {
  if (!edge->getLivenessInfo().has_value()) {
    return true;
  }

  auto [liveness_source, liveness_kind] = edge->getLivenessInfo().value();
  auto liveness_source_state_ref = GetStateAt(liveness_source);
  if (liveness_source_state_ref.value().IsUninitialized()) {
    return false;
  } else if (liveness_source_state_ref.value().IsUnknown()) {
    return true;
  }

  mlir::Attribute true_attr =
      mlir::BoolAttr::get(liveness_source.getContext(), true);
  mlir::Attribute false_attr =
      mlir::BoolAttr::get(liveness_source.getContext(), false);
  mlir::Attribute null_attr =
      JsirNullLiteralAttr::get(liveness_source.getContext());

  switch (liveness_kind) {
    case LivenessKind::kLiveIfTrueOrUnknown:
      return *liveness_source_state_ref.value() == true_attr;
    case LivenessKind::kLiveIfFalseOrUnknown:
      return *liveness_source_state_ref.value() == false_attr;
    case LivenessKind::kLiveIfNullOrUnknown:
      return *liveness_source_state_ref.value() == null_attr;
    case LivenessKind::kLiveIfNonNullOrUnknown:
      return *liveness_source_state_ref.value() != null_attr;
  }
}

void JsirConstantPropagationAnalysis::VisitOp(
    mlir::Operation *op,
    llvm::ArrayRef<const JsirConstantPropagationValue *> operands,
    const JsirConstantPropagationState *before,
    llvm::MutableArrayRef<JsirStateRef<JsirConstantPropagationValue>> results,
    JsirStateRef<JsirConstantPropagationState> after) {
  VisitOpCommon(op, operands, before, results, after);
}

}  // namespace maldoca
