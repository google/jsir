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

#ifndef MALDOCA_JS_IR_ANALYSES_CONSTANT_PROPAGATION_ANALYSIS_H_
#define MALDOCA_JS_IR_ANALYSES_CONSTANT_PROPAGATION_ANALYSIS_H_

#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "maldoca/js/babel/babel.pb.h"
#include "maldoca/js/ir/analyses/conditional_forward_per_var_dataflow_analysis.h"
#include "maldoca/js/ir/analyses/dataflow_analysis.h"
#include "maldoca/js/ir/analyses/per_var_state.h"
#include "maldoca/js/ir/analyses/state.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {

// The lattice of each variable in constant propagation.
//
// Represents 3 cases:
// - Uninitialized
//   This is used when a variable has never been assigned to.
//
// - Some constant
//   Represented using an mlir::Attribute.
//
// - Unknown
//   This is used after we join two constants; or as the initial state of a
//   function parameter.
class JsirConstantPropagationValue
    : public JsirState<JsirConstantPropagationValue> {
 public:
  explicit JsirConstantPropagationValue() = default;

  explicit JsirConstantPropagationValue(std::optional<mlir::Attribute> value)
      : value_(value) {}

  // Joins this value with another, modifying the current value in-place.
  // Returns if the value has changed by the join.
  //
  // Rules:
  // - Join(Uninitialized, Anything) = Anything
  // - Join(Unknown, Anything) = Unknown
  // - Join(Const_1, Const_1) = Const_1
  // - Join(Const_1, Const_2) = Unknown
  mlir::ChangeResult Join(const JsirConstantPropagationValue &other) override;

  // Whether the value is uninitialized.
  // This is used when a variable has never been assigned to.
  bool IsUninitialized() const { return !value_.has_value(); }

  static JsirConstantPropagationValue Uninitialized() {
    return JsirConstantPropagationValue{std::nullopt};
  }

  // Whether the value is unknown.
  // This is used after we join two constants; or as the initial state of a
  // function parameter.
  bool IsUnknown() const { return value_.has_value() && *value_ == nullptr; }

  static JsirConstantPropagationValue Unknown() {
    return JsirConstantPropagationValue{mlir::Attribute(nullptr)};
  }

  // Returns the stored value.
  // - Uninitialized:
  //   std::nullopt
  // - SomeConstant:
  //   mlir::Attribute(SomeConstant)
  // - Unknown:
  //   mlir::Attribute(nullptr)
  const std::optional<mlir::Attribute> &operator*() const { return value_; }

  bool operator==(const JsirConstantPropagationValue &rhs) const override {
    return value_ == rhs.value_;
  }

  bool operator!=(const JsirConstantPropagationValue &rhs) const override {
    return !(operator==(rhs));
  }

  void print(llvm::raw_ostream &os) const override;

 private:
  std::optional<mlir::Attribute> value_;
};

using JsirConstantPropagationState =
    JsirPerVarState<JsirConstantPropagationValue>;

class JsirConstantPropagationAnalysis
    : public JsirConditionalForwardPerVarDataFlowAnalysis<
          JsirConstantPropagationValue> {
 public:
  using ValueT = JsirConstantPropagationValue;
  using Base = JsirConditionalForwardPerVarDataFlowAnalysis<ValueT>;

  explicit JsirConstantPropagationAnalysis(mlir::DataFlowSolver &solver,
                                           const BabelScopes *scopes)
      : Base(solver, scopes) {}

  JsirConstantPropagationValue BoundaryInitialValue() const override {
    return JsirConstantPropagationValue::Unknown();
  }

  void VisitOpCommon(
      mlir::Operation *op,
      llvm::ArrayRef<const JsirConstantPropagationValue *> operands,
      const JsirConstantPropagationState *before,
      llvm::MutableArrayRef<JsirStateRef<JsirConstantPropagationValue>> results,
      JsirStateRef<JsirConstantPropagationState> after);

  void VisitOpDefault(
      mlir::Operation *op,
      llvm::ArrayRef<const JsirConstantPropagationValue *> operands,
      const JsirConstantPropagationState *before,
      llvm::MutableArrayRef<JsirStateRef<JsirConstantPropagationValue>> results,
      JsirStateRef<JsirConstantPropagationState> after);

  // <left> = <right>
  //
  // If <right> is a constant `C`, and <left> is an identifier `id`, then in the
  // `after` state, set the record `id = C`.
  //
  // TODO(b/259309463) Only fold pure operation.
  //
  // Details:
  //
  //  The `<left> = <right>` expression returns the value of `<right>`, so the
  //  natural behavior here is to set `results[0]` to that value. However, if we
  //  do that, then in the constant propagation transform, we would optimize the
  //  assignment away.
  //
  //  For example, consider the following code:
  //
  //  ```
  //  a = 1 + 2;
  //  ```
  //
  //  - If we DO set `results[0]` to `3`, then the statement optimizes to:
  //
  //    ```
  //    3;
  //    ```
  //
  //    This is problematic, because the variable `a` might be referenced later.
  //
  //  - If we DON'T set `results[0]` to `3`, then the statement optimizes to:
  //
  //    ```
  //    a = 3;
  //    ```
  //
  //    This keeps the variable `a`, but causes another (slightly smaller)
  //    problem: for the following statement:
  //
  //    ```
  //    a = b = 1 + 2;
  //    ```
  //
  //    we can only know that `b = 3`, but we don't know what `a` is.
  //
  // To fix this issue, we need to:
  // - DO set `results[0]` to the constant value;
  // - BUT modify the constant propagation transform and don't fold assignments,
  //   because assignments are non-pure.
  void VisitAssignmentExpression(
      JsirAssignmentExpressionOp op,
      OperandStates<JsirAssignmentExpressionOp> operands,
      const JsirConstantPropagationState *before,
      llvm::MutableArrayRef<JsirStateRef<JsirConstantPropagationValue>> results,
      JsirStateRef<JsirConstantPropagationState> after);

  void VisitUpdateExpression(
      JsirUpdateExpressionOp op, OperandStates<JsirUpdateExpressionOp> operands,
      const JsirConstantPropagationState *before,
      llvm::MutableArrayRef<JsirStateRef<JsirConstantPropagationValue>> results,
      JsirStateRef<JsirConstantPropagationState> after);

  bool IsCfgEdgeExecutable(JsirGeneralCfgEdge *edge) override;

  void VisitOp(
      mlir::Operation *op,
      llvm::ArrayRef<const JsirConstantPropagationValue *> operands,
      const JsirConstantPropagationState *before,
      llvm::MutableArrayRef<JsirStateRef<JsirConstantPropagationValue>> results,
      JsirStateRef<JsirConstantPropagationState> after) override;
};

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_ANALYSES_CONSTANT_PROPAGATION_ANALYSIS_H_
