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

#ifndef MALDOCA_JS_IR_ANALYSES_DYNAMIC_CONSTANT_PROPAGATION_ANALYSIS_H_
#define MALDOCA_JS_IR_ANALYSES_DYNAMIC_CONSTANT_PROPAGATION_ANALYSIS_H_

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "absl/container/flat_hash_map.h"
#include "llvm/Support/raw_ostream.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/driver/driver.pb.h"
#include "maldoca/js/ir/analyses/constant_propagation/analysis.h"
#include "maldoca/js/ir/analyses/dataflow_analysis.h"
#include "maldoca/js/ir/analyses/dynamic_constant_propagation/dynamic_prelude.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {

absl::flat_hash_map<JsSymbolId, mlir::Attribute> GetConstBindings(
    const BabelScopes &scopes, mlir::Operation *root);

void PrintBindings(
    llvm::raw_ostream &os,
    const absl::flat_hash_map<JsSymbolId, mlir::Attribute> &bindings);

// The dynamic constant propagation analysis extends the "normal" constant
// propagation, with additional logic of prelude function matching and
// execution.
class JsirDynamicConstantPropagationAnalysis
    : public JsirConstantPropagationAnalysis {
 public:
  explicit JsirDynamicConstantPropagationAnalysis(
      mlir::DataFlowSolver &solver, const BabelScopes *scopes,
      DynamicPrelude *dynamic_prelude,
      absl::flat_hash_map<JsSymbolId, mlir::Attribute> const_bindings)
      : JsirConstantPropagationAnalysis(solver, scopes),
        dynamic_prelude_(dynamic_prelude),
        const_bindings_(std::move(const_bindings)) {}

  void VisitOp(
      mlir::Operation *op,
      llvm::ArrayRef<const JsirConstantPropagationValue *> operands,
      const JsirConstantPropagationState *before,
      llvm::MutableArrayRef<JsirStateRef<JsirConstantPropagationValue>> results,
      JsirStateRef<JsirConstantPropagationState> after) override;

  void VisitIdentifier(JsirIdentifierOp op,
                       OperandStates<JsirIdentifierOp> operands,
                       const JsirConstantPropagationState *before,
                       JsirStateRef<JsirConstantPropagationValue> result);

  std::optional<mlir::Attribute> Eval(
      mlir::Attribute expr,
      const absl::flat_hash_map<JsSymbolId, mlir::Attribute> &bindings);

  std::optional<mlir::Attribute> EvalIdentifier(
      mlir::StringAttr name, std::optional<int64_t> def_scope_id,
      const absl::flat_hash_map<JsSymbolId, mlir::Attribute> &bindings);

  std::optional<mlir::Attribute> EvalCallExpression(
      mlir::Attribute callee, std::vector<mlir::Attribute> arguments);

  void VisitCallExpression(
      JsirCallExpressionOp op, OperandStates<JsirCallExpressionOp> operands,
      const JsirConstantPropagationState *before,
      llvm::MutableArrayRef<JsirStateRef<JsirConstantPropagationValue>> results,
      JsirStateRef<JsirConstantPropagationState> after);

  void VisitMemberExpression(
      JsirMemberExpressionOp op, OperandStates<JsirMemberExpressionOp> operands,
      const JsirConstantPropagationState *before,
      llvm::MutableArrayRef<JsirStateRef<JsirConstantPropagationValue>> results,
      JsirStateRef<JsirConstantPropagationState> after);

 private:
  DynamicPrelude *dynamic_prelude_;
  absl::flat_hash_map<JsSymbolId, mlir::Attribute> const_bindings_;
};

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_ANALYSES_DYNAMIC_CONSTANT_PROPAGATION_ANALYSIS_H_
