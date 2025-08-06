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

#include "maldoca/js/ir/transforms/dynamic_constant_propagation/pass.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llvm/Support/raw_ostream.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/FoldUtils.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "maldoca/js/babel/babel.h"
#include "maldoca/js/babel/scope.h"
#include "maldoca/js/ir/analyses/constant_propagation/analysis.h"
#include "maldoca/js/ir/analyses/dataflow_analysis.h"
#include "maldoca/js/ir/analyses/dynamic_constant_propagation/analysis.h"
#include "maldoca/js/ir/analyses/dynamic_constant_propagation/dynamic_prelude.h"
#include "maldoca/js/ir/ir.h"
#include "maldoca/js/ir/transforms/constant_propagation/pass.h"

namespace maldoca {

mlir::LogicalResult PerformDynamicConstantPropagation(
    mlir::Operation *op, const BabelScopes &scopes,
    const JsirAnalysisConfig::DynamicConstantPropagation &config, Babel &babel,
    JsirAnalysisResult::DynamicConstantPropagation
        *absl_nullable analysis_result) {
  absl::StatusOr<DynamicPrelude> dynamic_prelude =
      DynamicPrelude::Create(config, babel);
  if (!dynamic_prelude.ok()) {
    return mlir::emitError(op->getLoc()) << dynamic_prelude.status().message();
  }

  return PerformDynamicConstantPropagation(op, scopes, &*dynamic_prelude,
                                           analysis_result);
}

mlir::LogicalResult PerformDynamicConstantPropagation(
    mlir::Operation *op, const BabelScopes &scopes,
    DynamicPrelude *dynamic_prelude,
    JsirAnalysisResult::DynamicConstantPropagation
        *absl_nullable analysis_result) {
  using ComputedConstant =
      JsirAnalysisResult::DynamicConstantPropagation::ComputedConstant;

  mlir::DataFlowSolver solver;
  absl::flat_hash_map<JsSymbolId, mlir::Attribute> const_bindings =
      GetConstBindings(scopes, op);

  auto *analysis = solver.load<JsirDynamicConstantPropagationAnalysis>(
      &scopes, dynamic_prelude, const_bindings);

  mlir::LogicalResult result = solver.initializeAndRun(op);
  if (mlir::failed(result)) {
    return result;
  }

  std::vector<ComputedConstant> computed_constants;

  op->walk([&](mlir::Operation *op) {
    ComputedConstant computed_constant;

    // If `op` is not a literal but we recovered a constant value for it, log
    // it.
    if (llvm::isa_and_nonnull<JsirLiteralOpInterface>(op)) {
      return;
    }
    if (op->getNumResults() != 1) {
      return;
    }

    JsirStateRef<JsirConstantPropagationValue> cp_state_ref =
        analysis->GetStateAt(op->getResult(0));
    if (cp_state_ref == nullptr) {
      return;
    }

    const JsirConstantPropagationValue &cp_value = cp_state_ref.value();
    if (cp_value.IsUninitialized() || cp_value.IsUnknown()) {
      return;
    }

    mlir::Attribute cp_attr = **cp_value;
    llvm::TypeSwitch<mlir::Attribute, void>(cp_attr)
        .Case([&](mlir::BoolAttr value) {
          computed_constant.set_bool_value(value.getValue());
        })
        .Case([&](mlir::FloatAttr value) {
          computed_constant.set_number_value(value.getValueAsDouble());
        })
        .Case([&](mlir::StringAttr value) {
          computed_constant.set_string_value(value.getValue());
        })
        .Case([&](JsirBigIntLiteralAttr value) {
          computed_constant.set_big_int_value(value.getValue().str());
        })
        .Default([&](mlir::Attribute value) {});
    if (computed_constant.value_kind_case() ==
        ComputedConstant::VALUE_KIND_NOT_SET) {
      return;
    }

    auto trivia = llvm::dyn_cast_if_present<JsirTriviaAttr>(op->getLoc());
    if (trivia == nullptr) {
      return;
    }
    if (!trivia.getLoc().getStartIndex().has_value()) {
      return;
    }
    if (!trivia.getLoc().getEndIndex().has_value()) {
      return;
    }

    computed_constant.set_start_offset(*trivia.getLoc().getStartIndex());
    computed_constant.set_end_offset(*trivia.getLoc().getEndIndex());

    computed_constants.push_back(std::move(computed_constant));
  });

  if (analysis_result != nullptr) {
    std::string const_bindings_str;
    llvm::raw_string_ostream os(const_bindings_str);
    PrintBindings(os, const_bindings);
    analysis_result->set_bindings(std::move(const_bindings_str));

    analysis_result->mutable_data_flow()->set_output(analysis->PrintOp(op));

    analysis_result->mutable_computed_constants()->Assign(
        computed_constants.begin(), computed_constants.end());
  }

  return PerformDynamicConstantPropagation(op, *analysis);
}

mlir::LogicalResult PerformDynamicConstantPropagation(
    mlir::Operation *op, JsirDynamicConstantPropagationAnalysis &analysis) {
  return PerformConstantPropagation(op, analysis);
}

}  // namespace maldoca
