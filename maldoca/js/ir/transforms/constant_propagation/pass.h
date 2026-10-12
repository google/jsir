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

#ifndef MALDOCA_JS_IR_TRANSFORMS_CONSTANT_PROPAGATION_PASS_H_
#define MALDOCA_JS_IR_TRANSFORMS_CONSTANT_PROPAGATION_PASS_H_

#include <utility>

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "absl/base/nullability.h"
#include "maldoca/js/babel/babel.h"
#include "maldoca/js/driver/driver.pb.h"
#include "maldoca/js/ir/analyses/constant_propagation/dynamic_analysis.h"

namespace maldoca {

class DynamicPrelude;

// Always runs JsirDynamicConstantPropagationAnalysis. With no prelude, the
// analysis falls back to ordinary constant propagation.
mlir::LogicalResult PerformConstantPropagation(mlir::Operation *op,
                                               const BabelScopes &scopes);

mlir::LogicalResult PerformConstantPropagation(
    mlir::Operation *op, JsirConstantPropagationAnalysis &analysis);

mlir::LogicalResult PerformDynamicConstantPropagation(
    mlir::Operation* op, const BabelScopes& scopes,
    const JsirAnalysisConfig::DynamicConstantPropagation& config, Babel& babel,
    JsirAnalysisResult::DynamicConstantPropagation* absl_nullable
        analysis_result);

mlir::LogicalResult PerformDynamicConstantPropagation(
    mlir::Operation* op, const BabelScopes& scopes,
    DynamicPrelude* absl_nullable dynamic_prelude,
    JsirAnalysisResult::DynamicConstantPropagation* absl_nullable
        analysis_result);

mlir::LogicalResult PerformDynamicConstantPropagation(
    mlir::Operation* op, JsirDynamicConstantPropagationAnalysis& analysis);

// Ordinary (constprop) and dynamic (dynconstprop) share this pass. When babel
// is set, prelude matching runs first; otherwise the analysis falls back.
struct JsirConstantPropagationPass
    : public mlir::PassWrapper<JsirConstantPropagationPass,
                               mlir::OperationPass<>> {
  using Base =
      mlir::PassWrapper<JsirConstantPropagationPass, mlir::OperationPass<>>;

  explicit JsirConstantPropagationPass(const BabelScopes *scopes)
      : Base(), scopes_(*scopes) {}

  JsirConstantPropagationPass(
      const BabelScopes* absl_nonnull scopes,
      JsirAnalysisConfig::DynamicConstantPropagation config,
      Babel* absl_nonnull babel,
      JsAnalysisOutputs* absl_nullable js_analysis_outputs)
      : Base(),
        scopes_(*scopes),
        dynamic_config_(std::move(config)),
        babel_(babel),
        js_analysis_outputs_(js_analysis_outputs) {}

  void getDependentDialects(mlir::DialectRegistry& registry) const override;

  void runOnOperation() override;

  const BabelScopes &scopes_;
  JsirAnalysisConfig::DynamicConstantPropagation dynamic_config_;
  Babel* absl_nullable babel_ = nullptr;
  JsAnalysisOutputs* absl_nullable js_analysis_outputs_ = nullptr;
};

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_TRANSFORMS_CONSTANT_PROPAGATION_PASS_H_
