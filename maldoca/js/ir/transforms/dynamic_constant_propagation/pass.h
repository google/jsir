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

#ifndef MALDOCA_JS_IR_TRANSFORMS_DYNAMIC_CONSTANT_PROPAGATION_PASS_H_
#define MALDOCA_JS_IR_TRANSFORMS_DYNAMIC_CONSTANT_PROPAGATION_PASS_H_

#include <memory>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/TypeName.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "absl/base/nullability.h"
#include "maldoca/js/babel/babel.h"
#include "maldoca/js/ir/analyses/dynamic_constant_propagation/analysis.h"
#include "maldoca/js/ir/analyses/dynamic_constant_propagation/attrs.h"
#include "maldoca/js/ir/analyses/dynamic_constant_propagation/dynamic_prelude.h"

namespace maldoca {

// Transforms `op` using the dynamic constant propagation analysis.
//
// `babel` is used to minify the prelude.
mlir::LogicalResult PerformDynamicConstantPropagation(
    mlir::Operation *op, const BabelScopes &scopes,
    const JsirAnalysisConfig::DynamicConstantPropagation &config, Babel &babel,
    JsirAnalysisResult::DynamicConstantPropagation
        *absl_nullable analysis_result);

mlir::LogicalResult PerformDynamicConstantPropagation(
    mlir::Operation *op, const BabelScopes &scopes,
    DynamicPrelude *dynamic_prelude,
    JsirAnalysisResult::DynamicConstantPropagation
        *absl_nullable analysis_result);

mlir::LogicalResult PerformDynamicConstantPropagation(
    mlir::Operation *op, JsirDynamicConstantPropagationAnalysis &analysis);

class JsirDynamicConstantPropagationPass : public mlir::OperationPass<> {
 public:
  using PassT = JsirDynamicConstantPropagationPass;

  static mlir::TypeID GetTypeID() {
    return mlir::TypeID::get<JsirDynamicConstantPropagationPass>();
  }

  explicit JsirDynamicConstantPropagationPass(
      const BabelScopes *absl_nonnull scopes,
      JsirAnalysisConfig::DynamicConstantPropagation config,
      Babel *absl_nonnull babel,
      JsAnalysisOutputs *absl_nullable js_analysis_outputs)
      : mlir::OperationPass<>(GetTypeID()),
        scopes_(scopes),
        config_(std::move(config)),
        babel_(babel),
        js_analysis_outputs_(js_analysis_outputs) {}

  static bool classof(const mlir::Pass *pass) {
    return pass->getTypeID() == GetTypeID();
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    OperationPass::getDependentDialects(registry);

    registry.insert<JsirBuiltinDialect>();
  }

  void runOnOperation() override {
    JsirAnalysisResult::DynamicConstantPropagation detailed_analysis_result;
    mlir::LogicalResult result = PerformDynamicConstantPropagation(
        getOperation(), *scopes_, config_, *babel_, &detailed_analysis_result);

    if (mlir::failed(result)) {
      // Failure means that some invariants in the IR have been broken, and the
      // IR might be in an invalid state.
      signalPassFailure();
    }

    if (js_analysis_outputs_ != nullptr) {
      JsAnalysisOutput *js_analysis_output =
          js_analysis_outputs_->add_outputs();
      JsirAnalysisResult *jsir_analysis_output =
          js_analysis_output->mutable_jsir_analysis();
      *jsir_analysis_output->mutable_dynamic_constant_propagation() =
          std::move(detailed_analysis_result);
    }
  }

 protected:
  llvm::StringRef getName() const override {
    return llvm::getTypeName<PassT>();
  }

  std::unique_ptr<mlir::Pass> clonePass() const override {
    return std::make_unique<PassT>(*static_cast<const PassT *>(this));
  }

 private:
  const BabelScopes *absl_nonnull scopes_;
  JsirAnalysisConfig::DynamicConstantPropagation config_;
  Babel *absl_nonnull babel_;
  JsAnalysisOutputs *absl_nullable js_analysis_outputs_;
};

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_TRANSFORMS_DYNAMIC_CONSTANT_PROPAGATION_PASS_H_
