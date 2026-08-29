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

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "absl/base/nullability.h"
#include "maldoca/js/babel/babel.h"
#include "maldoca/js/ir/analyses/dynamic_constant_propagation/analysis.h"
#include "maldoca/js/ir/analyses/dynamic_constant_propagation/dynamic_prelude.h"

namespace maldoca {

// Dynamic constant propagation is the same rewrite as ordinary constant
// propagation, with JsirDynamicConstantPropagationAnalysis. The MLIR pass
// lives in JsirConstantPropagationPass.

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

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_TRANSFORMS_DYNAMIC_CONSTANT_PROPAGATION_PASS_H_
