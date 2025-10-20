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

#ifndef MALDOCA_JS_IR_TRANSFORMS_TRANSFORM_H_
#define MALDOCA_JS_IR_TRANSFORMS_TRANSFORM_H_

#include <memory>
#include <vector>

#include "mlir/Pass/Pass.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/babel/babel.h"
#include "maldoca/js/babel/babel.pb.h"
#include "maldoca/js/driver/driver.pb.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {

// Creates the corresponding pass based on the given transform config.
//
// - scopes: Symbol table required for some analyses.
// - config: The specification of the transform.
// - babel: Used for simple code manipulation for some analyses.
// - analysis_outputs: Used to append analysis results for some analyses.
absl::StatusOr<std::unique_ptr<mlir::Pass>> CreateJsirTransformPass(
    const BabelScopes *absl_nullable scopes, const JsirTransformConfig &config,
    Babel *absl_nullable babel,
    JsAnalysisOutputs *absl_nullable analysis_outputs);

// Performs a single transform on a JSHIR module.
absl::Status TransformJsir(JsirFileOp jsir_file, const BabelScopes &scopes,
                           const JsirTransformConfig &config,
                           Babel *absl_nullable babel,
                           JsAnalysisOutputs *absl_nullable analysis_outputs);

// Performs the given list of transforms on a JSHIR module.
absl::Status TransformJsir(JsirFileOp jsir_file, const BabelScopes &scopes,
                           std::vector<JsirTransformConfig> configs,
                           Babel *absl_nullable babel,
                           JsAnalysisOutputs *absl_nullable analysis_outputs);

// Converts the AST into JSHIR, performs the given list of transforms, and
// converts back to an AST.
absl::StatusOr<std::unique_ptr<JsFile>> TransformJsAst(
    const JsFile &ast, const BabelScopes &scopes,
    std::vector<JsirTransformConfig> configs, Babel *absl_nullable babel);

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_TRANSFORMS_TRANSFORM_H_
