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

// Converts a JavaScript file into JSIR.

#ifndef MALDOCA_JS_IR_JSIR_GEN_LIB_H_
#define MALDOCA_JS_IR_JSIR_GEN_LIB_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "maldoca/js/babel/babel.h"
#include "maldoca/js/driver/driver.pb.h"

namespace maldoca {

// Types of passes that convert the code to some other format.
enum class JsirPassKind {
  // Conversion
  kSourceToAst,
  kAstToJshir,
  kJshirToJslir,
  kJslirToJshir,
  kJshirToAst,
  kAstToSource,

  // Transform
  kConstantPropagation,
  kDynamicConstantPropagation,
  kMoveNamedFunctions,
  kNormalizeObjectProperties,
  kPeelParentheses,
  kSplitSequenceExpressions,
  kSplitDeclarationStatements,
  kEraseComments,
  kExtractPrelude,
  kRemoveDirectives,
  kNormalizeMemberExpressions,
};

// Analyzes and transforms the provided source code. It first translates the
// JavaScript source code to JSIR, performs transformations on it (determined by
// `passes`), runs the provided analysis on it, and returns the result as a
// Status or string.
struct JsirGenOutput {
  std::string repr;
  JsAnalysisOutputs analysis_outputs;
};

std::string DumpJsAnalysisOutput(absl::string_view original_source,
                                 const JsAnalysisOutput& output);

absl::StatusOr<JsirGenOutput> JsirGen(
    Babel& babel, absl::string_view source,
    const std::vector<JsirPassKind>& passes, JsirAnalysisConfig analysis_config,
    const std::vector<JsirTransformConfig>& transform_configs);

// Analyzes and transforms the provided source code. It first translates the
// JavaScript source code to JSIR, performs transformations on it (determined by
// `passes`), runs the provided analysis on it, and returns the result as a
// Status or string. JsirGen uses a sandboxed babel under the hood. It creates
// the sandboxed babel based off the binary's runfiles directory. The babel
// runfiles for JsirGenHermetic are packaged into the library.
absl::StatusOr<JsirGenOutput> JsirGenHermetic(
    absl::string_view source, const std::vector<JsirPassKind>& passes,
    JsirAnalysisConfig analysis_config,
    const std::vector<JsirTransformConfig>& transform_configs);

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_JSIR_GEN_LIB_H_
