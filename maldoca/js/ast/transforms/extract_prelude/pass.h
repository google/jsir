// Copyright 2025 Google LLC
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

#ifndef MALDOCA_JS_AST_ANALYSES_EXTRACT_PRELUDE_ANALYSIS_H_
#define MALDOCA_JS_AST_ANALYSES_EXTRACT_PRELUDE_ANALYSIS_H_

#include <cstddef>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/driver/driver.pb.h"

namespace maldoca {

absl::Status ForEachTopLevelNode(
    const JsFile& ast, absl::FunctionRef<absl::Status(const JsNode&)> callback);

// Removes parts of the code corresponding to the AST nodes at the given
// indices from the AST, and returns those parts as a string.
//
// Parameters:
// - original_source: The original source code that source ranges in the AST
//                    refer to.
// - indices: The indices of the AST nodes to remove.
// - ast: The AST to transform. Note that `ast` does not need to be equivalent
//        to `original_source`.
//
// Returns:
// - The extracted prelude code.
//
// Modifies:
// - ast: The AST will be modified in place.
JsirAnalysisConfig::DynamicConstantPropagation ExtractPreludeByIndices(
    absl::string_view original_source,
    const absl::flat_hash_set<size_t>& indices, JsFile& ast);

// Removes parts of the code wrapped with `// exec:begin` and `// exec:end`
// comments from the AST, and returns those parts as a string.
//
// For example, if `ast` corresponds to the following code:
//
// ```
// let a = 1;
// // exec:begin
// function foo() {
//   console.log("foo");
// }
// // exec:end
// let b = 2;
// function bar() {
//   console.log("bar");
// }
// ```
//
// Then `ast` will be modified in place into this:
//
// ```
// let a = 1;
// // exec:end
// let b = 2;
// function bar() {
//   console.log("bar");
// }
//
// And the return value will be:
//
// ```
// function foo() {
//   console.log("foo");
// }
// ```
//
// Parameters:
// - original_source: The original source code that source ranges in the AST
//                    refer to.
// - ast: The AST to transform. Note that `ast` does not need to be equivalent
//        to `original_source`.
//
// Returns:
// - The extracted prelude code.
//
// Modifies:
// - ast: The AST will be modified in place.
JsirAnalysisConfig::DynamicConstantPropagation ExtractPreludeByAnnotations(
    absl::string_view original_source, JsFile& ast);

JsirAnalysisConfig::DynamicConstantPropagation
ExtractPreludeByIndicesAndAnnotations(
    absl::string_view original_source,
    const absl::flat_hash_set<size_t>& indices, JsFile& ast);

}  // namespace maldoca

#endif  // MALDOCA_JS_AST_ANALYSES_EXTRACT_PRELUDE_ANALYSIS_H_
