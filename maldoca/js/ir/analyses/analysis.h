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

#ifndef MALDOCA_JS_IR_ANALYSES_ANALYSIS_H_
#define MALDOCA_JS_IR_ANALYSES_ANALYSIS_H_

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "maldoca/js/babel/babel.h"
#include "maldoca/js/babel/babel.pb.h"
#include "maldoca/js/driver/driver.pb.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {

absl::StatusOr<JsirAnalysisResult> RunJsirAnalysis(
    JsirFileOp op, std::optional<absl::string_view> source_code,
    const BabelScopes& scopes, const JsirAnalysisConfig& config,
    Babel* absl_nullable babel);

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_ANALYSES_ANALYSIS_H_
