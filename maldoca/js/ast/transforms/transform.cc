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

#include "maldoca/js/ast/transforms/transform.h"

#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/ast/transforms/erase_comments/pass.h"
#include "maldoca/js/ast/transforms/extract_prelude/pass.h"
#include "maldoca/js/babel/babel.pb.h"
#include "maldoca/js/driver/driver.pb.h"

namespace maldoca {

absl::Status TransformJsAst(
    std::optional<absl::string_view> original_source, const BabelScopes &scopes,
    const JsAstTransformConfig &config, JsFile &ast,
    std::optional<JsAstAnalysisResult> &optional_analysis_result) {
  switch (config.kind_case()) {
    case JsAstTransformConfig::KIND_NOT_SET: {
      return absl::InvalidArgumentError("JsAstTransformConfig kind not set");
    }

    case JsAstTransformConfig::kEraseComments: {
      EraseCommentsInAst(ast);
      return absl::OkStatus();
    }

    case JsAstTransformConfig::kExtractPrelude: {
      if (!original_source.has_value()) {
        return absl::InvalidArgumentError(
            "original_source is required for ExtractPrelude analysis");
      }

      JsirAnalysisConfig::DynamicConstantPropagation prelude =
          ExtractPrelude(*original_source, ast);

      JsAstAnalysisResult result;
      *result.mutable_extract_prelude() = std::move(prelude);

      optional_analysis_result = std::move(result);

      return absl::OkStatus();
    }
  }
}

}  // namespace maldoca
