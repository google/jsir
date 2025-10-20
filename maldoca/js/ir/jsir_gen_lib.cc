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

#include "maldoca/js/ir/jsir_gen_lib.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/duration.pb.h"
#include "mlir/Pass/PassManager.h"
#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "maldoca/base/ret_check.h"
#include "maldoca/js/babel/babel.h"
#include "maldoca/js/babel/babel.pb.h"
#include "maldoca/js/driver/driver.h"
#include "maldoca/js/driver/driver.pb.h"
#include "maldoca/js/ir/conversion/utils.h"
#include "maldoca/js/quickjs_babel/quickjs_babel.h"

namespace maldoca {
namespace {

std::string DumpJsAstAnalysisResult(absl::string_view original_source,
                                    const JsAstAnalysisResult &result) {
  switch (result.kind_case()) {
    case JsAstAnalysisResult::KIND_NOT_SET:
      return "";
    case JsAstAnalysisResult::kExtractPrelude:
      return absl::StrCat(
          "================================================================\n",
          "Prelude:\n",
          "================================================================\n",
          result.extract_prelude().prelude_source(),
          "================================================================\n");
  }
}

std::string DumpJsirAnalysisResult(absl::string_view original_source,
                                   const JsirAnalysisResult &result) {
  switch (result.kind_case()) {
    case JsirAnalysisResult::KIND_NOT_SET:
      return "";

    case JsirAnalysisResult::kConstantPropagation:
      return result.constant_propagation().output();

    case JsirAnalysisResult::kDynamicConstantPropagation: {
      using ComputedConstant =
          JsirAnalysisResult::DynamicConstantPropagation::ComputedConstant;

      auto formatter = [&](std::string *out, const ComputedConstant &constant) {
        absl::StrAppendFormat(
            out, "From [%d, %d): `%s` -> `", constant.start_offset(),
            constant.end_offset(),
            original_source.substr(
                constant.start_offset(),
                constant.end_offset() - constant.start_offset()));
        switch (constant.value_kind_case()) {
          case ComputedConstant::kStringValue:
            absl::StrAppend(out, constant.string_value());
            break;
          case ComputedConstant::kNumberValue:
            absl::StrAppend(out, constant.number_value());
            break;
          case ComputedConstant::kBigIntValue:
            absl::StrAppend(out, "BigInt(", constant.big_int_value(), ")");
            break;
          case ComputedConstant::kBoolValue:
            absl::StrAppend(out, constant.bool_value() ? "true" : "false");
            break;
          default:
            break;
        }
        absl::StrAppend(out, "`");
      };

      return absl::StrCat(
          "================================================================\n",
          "Dataflow:\n",
          "================================================================\n",
          result.dynamic_constant_propagation().data_flow().output(), "\n",
          "================================================================\n",
          "\n",
          "================================================================\n",
          "Bindings:\n",
          "================================================================\n",
          result.dynamic_constant_propagation().bindings(),
          "================================================================\n",
          "\n",
          "================================================================\n",
          "Computed constants:\n",
          "================================================================\n",
          absl::StrJoin(
              result.dynamic_constant_propagation().computed_constants(), "\n",
              formatter),
          "\n",
          "================================================================\n");
    }
  }
}

}  // namespace

std::string DumpJsAnalysisOutput(absl::string_view original_source,
                                 const JsAnalysisOutput &output) {
  switch (output.kind_case()) {
    case JsAnalysisOutput::KIND_NOT_SET:
      LOG(FATAL) << "JsAnalysisOutput::KIND_NOT_SET";
    case JsAnalysisOutput::kAstAnalysis:
      return DumpJsAstAnalysisResult(original_source, output.ast_analysis());
    case JsAnalysisOutput::kJsirAnalysis:
      return DumpJsirAnalysisResult(original_source, output.jsir_analysis());
  }
}

absl::StatusOr<JsirGenOutput> JsirGen(
    Babel &babel, absl::string_view source,
    const std::vector<JsirPassKind> &passes, JsirAnalysisConfig analysis_config,
    const std::vector<JsirTransformConfig> &transform_configs) {
  mlir::MLIRContext mlir_context;
  LoadNecessaryDialects(mlir_context);

  JsPassContext pass_context{
      .original_source = std::string(source),
      .repr = std::make_unique<JsSourceRepr>(source),
      .outputs = {},
  };

  JsPassConfigs pass_configs;
  for (JsirPassKind pass_kind : passes) {
    switch (pass_kind) {
      case JsirPassKind::kSourceToAst: {
        {
          BabelParseRequest request;
          request.set_compute_scopes(true);

          google::protobuf::Duration timeout;
          timeout.set_seconds(60);

          JsSourceToAstStringConfig source_to_ast_string;
          *source_to_ast_string.mutable_babel_parse_request() =
              std::move(request);
          *source_to_ast_string.mutable_timeout() = std::move(timeout);

          JsConversionConfig conversion;
          *conversion.mutable_js_source_to_ast_string() =
              std::move(source_to_ast_string);

          *pass_configs.add_passes()->mutable_conversion() =
              std::move(conversion);
        }
        {
          JsAstStringToAstConfig ast_string_to_ast;

          JsConversionConfig conversion;
          *conversion.mutable_js_ast_string_to_ast() =
              std::move(ast_string_to_ast);

          *pass_configs.add_passes()->mutable_conversion() =
              std::move(conversion);
        }
        break;
      }

      case JsirPassKind::kAstToJshir: {
        JsConversionConfig conversion;
        *conversion.mutable_js_ast_to_hir() = {};

        *pass_configs.add_passes()->mutable_conversion() =
            std::move(conversion);

        break;
      }

      case JsirPassKind::kJshirToAst: {
        JsConversionConfig conversion;
        *conversion.mutable_js_hir_to_ast() = {};

        *pass_configs.add_passes()->mutable_conversion() =
            std::move(conversion);

        break;
      }

      case JsirPassKind::kAstToSource: {
        {
          JsAstToAstStringConfig ast_to_ast_string;

          JsConversionConfig conversion;
          *conversion.mutable_js_ast_to_ast_string() =
              std::move(ast_to_ast_string);

          *pass_configs.add_passes()->mutable_conversion() =
              std::move(conversion);
        }
        {
          JsAstStringToSourceConfig ast_string_to_source;

          JsConversionConfig conversion;
          *conversion.mutable_js_ast_string_to_source() =
              std::move(ast_string_to_source);

          *pass_configs.add_passes()->mutable_conversion() =
              std::move(conversion);
        }
        break;
      }

      case JsirPassKind::kEraseComments: {
        JsAstTransformConfig transform;
        *transform.mutable_erase_comments() = {};

        *pass_configs.add_passes()->mutable_ast_transform() =
            std::move(transform);

        break;
      }

      case JsirPassKind::kConstantPropagation: {
        JsirTransformConfig transform;
        transform.mutable_constant_propagation();

        *pass_configs.add_passes()->mutable_jsir_transform() =
            std::move(transform);

        break;
      }

      case JsirPassKind::kMoveNamedFunctions: {
        JsirTransformConfig transform;
        *transform.mutable_move_named_functions() = {};

        *pass_configs.add_passes()->mutable_jsir_transform() =
            std::move(transform);

        break;
      }

      case JsirPassKind::kNormalizeObjectProperties: {
        JsirTransformConfig transform;
        *transform.mutable_normalize_object_properties() = {};

        *pass_configs.add_passes()->mutable_jsir_transform() =
            std::move(transform);

        break;
      }

      case JsirPassKind::kPeelParentheses: {
        JsirTransformConfig transform;
        *transform.mutable_peel_parentheses() = {};

        *pass_configs.add_passes()->mutable_jsir_transform() =
            std::move(transform);

        break;
      }

      case JsirPassKind::kSplitSequenceExpressions: {
        JsirTransformConfig transform;
        *transform.mutable_split_sequence_expressions() = {};

        *pass_configs.add_passes()->mutable_jsir_transform() =
            std::move(transform);

        break;
      }

      case JsirPassKind::kSplitDeclarationStatements: {
        JsirTransformConfig transform;
        *transform.mutable_split_declaration_statements() = {};

        *pass_configs.add_passes()->mutable_jsir_transform() =
            std::move(transform);

        break;
      }

      case JsirPassKind::kRemoveDirectives: {
        JsirTransformConfig transform;
        *transform.mutable_remove_directives() = {};

        *pass_configs.add_passes()->mutable_jsir_transform() =
            std::move(transform);

        break;
      }

      case JsirPassKind::kDynamicConstantPropagation: {
        auto it = absl::c_find_if(transform_configs, [](const auto &config) {
          return config.has_dynamic_constant_propagation();
        });
        MALDOCA_RET_CHECK(it != transform_configs.end());
        const JsirTransformConfig &transform = *it;

        *pass_configs.add_passes()->mutable_jsir_transform() =
            std::move(transform);

        break;
      }

      case JsirPassKind::kExtractPrelude: {
        JsAstTransformConfig transform;
        *transform.mutable_extract_prelude() = {};

        *pass_configs.add_passes()->mutable_ast_transform() =
            std::move(transform);

        break;
      }
    }
  }

  switch (analysis_config.kind_case()) {
    case JsirAnalysisConfig::KIND_NOT_SET:
      break;
    default:
      *pass_configs.add_passes()->mutable_jsir_analysis() =
          std::move(analysis_config);
      break;
  }

  MALDOCA_RET_CHECK_OK(
      RunPasses(pass_configs, pass_context, &babel, &mlir_context));

  return JsirGenOutput{
      .repr = pass_context.repr->Dump(),
      .analysis_outputs = std::move(pass_context.outputs),
  };
}

absl::StatusOr<JsirGenOutput> JsirGenHermetic(
    absl::string_view source, const std::vector<JsirPassKind> &passes,
    JsirAnalysisConfig analysis_config,
    const std::vector<JsirTransformConfig> &transform_configs) {
  QuickJsBabel babel;

  return maldoca::JsirGen(babel, source, passes, analysis_config,
                          transform_configs);
}

}  // namespace maldoca
