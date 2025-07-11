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

#include "maldoca/js/driver/internal/conversions.h"

#include <memory>
#include <optional>
#include <utility>

#include "google/protobuf/duration.pb.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "maldoca/base/ret_check.h"
#include "maldoca/base/status_macros.h"
#include "maldoca/js/ast/ast_util.h"
#include "maldoca/js/babel/babel.h"
#include "maldoca/js/driver/driver.h"
#include "maldoca/js/ir/conversion/utils.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {

static absl::Duration DecodeDurationProto(
    const google::protobuf::Duration &proto) {
  return absl::Seconds(proto.seconds()) + absl::Nanoseconds(proto.nanos());
}

// =============================================================================
// JsConversion
// =============================================================================

absl::StatusOr<std::unique_ptr<JsConversion>> JsConversion::Create(
    const JsConversionConfig &config, Babel *absl_nullable babel,
    mlir::MLIRContext *absl_nullable mlir_context) {
  switch (config.kind_case()) {
    case JsConversionConfig::KindCase::KIND_NOT_SET: {
      LOG(FATAL) << "Invalid JsConversionConfig kind";
    }

    case JsConversionConfig::KindCase::kJsSourceToAstString: {
      MALDOCA_RET_CHECK(babel != nullptr);
      return JsSourceToAstString::Create(config.js_source_to_ast_string(),
                                         babel);
    }

    case JsConversionConfig::KindCase::kJsAstStringToSource: {
      MALDOCA_RET_CHECK(babel != nullptr);
      return JsAstStringToSource::Create(config.js_ast_string_to_source(),
                                         babel);
    }

    case JsConversionConfig::KindCase::kJsAstStringToAst: {
      return JsAstStringToAst::Create(config.js_ast_string_to_ast());
    }

    case JsConversionConfig::KindCase::kJsAstToAstString: {
      return std::make_unique<JsAstToAstString>();
    }

    case JsConversionConfig::KindCase::kJsAstToHir: {
      MALDOCA_RET_CHECK(mlir_context != nullptr);
      return std::make_unique<JsAstToHir>(mlir_context);
    }

    case JsConversionConfig::KindCase::kJsHirToAst: {
      return std::make_unique<JsHirToAst>();
    }

    case JsConversionConfig::KindCase::kJsHirToLir: {
      return std::make_unique<JsHirToLir>();
    }

    case JsConversionConfig::KindCase::kJsLirToHir: {
      return std::make_unique<JsLirToHir>();
    }
  }
}

// =============================================================================
// JsSourceToAstString
// =============================================================================

absl::StatusOr<std::unique_ptr<JsAstStringRepr>> JsSourceToAstString::Convert(
    const JsSourceRepr &from) {
  MALDOCA_ASSIGN_OR_RETURN(auto parse_result,
                           babel_.Parse(from.source, request_, timeout_));
  return std::make_unique<JsAstStringRepr>(std::move(parse_result.ast_string));
}

absl::StatusOr<std::unique_ptr<JsSourceToAstString>>
JsSourceToAstString::Create(const JsSourceToAstStringConfig &config,
                            Babel *absl_nonnull babel) {
  absl::Duration timeout = absl::InfiniteDuration();
  if (config.has_timeout()) {
    timeout = DecodeDurationProto(config.timeout());
  }
  return std::make_unique<JsSourceToAstString>(
      babel, config.babel_parse_request(), timeout);
}

// =============================================================================
// JsAstStringToSource
// =============================================================================

absl::StatusOr<std::unique_ptr<JsSourceRepr>> JsAstStringToSource::Convert(
    const JsAstStringRepr &from) {
  MALDOCA_ASSIGN_OR_RETURN(
      auto generate_result,
      babel_.Generate(from.ast_string, options_, timeout_));
  return std::make_unique<JsSourceRepr>(std::move(generate_result.source_code));
}

absl::StatusOr<std::unique_ptr<JsAstStringToSource>>
JsAstStringToSource::Create(const JsAstStringToSourceConfig &config,
                            Babel *absl_nonnull babel) {
  absl::Duration timeout = absl::InfiniteDuration();
  if (config.has_timeout()) {
    timeout = DecodeDurationProto(config.timeout());
  }
  return std::make_unique<JsAstStringToSource>(
      babel, config.babel_generate_options(), timeout);
}

// =============================================================================
// JsAstStringToAst
// =============================================================================

absl::StatusOr<std::unique_ptr<JsAstRepr>> JsAstStringToAst::Convert(
    const JsAstStringRepr &from) {
  MALDOCA_ASSIGN_OR_RETURN(
      auto ast,
      GetFileAstFromAstString(from.ast_string, recursion_depth_limit_));
  return std::make_unique<JsAstRepr>(std::move(ast), from.ast_string.scopes());
}

absl::StatusOr<std::unique_ptr<JsAstStringToAst>> JsAstStringToAst::Create(
    const JsAstStringToAstConfig &config) {
  std::optional<int> recursion_depth_limit;
  if (config.has_recursion_depth_limit()) {
    recursion_depth_limit = config.recursion_depth_limit();
  }
  return std::make_unique<JsAstStringToAst>(recursion_depth_limit);
}

// =============================================================================
// JsAstToAstString
// =============================================================================

absl::StatusOr<std::unique_ptr<JsAstStringRepr>> JsAstToAstString::Convert(
    const JsAstRepr &from) {
  auto ast_string = GetAstStringFromFileAst(*from.ast);
  return std::make_unique<JsAstStringRepr>(std::move(ast_string));
}

// =============================================================================
// JsAstToHir
// =============================================================================

absl::StatusOr<std::unique_ptr<JsHirRepr>> JsAstToHir::Convert(
    const JsAstRepr &from) {
  MALDOCA_ASSIGN_OR_RETURN(auto op, AstToJshirFile(*from.ast, mlir_context_));
  return std::make_unique<JsHirRepr>(std::move(op), from.scopes);
}

// =============================================================================
// JsHirToAst
// =============================================================================

absl::StatusOr<std::unique_ptr<JsAstRepr>> JsHirToAst::Convert(
    const JsHirRepr &from) {
  MALDOCA_ASSIGN_OR_RETURN(auto ast, JshirFileToAst(*from.op));
  return std::make_unique<JsAstRepr>(std::move(ast), from.scopes);
}

// =============================================================================
// JsHirToLir
// =============================================================================

absl::StatusOr<std::unique_ptr<JsLirRepr>> JsHirToLir::Convert(
    const JsHirRepr &from) {
  auto op = JshirFileToJslir(*from.op);
  return std::make_unique<JsLirRepr>(std::move(op), from.scopes);
}

// =============================================================================
// JsLirToHir
// =============================================================================

absl::StatusOr<std::unique_ptr<JsHirRepr>> JsLirToHir::Convert(
    const JsLirRepr &from) {
  auto op = JslirFileToJshir(*from.op);
  return std::make_unique<JsHirRepr>(std::move(op), from.scopes);
}

}  // namespace maldoca
