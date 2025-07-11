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

#include "maldoca/js/driver/driver.h"

#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/DebugStringHelper.h"
#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "maldoca/base/status_macros.h"
#include "maldoca/js/babel/babel.h"
#include "maldoca/js/driver/internal/conversions.h"
#include "maldoca/js/ir/conversion/utils.h"

namespace maldoca {

// =============================================================================
// JsRepr
// =============================================================================

std::ostream &operator<<(std::ostream &os, JsReprKind kind) {
  switch (kind) {
    case JsReprKind::kJsSource:
      return os << "JsSource";
    case JsReprKind::kAstString:
      return os << "AstString";
    case JsReprKind::kAst:
      return os << "Ast";
    case JsReprKind::kJshir:
      return os << "Jshir";
    case JsReprKind::kJslir:
      return os << "Jslir";
  }
}

absl::StatusOr<std::unique_ptr<JsRepr>> JsRepr::FromProto(
    const JsReprPb &proto) {
  switch (proto.kind_case()) {
    case JsReprPb::KIND_NOT_SET:
      return absl::InvalidArgumentError("JsReprPb kind not set");
    case JsReprPb::kJsSource:
      return std::make_unique<JsSourceRepr>(proto.js_source());
    case JsReprPb::kBabelAstString:
      return std::make_unique<JsAstStringRepr>(proto.babel_ast_string());
    case JsReprPb::kJsHir:
    case JsReprPb::kJsLir:
      return absl::UnimplementedError("JSIR parsing not supported");
  }
}

absl::StatusOr<JsReprPb> JsSourceRepr::ToProto() const {
  JsReprPb proto;
  proto.set_js_source(source);
  return proto;
}

absl::StatusOr<JsReprPb> JsAstStringRepr::ToProto() const {
  JsReprPb proto;
  *proto.mutable_babel_ast_string() = ast_string;
  return proto;
}

absl::StatusOr<JsReprPb> JsHirRepr::ToProto() const {
  JsReprPb proto;
  proto.set_js_hir(mlir::debugString(*op));
  return proto;
}

absl::StatusOr<JsReprPb> JsLirRepr::ToProto() const {
  JsReprPb proto;
  proto.set_js_lir(mlir::debugString(*op));
  return proto;
}

// =============================================================================
// JsPass
// =============================================================================

absl::StatusOr<std::unique_ptr<JsPass>> JsPass::Create(
    const JsPassConfig &config, Babel *absl_nullable babel,
    mlir::MLIRContext *absl_nullable mlir_context) {
  switch (config.kind_case()) {
    case JsPassConfig::KIND_NOT_SET: {
      LOG(FATAL) << "Pass kind not set";
    }

    case JsPassConfig::kConversion: {
      const JsConversionConfig &conversion = config.conversion();
      return JsConversion::Create(conversion, babel, mlir_context);
    }

    case JsPassConfig::kJsirAnalysis: {
      const JsirAnalysisConfig &jsir_analysis = config.jsir_analysis();
      return std::make_unique<JsirAnalysis>(jsir_analysis, babel);
    }

    case JsPassConfig::kAstTransform: {
      const JsAstTransformConfig &transform = config.ast_transform();
      return std::make_unique<JsAstTransform>(transform);
    }

    case JsPassConfig::kJsirTransform: {
      const JsirTransformConfig &transform = config.jsir_transform();
      return std::make_unique<JsirTransform>(transform, babel);
    }
  }
}

// =============================================================================
// RunPasses
// =============================================================================

absl::Status RunPasses(absl::Span<const std::unique_ptr<JsPass>> passes,
                       JsPassContext &context) {
  for (const auto &pass : passes) {
    absl::Status status = pass->Run(context);
    if (!status.ok()) {
      status.SetPayload("pass", absl::Cord{pass->name()});
      return status;
    }
  }
  return absl::OkStatus();
}

absl::Status RunPasses(const JsPassConfigs &pass_configs,
                       JsPassContext &context, Babel *absl_nullable babel,
                       mlir::MLIRContext *absl_nullable mlir_context) {
  std::vector<std::unique_ptr<JsPass>> passes;

  for (const JsPassConfig &pass_config : pass_configs.passes()) {
    MALDOCA_ASSIGN_OR_RETURN(std::unique_ptr<JsPass> pass,
                             JsPass::Create(pass_config, babel, mlir_context));
    passes.push_back(std::move(pass));
  }

  return RunPasses(passes, context);
}

bool PassRequiresBabel(const JsPassConfig &pass) {
  switch (pass.kind_case()) {
    case JsPassConfig::kConversion:
      switch (pass.conversion().kind_case()) {
        case JsConversionConfig::KindCase::kJsSourceToAstString:
        case JsConversionConfig::KindCase::kJsAstStringToSource:
          return true;
        default:
          return false;
      }
    default:
      return false;
  }
}

absl::StatusOr<JsPassRunner::Result> UnsandboxedJsPassRunner::Run(
    absl::string_view original_source, const JsReprPb &input_repr_pb,
    const JsPassConfigs &passes) {
  MALDOCA_ASSIGN_OR_RETURN(std::unique_ptr<JsRepr> input_repr,
                           JsRepr::FromProto(input_repr_pb));

  JsPassContext context{
      .original_source = std::string(original_source),
      .repr = std::move(input_repr),
      .outputs = {},
  };

  mlir::MLIRContext mlir_context;
  LoadNecessaryDialects(mlir_context);

  MALDOCA_RETURN_IF_ERROR(RunPasses(passes, context, babel_, &mlir_context));

  MALDOCA_ASSIGN_OR_RETURN(JsReprPb output_repr_pb, context.repr->ToProto());

  return JsPassRunner::Result{
      .output_repr_pb = std::move(output_repr_pb),
      .analysis_outputs = std::move(context.outputs),
  };
}

}  // namespace maldoca
