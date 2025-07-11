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

#ifndef MALDOCA_JS_DRIVER_DRIVER_H_
#define MALDOCA_JS_DRIVER_DRIVER_H_

#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

#include "llvm/Support/Casting.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/DebugStringHelper.h"
#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "nlohmann/json.hpp"
#include "maldoca/base/ret_check.h"
#include "maldoca/base/status_macros.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/ast/transforms/transform.h"
#include "maldoca/js/babel/babel.h"
#include "maldoca/js/driver/driver.pb.h"
#include "maldoca/js/ir/analyses/analysis.h"
#include "maldoca/js/ir/ir.h"
#include "maldoca/js/ir/transforms/transform.h"

namespace maldoca {

// =============================================================================
// JsRepr: A representation of JavaScript code
// =============================================================================

enum class JsReprKind {
  kJsSource,
  kAstString,
  kAst,
  kJshir,
  kJslir,
};

std::ostream &operator<<(std::ostream &os, JsReprKind kind);

struct JsRepr {
  const JsReprKind kind;

  virtual ~JsRepr() = default;

  virtual std::string Dump() const = 0;

  virtual absl::StatusOr<JsReprPb> ToProto() const {
    return absl::UnimplementedError(
        absl::StrCat(kind, " cannot be converted to JsReprPb"));
  }

  static absl::StatusOr<std::unique_ptr<JsRepr>> FromProto(
      const JsReprPb &proto);

  template <typename ReprT>
  static absl::StatusOr<ReprT *> Cast(JsRepr *repr) {
    static_assert(std::is_base_of_v<JsRepr, ReprT>,
                  "ReprT must be a subclass of JsRepr");
    if (!llvm::isa<ReprT>(repr)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "expected %s, got %s", typeid(ReprT).name(), typeid(*repr).name()));
    }
    return llvm::cast<ReprT>(repr);
  }

  template <typename ReprT>
  static absl::StatusOr<const ReprT *> Cast(const JsRepr *repr) {
    static_assert(std::is_base_of_v<JsRepr, ReprT>,
                  "ReprT must be a subclass of JsRepr");
    if (!llvm::isa<ReprT>(repr)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "expected %s, got %s", typeid(ReprT).name(), typeid(*repr).name()));
    }
    return llvm::cast<ReprT>(repr);
  }

  template <typename ReprT>
  static absl::StatusOr<std::unique_ptr<ReprT>> Cast(
      std::unique_ptr<JsRepr> repr) {
    static_assert(std::is_base_of_v<JsRepr, ReprT>,
                  "ReprT must be a subclass of JsRepr");
    if (!llvm::isa<ReprT>(repr.get())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "expected %s, got %s", typeid(ReprT).name(), typeid(*repr).name()));
    }
    return llvm::cast<ReprT>(std::move(repr));
  }

 protected:
  explicit JsRepr(JsReprKind kind) : kind(kind) {}
};

struct JsSourceRepr : JsRepr {
  std::string source;

  explicit JsSourceRepr(absl::string_view source)
      : JsRepr(JsReprKind::kJsSource), source(source) {}

  static bool classof(const JsRepr *repr) {
    return repr->kind == JsReprKind::kJsSource;
  }

  std::string Dump() const override { return source; }

  absl::StatusOr<JsReprPb> ToProto() const override;
};

struct JsAstStringRepr : JsRepr {
  BabelAstString ast_string;

  explicit JsAstStringRepr(BabelAstString ast_string)
      : JsRepr(JsReprKind::kAstString), ast_string(std::move(ast_string)) {}

  static bool classof(const JsRepr *repr) {
    return repr->kind == JsReprKind::kAstString;
  }

  std::string Dump() const override { return ast_string.value(); }

  absl::StatusOr<JsReprPb> ToProto() const override;
};

struct JsAstRepr : JsRepr {
  std::unique_ptr<maldoca::JsFile> ast;
  BabelScopes scopes;

  explicit JsAstRepr(std::unique_ptr<maldoca::JsFile> ast, BabelScopes scopes)
      : JsRepr(JsReprKind::kAst),
        ast(std::move(ast)),
        scopes(std::move(scopes)) {}

  static bool classof(const JsRepr *repr) {
    return repr->kind == JsReprKind::kAst;
  }

  std::string Dump() const override {
    std::stringstream ss;
    ast->Serialize(ss);
    auto json = nlohmann::ordered_json::parse(ss.str(), /*cb=*/nullptr,
                                              /* allow_exceptions=*/false);
    return json.dump(2);
  }
};

struct JsirRepr : JsRepr {
  mlir::OwningOpRef<JsirFileOp> op;
  BabelScopes scopes;

  static bool classof(const JsRepr *repr) {
    return repr->kind == JsReprKind::kJshir || repr->kind == JsReprKind::kJslir;
  }

  std::string Dump() const override { return mlir::debugString(*op); }

 protected:
  explicit JsirRepr(JsReprKind kind, mlir::OwningOpRef<JsirFileOp> op,
                    BabelScopes scopes)
      : JsRepr(kind), op(std::move(op)), scopes(std::move(scopes)) {}
};

struct JsHirRepr : JsirRepr {
  explicit JsHirRepr(mlir::OwningOpRef<JsirFileOp> op, BabelScopes scopes)
      : JsirRepr(JsReprKind::kJshir, std::move(op), std::move(scopes)) {}

  static bool classof(const JsRepr *repr) {
    return repr->kind == JsReprKind::kJshir;
  }

  absl::StatusOr<JsReprPb> ToProto() const override;
};

struct JsLirRepr : JsirRepr {
  explicit JsLirRepr(mlir::OwningOpRef<JsirFileOp> op, BabelScopes scopes)
      : JsirRepr(JsReprKind::kJslir, std::move(op), std::move(scopes)) {}

  static bool classof(const JsRepr *repr) {
    return repr->kind == JsReprKind::kJslir;
  }

  absl::StatusOr<JsReprPb> ToProto() const override;
};

// =============================================================================
// JsPass
// =============================================================================

struct JsPassContext {
  std::optional<std::string> original_source;
  std::unique_ptr<JsRepr> repr;
  JsAnalysisOutputs outputs;
};

class JsPass {
 public:
  virtual ~JsPass() = default;
  virtual absl::Status Run(JsPassContext &context) = 0;

  virtual std::string name() const = 0;

  static absl::StatusOr<std::unique_ptr<JsPass>> Create(
      const JsPassConfig &config, Babel *absl_nullable babel,
      mlir::MLIRContext *absl_nullable mlir_context);
};

// =============================================================================
// RunPasses
// =============================================================================

absl::Status RunPasses(const JsPassConfigs &pass_configs,
                       JsPassContext &context, Babel *absl_nullable babel,
                       mlir::MLIRContext *absl_nullable mlir_context);

absl::Status RunPasses(absl::Span<const std::unique_ptr<JsPass>> passes,
                       JsPassContext &context);

bool PassRequiresBabel(const JsPassConfig &pass);

class JsPassRunner {
 public:
  virtual ~JsPassRunner() = default;

  struct Result {
    JsReprPb output_repr_pb;
    JsAnalysisOutputs analysis_outputs;
  };

  virtual absl::StatusOr<Result> Run(absl::string_view original_source,
                                     const JsReprPb &input_repr_pb,
                                     const JsPassConfigs &passes) = 0;
};

class UnsandboxedJsPassRunner : public JsPassRunner {
 public:
  explicit UnsandboxedJsPassRunner(Babel *absl_nullable babel)
      : babel_(babel) {}

  absl::StatusOr<Result> Run(absl::string_view original_source,
                             const JsReprPb &input_repr_pb,
                             const JsPassConfigs &passes) override;

 private:
  Babel *absl_nullable babel_;
};

// =============================================================================
// JsConversion <: JsPass
// =============================================================================
//
// See internal/conversions.{h,cc}. All conversions should be builtin, and
// analysis/transform authors should not modify them.

// =============================================================================
// JsAnalysis <: JsPass
// =============================================================================

class JsAnalysis : public JsPass {
 public:
  virtual absl::Status Analyze(std::optional<absl::string_view> original_source,
                               const JsRepr &repr,
                               JsAnalysisOutputs &output) = 0;

 protected:
  absl::Status Run(JsPassContext &context) override {
    return Analyze(context.original_source, *context.repr, context.outputs);
  }
};

template <typename ReprT>
class JsAnalysisTmpl : public JsAnalysis {
  static_assert(std::is_base_of_v<JsRepr, ReprT>,
                "ReprT must be a subclass of JsRepr");

 public:
  virtual absl::Status Analyze(std::optional<absl::string_view> original_source,
                               const ReprT &repr,
                               JsAnalysisOutputs &outputs) = 0;

 protected:
  absl::Status Analyze(std::optional<absl::string_view> original_source,
                       const JsRepr &repr,
                       JsAnalysisOutputs &outputs) override {
    MALDOCA_ASSIGN_OR_RETURN(const ReprT *repr_cast,
                             JsRepr::Cast<ReprT>(&repr));
    return Analyze(original_source, *repr_cast, outputs);
  }
};

class JsirAnalysis : public JsAnalysisTmpl<JsirRepr> {
 public:
  explicit JsirAnalysis(JsirAnalysisConfig config, Babel *absl_nullable babel)
      : config_(std::move(config)), babel_(babel) {}

  std::string name() const override {
    return absl::StrCat("JsirAnalysis ", config_.kind_case());
  }

  absl::Status Analyze(std::optional<absl::string_view> original_source,
                       const JsirRepr &repr,
                       JsAnalysisOutputs &outputs) override {
    MALDOCA_ASSIGN_OR_RETURN(
        JsirAnalysisResult result,
        RunJsirAnalysis(*repr.op, repr.scopes, config_, babel_));
    *outputs.add_outputs()->mutable_jsir_analysis() = std::move(result);
    return absl::OkStatus();
  }

 private:
  JsirAnalysisConfig config_;
  Babel *absl_nullable babel_;
};

// =============================================================================
// JsTransform
// =============================================================================

class JsTransform : public JsPass {
 public:
  virtual absl::Status Transform(
      std::optional<absl::string_view> original_source, JsRepr &repr,
      JsAnalysisOutputs &outputs) = 0;

 protected:
  absl::Status Run(JsPassContext &context) override {
    return Transform(context.original_source, *context.repr, context.outputs);
  }
};

template <typename ReprT>
class JsTransformTmpl : public JsTransform {
  static_assert(std::is_base_of_v<JsRepr, ReprT>,
                "ReprT must be a subclass of JsRepr");

 public:
  virtual absl::Status Transform(
      std::optional<absl::string_view> original_source, ReprT &repr,
      JsAnalysisOutputs &outputs) = 0;

 protected:
  absl::Status Transform(std::optional<absl::string_view> original_source,
                         JsRepr &repr, JsAnalysisOutputs &outputs) override {
    MALDOCA_ASSIGN_OR_RETURN(ReprT * repr_cast, JsRepr::Cast<ReprT>(&repr));
    return Transform(original_source, *repr_cast, outputs);
  }
};

class JsAstTransform : public JsTransformTmpl<JsAstRepr> {
 public:
  explicit JsAstTransform(JsAstTransformConfig config)
      : config_(std::move(config)) {}

  std::string name() const override {
    return absl::StrCat("JsAstTransform ", config_.kind_case());
  }

  absl::Status Transform(std::optional<absl::string_view> original_source,
                         JsAstRepr &repr, JsAnalysisOutputs &outputs) override {
    std::optional<JsAstAnalysisResult> optional_analysis_result;
    MALDOCA_RETURN_IF_ERROR(TransformJsAst(original_source, repr.scopes,
                                           config_, *repr.ast,
                                           optional_analysis_result));
    if (optional_analysis_result.has_value()) {
      *outputs.add_outputs()->mutable_ast_analysis() =
          std::move(*optional_analysis_result);
    }
    return absl::OkStatus();
  }

  JsAstTransformConfig config_;
};

class JsirTransform : public JsTransformTmpl<JsirRepr> {
 public:
  explicit JsirTransform(JsirTransformConfig config, Babel *absl_nullable babel)
      : config_(std::move(config)), babel_(babel) {}

  std::string name() const override {
    return absl::StrCat("JsirTransform ", config_.kind_case());
  }

 private:
  absl::Status Transform(std::optional<absl::string_view> original_source,
                         JsirRepr &repr, JsAnalysisOutputs &outputs) override {
    return TransformJsir(*repr.op, repr.scopes, config_, babel_, &outputs);
  }

  JsirTransformConfig config_;
  Babel *absl_nullable babel_;
};

}  // namespace maldoca

#endif  // MALDOCA_JS_DRIVER_DRIVER_H_
