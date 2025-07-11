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

#ifndef MALDOCA_JS_DRIVER_INTERNAL_CONVERSIONS_H_
#define MALDOCA_JS_DRIVER_INTERNAL_CONVERSIONS_H_

#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

#include "mlir/IR/MLIRContext.h"
#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "maldoca/base/status_macros.h"
#include "maldoca/js/babel/babel.h"
#include "maldoca/js/driver/driver.h"
#include "maldoca/js/driver/driver.pb.h"

namespace maldoca {

class JsConversion : public JsPass {
 public:
  static absl::StatusOr<std::unique_ptr<JsConversion>> Create(
      const JsConversionConfig &config, Babel *absl_nullable babel,
      mlir::MLIRContext *absl_nullable mlir_context);

  virtual absl::Status Convert(std::unique_ptr<JsRepr> &repr) = 0;

 protected:
  absl::Status Run(JsPassContext &context) override {
    return Convert(context.repr);
  }
};

template <typename FromRepr, typename ToRepr>
class JsConversionTmpl : public JsConversion {
  static_assert(std::is_base_of_v<JsRepr, FromRepr>,
                "FromRepr must be a subclass of JsRepr");
  static_assert(std::is_base_of_v<JsRepr, ToRepr>,
                "ToRepr must be a subclass of JsRepr");

 public:
  virtual absl::StatusOr<std::unique_ptr<ToRepr>> Convert(
      const FromRepr &from) = 0;

 protected:
  absl::Status Convert(std::unique_ptr<JsRepr> &repr) override {
    MALDOCA_ASSIGN_OR_RETURN(auto *from, JsRepr::Cast<FromRepr>(repr.get()));
    MALDOCA_ASSIGN_OR_RETURN(repr, Convert(*from));
    return absl::OkStatus();
  }
};

// +------+----------------+-------------------------------+
// |      | Type           | Representation                |
// +------+----------------+-------------------------------+
// | From | std::string    | JavaScript source             |
// +------+----------------+-------------------------------+
// |  To  | BabelAstString | JavaScript AST as JSON string |
// +------+----------------+-------------------------------+
class JsSourceToAstString final
    : public JsConversionTmpl<JsSourceRepr, JsAstStringRepr> {
 public:
  explicit JsSourceToAstString(Babel *absl_nonnull babel,
                               BabelParseRequest request,
                               absl::Duration timeout)
      : babel_(*babel), request_(std::move(request)), timeout_(timeout) {}

  static absl::StatusOr<std::unique_ptr<JsSourceToAstString>> Create(
      const JsSourceToAstStringConfig &config, Babel *absl_nonnull babel);

  std::string name() const override { return "JsSourceToAstString"; }

 private:
  absl::StatusOr<std::unique_ptr<JsAstStringRepr>> Convert(
      const JsSourceRepr &from) override;

  Babel &babel_;
  BabelParseRequest request_;
  absl::Duration timeout_;
};

// +------+----------------+-------------------------------+
// |      | Type           | Representation                |
// +------+----------------+-------------------------------+
// | From | BabelAstString | JavaScript AST as JSON string |
// +------+----------------+-------------------------------+
// |  To  | std::string    | JavaScript source             |
// +------+----------------+-------------------------------+
class JsAstStringToSource final
    : public JsConversionTmpl<JsAstStringRepr, JsSourceRepr> {
 public:
  explicit JsAstStringToSource(Babel *absl_nonnull babel,
                               BabelGenerateOptions options,
                               absl::Duration timeout)
      : babel_(*babel), options_(std::move(options)), timeout_(timeout) {}

  static absl::StatusOr<std::unique_ptr<JsAstStringToSource>> Create(
      const JsAstStringToSourceConfig &config, Babel *absl_nonnull babel);

  std::string name() const override { return "JsAstStringToSource"; }

 private:
  absl::StatusOr<std::unique_ptr<JsSourceRepr>> Convert(
      const JsAstStringRepr &from) override;

  Babel &babel_;
  BabelGenerateOptions options_;
  absl::Duration timeout_;
};

// +------+-------------------------+-------------------------------+
// |      | Type                    | Representation                |
// +------+-------------------------+-------------------------------+
// | From | BabelAstString          | JavaScript AST as JSON string |
// +------+-------------------------+-------------------------------+
// |  To  | std::unique_ptr<JsFile> | JavaScript AST                |
// +------+-------------------------+-------------------------------+
class JsAstStringToAst final
    : public JsConversionTmpl<JsAstStringRepr, JsAstRepr> {
 public:
  explicit JsAstStringToAst(std::optional<int> recursion_depth_limit)
      : recursion_depth_limit_(recursion_depth_limit) {}

  static absl::StatusOr<std::unique_ptr<JsAstStringToAst>> Create(
      const JsAstStringToAstConfig &config);

  std::string name() const override { return "JsAstStringToAst"; }

 private:
  absl::StatusOr<std::unique_ptr<JsAstRepr>> Convert(
      const JsAstStringRepr &from) override;

  std::optional<int> recursion_depth_limit_;
};

// +------+-------------------------+-------------------------------+
// |      | Type                    | Representation                |
// +------+-------------------------+-------------------------------+
// | From | std::unique_ptr<JsFile> | JavaScript AST                |
// +------+-------------------------+-------------------------------+
// |  To  | BabelAstString          | JavaScript AST as JSON string |
// +------+-------------------------+-------------------------------+
class JsAstToAstString final
    : public JsConversionTmpl<JsAstRepr, JsAstStringRepr> {
 public:
  explicit JsAstToAstString() = default;

  std::string name() const override { return "JsAstToAstString"; }

 private:
  absl::StatusOr<std::unique_ptr<JsAstStringRepr>> Convert(
      const JsAstRepr &from) override;
};

// +------+-----------------------------+-------------------------------+
// |      | Type                        | Representation                |
// +------+-----------------------------+-------------------------------+
// | From | std::unique_ptr<JsFile>     | JavaScript AST                |
// +------+-----------------------------+-------------------------------+
// |  To  | mlir::OwningOpRef<JsFileOp> | JavaScript high-level IR      |
// +------+-----------------------------+-------------------------------+
class JsAstToHir final : public JsConversionTmpl<JsAstRepr, JsHirRepr> {
 public:
  explicit JsAstToHir(mlir::MLIRContext *absl_nonnull mlir_context)
      : mlir_context_(*mlir_context) {}

  std::string name() const override { return "JsAstToHir"; }

 private:
  absl::StatusOr<std::unique_ptr<JsHirRepr>> Convert(
      const JsAstRepr &from) override;

  mlir::MLIRContext &mlir_context_;
};

// +------+-------------------------------+-------------------------------+
// |      | Type                          | Representation                |
// +------+-------------------------------+-------------------------------+
// | From | mlir::OwningOpRef<JsirFileOp> | JavaScript high-level IR      |
// +------+-------------------------------+-------------------------------+
// |  To  | std::unique_ptr<JsFile>       | JavaScript AST                |
// +------+-------------------------------+-------------------------------+
class JsHirToAst final : public JsConversionTmpl<JsHirRepr, JsAstRepr> {
 public:
  explicit JsHirToAst() = default;

  std::string name() const override { return "JsHirToAst"; }

 private:
  absl::StatusOr<std::unique_ptr<JsAstRepr>> Convert(
      const JsHirRepr &from) override;
};

// +------+-------------------------------+-------------------------------+
// |      | Type                          | Representation                |
// +------+-------------------------------+-------------------------------+
// | From | mlir::OwningOpRef<JsirFileOp> | JavaScript high-level IR      |
// +------+-------------------------------+-------------------------------+
// |  To  | mlir::OwningOpRef<JsirFileOp> | JavaScript low-level IR       |
// +------+-------------------------------+-------------------------------+
class JsHirToLir final : public JsConversionTmpl<JsHirRepr, JsLirRepr> {
 public:
  explicit JsHirToLir() = default;

  std::string name() const override { return "JsHirToLir"; }

 private:
  absl::StatusOr<std::unique_ptr<JsLirRepr>> Convert(
      const JsHirRepr &from) override;
};

// +------+-------------------------------+-------------------------------+
// |      | Type                          | Representation                |
// +------+-------------------------------+-------------------------------+
// | From | mlir::OwningOpRef<JsirFileOp> | JavaScript low-level IR       |
// +------+-------------------------------+-------------------------------+
// |  To  | mlir::OwningOpRef<JsirFileOp> | JavaScript high-level IR      |
// +------+-------------------------------+-------------------------------+
class JsLirToHir final : public JsConversionTmpl<JsLirRepr, JsHirRepr> {
 public:
  explicit JsLirToHir() = default;

  std::string name() const override { return "JsLirToHir"; }

 private:
  absl::StatusOr<std::unique_ptr<JsHirRepr>> Convert(
      const JsLirRepr &from) override;
};

}  // namespace maldoca

#endif  // MALDOCA_JS_DRIVER_INTERNAL_CONVERSIONS_H_
