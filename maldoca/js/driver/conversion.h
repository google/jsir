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

#ifndef MALDOCA_JS_DRIVER_CONVERSION_H_
#define MALDOCA_JS_DRIVER_CONVERSION_H_

#include <optional>

#include "mlir/IR/MLIRContext.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/babel/babel.h"
#include "maldoca/js/babel/babel.pb.h"
#include "maldoca/js/driver/driver.h"

namespace maldoca {

// We have 5 types of JavaScript representations:

// 1. Source
// 2. AST string
// 3. AST
// 4. HIR
// 5. LIR
//
// The following functions convert between these representations.

struct ToJsSourceRepr {
  static absl::StatusOr<JsSourceRepr> FromJsAstStringRepr(
      const JsAstStringRepr &ast_string_repr,
      BabelGenerateOptions generate_options, absl::Duration timeout,
      Babel &babel);

  static absl::StatusOr<JsSourceRepr> FromJsAstRepr(
      const JsAstRepr &ast_repr, BabelGenerateOptions generate_options,
      absl::Duration timeout, Babel &babel);

  static absl::StatusOr<JsSourceRepr> FromJsirRepr(
      const JsirRepr& ir_repr, BabelGenerateOptions generate_options,
      absl::Duration timeout, Babel& babel);
};

struct ToJsAstStringRepr {
  static absl::StatusOr<JsAstStringRepr> FromJsSourceRepr(
      const JsSourceRepr &source_repr, BabelParseRequest parse_request,
      absl::Duration timeout, Babel &babel);

  static absl::StatusOr<JsAstStringRepr> FromJsAstRepr(
      const JsAstRepr &ast_repr);

  static absl::StatusOr<JsAstStringRepr> FromJsirRepr(const JsirRepr& ir_repr);
};

struct ToJsAstRepr {
  static absl::StatusOr<JsAstRepr> FromJsSourceRepr(
      const JsSourceRepr &source_repr, BabelParseRequest parse_request,
      absl::Duration timeout, std::optional<int> recursion_depth_limit,
      Babel &babel);

  static absl::StatusOr<JsAstRepr> FromJsAstStringRepr(
      const JsAstStringRepr &ast_string_repr,
      std::optional<int> recursion_depth_limit);

  static absl::StatusOr<JsAstRepr> FromJsirRepr(const JsirRepr& ir_repr);
};

struct ToJsirRepr {
  static absl::StatusOr<JsirRepr> FromJsSourceRepr(
      const JsSourceRepr& source_repr, BabelParseRequest parse_request,
      absl::Duration timeout, std::optional<int> recursion_depth_limit,
      Babel& babel, mlir::MLIRContext& mlir_context);

  static absl::StatusOr<JsirRepr> FromJsAstStringRepr(
      const JsAstStringRepr& ast_string_repr,
      std::optional<int> recursion_depth_limit,
      mlir::MLIRContext& mlir_context);

  static absl::StatusOr<JsirRepr> FromJsAstRepr(
      const JsAstRepr& ast_repr, mlir::MLIRContext& mlir_context);
};

}  // namespace maldoca

#endif  // MALDOCA_JS_DRIVER_CONVERSION_H_
