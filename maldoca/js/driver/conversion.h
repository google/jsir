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
      const BabelAstString &ast_string, BabelGenerateOptions generate_options,
      absl::Duration timeout, Babel &babel);

  static absl::StatusOr<JsSourceRepr> FromJsAstRepr(
      const JsFile &ast, BabelGenerateOptions generate_options,
      absl::Duration timeout, Babel &babel);

  static absl::StatusOr<JsSourceRepr> FromJsHirRepr(
      const JsHirRepr &hir_repr, BabelGenerateOptions generate_options,
      absl::Duration timeout, Babel &babel);
};

struct ToJsAstStringRepr {
  static absl::StatusOr<JsAstStringRepr> FromJsSourceRepr(
      absl::string_view source, BabelParseRequest parse_request,
      absl::Duration timeout, Babel &babel);

  static absl::StatusOr<JsAstStringRepr> FromJsAstRepr(
      const JsFile &ast, const BabelScopes &scopes);

  static absl::StatusOr<JsAstStringRepr> FromJsHirRepr(
      const JsHirRepr &hir_repr);
};

struct ToJsAstRepr {
  static absl::StatusOr<JsAstRepr> FromJsSourceRepr(
      absl::string_view source, BabelParseRequest parse_request,
      absl::Duration timeout, std::optional<int> recursion_depth_limit,
      Babel &babel);

  static absl::StatusOr<JsAstRepr> FromJsAstStringRepr(
      const BabelAstString &ast_string,
      std::optional<int> recursion_depth_limit);

  static absl::StatusOr<JsAstRepr> FromJsHirRepr(const JsHirRepr &hir_repr);
};

struct ToJsHirRepr {
  static absl::StatusOr<JsHirRepr> FromJsSourceRepr(
      absl::string_view source, BabelParseRequest parse_request,
      absl::Duration timeout, std::optional<int> recursion_depth_limit,
      Babel &babel, mlir::MLIRContext &mlir_context);

  static absl::StatusOr<JsHirRepr> FromJsAstStringRepr(
      const BabelAstString &ast_string,
      std::optional<int> recursion_depth_limit,
      mlir::MLIRContext &mlir_context);

  static absl::StatusOr<JsHirRepr> FromJsAstRepr(
      const maldoca::JsFile &ast, const BabelScopes &scopes,
      mlir::MLIRContext &mlir_context);
};

struct ToJsLirRepr {
  static absl::StatusOr<JsLirRepr> FromJsSourceRepr(
      absl::string_view source, BabelParseRequest parse_request,
      absl::Duration timeout, std::optional<int> recursion_depth_limit,
      Babel &babel, mlir::MLIRContext &mlir_context);

  static absl::StatusOr<JsLirRepr> FromJsAstStringRepr(
      const BabelAstString &ast_string,
      std::optional<int> recursion_depth_limit,
      mlir::MLIRContext &mlir_context);

  static absl::StatusOr<JsLirRepr> FromJsAstRepr(
      const JsFile &ast, const BabelScopes &scopes,
      mlir::MLIRContext &mlir_context);
};

}  // namespace maldoca

#endif  // MALDOCA_JS_DRIVER_CONVERSION_H_
