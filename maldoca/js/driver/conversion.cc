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

#include "maldoca/js/driver/conversion.h"

#include <memory>
#include <optional>
#include <utility>

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "maldoca/base/status_macros.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/ast/ast_util.h"
#include "maldoca/js/babel/babel.h"
#include "maldoca/js/driver/driver.h"
#include "maldoca/js/ir/conversion/utils.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {

// =============================================================================
// Lowering conversions
// =============================================================================

// -----------------------------------------------------------------------------
// Source -> AST string
// -----------------------------------------------------------------------------

absl::StatusOr<JsAstStringRepr> ToJsAstStringRepr::FromJsSourceRepr(
    absl::string_view source, BabelParseRequest parse_request,
    absl::Duration timeout, Babel &babel) {
  MALDOCA_ASSIGN_OR_RETURN(BabelParseResult parse_result,
                           babel.Parse(source, parse_request, timeout));
  return JsAstStringRepr{std::move(parse_result.ast_string)};
}

// -----------------------------------------------------------------------------
// AST string -> AST
// -----------------------------------------------------------------------------

absl::StatusOr<JsAstRepr> ToJsAstRepr::FromJsAstStringRepr(
    const BabelAstString &ast_string,
    std::optional<int> recursion_depth_limit) {
  MALDOCA_ASSIGN_OR_RETURN(
      std::unique_ptr<JsFile> ast,
      GetFileAstFromAstString(ast_string, recursion_depth_limit));
  return JsAstRepr{std::move(ast), ast_string.scopes()};
}

// -----------------------------------------------------------------------------
// AST -> HIR
// -----------------------------------------------------------------------------

absl::StatusOr<JsHirRepr> ToJsHirRepr::FromJsAstRepr(
    const JsFile &ast, const BabelScopes &scopes,
    mlir::MLIRContext &mlir_context) {
  MALDOCA_ASSIGN_OR_RETURN(mlir::OwningOpRef<JsirFileOp> op,
                   AstToJshirFile(ast, mlir_context));
  return JsHirRepr{std::move(op), scopes};
}

// -----------------------------------------------------------------------------
// HIR -> LIR
// -----------------------------------------------------------------------------

JsLirRepr ToJsLirRepr::FromJsHirRepr(const JsHirRepr &hir_repr) {
  mlir::OwningOpRef<JsirFileOp> lir_op = JshirFileToJslir(hir_repr.op.get());
  return JsLirRepr{std::move(lir_op), hir_repr.scopes};
}

// -----------------------------------------------------------------------------
// Source -> AST string -> AST
// -----------------------------------------------------------------------------

absl::StatusOr<JsAstRepr> ToJsAstRepr::FromJsSourceRepr(
    absl::string_view source, BabelParseRequest parse_request,
    absl::Duration timeout, std::optional<int> recursion_depth_limit,
    Babel &babel) {
  MALDOCA_ASSIGN_OR_RETURN(JsAstStringRepr ast_string,
                           ToJsAstStringRepr::FromJsSourceRepr(
                               source, parse_request, timeout, babel));
  return ToJsAstRepr::FromJsAstStringRepr(ast_string.ast_string,
                                          recursion_depth_limit);
}

// -----------------------------------------------------------------------------
// Source -> AST string -> AST -> HIR
// -----------------------------------------------------------------------------

absl::StatusOr<JsHirRepr> ToJsHirRepr::FromJsSourceRepr(
    absl::string_view source, BabelParseRequest parse_request,
    absl::Duration timeout, std::optional<int> recursion_depth_limit,
    Babel &babel, mlir::MLIRContext &mlir_context) {
  MALDOCA_ASSIGN_OR_RETURN(JsAstRepr ast, ToJsAstRepr::FromJsSourceRepr(
                                              source, parse_request, timeout,
                                              recursion_depth_limit, babel));
  return ToJsHirRepr::FromJsAstRepr(*ast.ast, ast.scopes, mlir_context);
}

// -----------------------------------------------------------------------------
// Source -> AST string -> AST -> HIR -> LIR
// -----------------------------------------------------------------------------

absl::StatusOr<JsLirRepr> ToJsLirRepr::FromJsSourceRepr(
    absl::string_view source, BabelParseRequest parse_request,
    absl::Duration timeout, std::optional<int> recursion_depth_limit,
    Babel &babel, mlir::MLIRContext &mlir_context) {
  MALDOCA_ASSIGN_OR_RETURN(
      JsHirRepr hir, ToJsHirRepr::FromJsSourceRepr(
                         source, parse_request, timeout, recursion_depth_limit,
                         babel, mlir_context));
  return ToJsLirRepr::FromJsHirRepr(hir);
}

// -----------------------------------------------------------------------------
// AST string -> AST -> HIR
// -----------------------------------------------------------------------------

absl::StatusOr<JsHirRepr> ToJsHirRepr::FromJsAstStringRepr(
    const BabelAstString &ast_string, std::optional<int> recursion_depth_limit,
    mlir::MLIRContext &mlir_context) {
  MALDOCA_ASSIGN_OR_RETURN(
      JsAstRepr ast,
      ToJsAstRepr::FromJsAstStringRepr(ast_string, recursion_depth_limit));
  return ToJsHirRepr::FromJsAstRepr(*ast.ast, ast.scopes, mlir_context);
}

// ----------------------------------------------------------------------------
// AST string -> AST -> HIR -> LIR
// -----------------------------------------------------------------------------

absl::StatusOr<JsLirRepr> ToJsLirRepr::FromJsAstStringRepr(
    const BabelAstString &ast_string, std::optional<int> recursion_depth_limit,
    mlir::MLIRContext &mlir_context) {
  MALDOCA_ASSIGN_OR_RETURN(
      JsHirRepr hir, ToJsHirRepr::FromJsAstStringRepr(
                         ast_string, recursion_depth_limit, mlir_context));
  return ToJsLirRepr::FromJsHirRepr(hir);
}

// ----------------------------------------------------------------------------
// AST -> HIR -> LIR
// -----------------------------------------------------------------------------

absl::StatusOr<JsLirRepr> ToJsLirRepr::FromJsAstRepr(
    const JsFile &ast, const BabelScopes &scopes,
    mlir::MLIRContext &mlir_context) {
  MALDOCA_ASSIGN_OR_RETURN(
      JsHirRepr hir, ToJsHirRepr::FromJsAstRepr(ast, scopes, mlir_context));
  return ToJsLirRepr::FromJsHirRepr(hir);
}

// =============================================================================
// Lifting conversions
// =============================================================================

// -----------------------------------------------------------------------------
// LIR -> HIR
// -----------------------------------------------------------------------------

JsHirRepr ToJsHirRepr::FromJsLirRepr(const JsLirRepr &lir_repr) {
  mlir::OwningOpRef<JsirFileOp> hir_op = JslirFileToJshir(lir_repr.op.get());
  return JsHirRepr{std::move(hir_op), lir_repr.scopes};
}

// -----------------------------------------------------------------------------
// HIR -> AST
// -----------------------------------------------------------------------------

absl::StatusOr<JsAstRepr> ToJsAstRepr::FromJsHirRepr(
    const JsHirRepr &hir_repr) {
  MALDOCA_ASSIGN_OR_RETURN(std::unique_ptr<JsFile> ast,
                           JshirFileToAst(hir_repr.op.get()));
  return JsAstRepr{std::move(ast), hir_repr.scopes};
}

// -----------------------------------------------------------------------------
// AST -> AST string
// -----------------------------------------------------------------------------

absl::StatusOr<JsAstStringRepr> ToJsAstStringRepr::FromJsAstRepr(
    const JsFile &ast, const BabelScopes &scopes) {
  BabelAstString ast_string = GetAstStringFromFileAst(ast);
  *ast_string.mutable_scopes() = scopes;
  return JsAstStringRepr{std::move(ast_string)};
}

// -----------------------------------------------------------------------------
// AST string -> Source
// -----------------------------------------------------------------------------

absl::StatusOr<JsSourceRepr> ToJsSourceRepr::FromJsAstStringRepr(
    const BabelAstString &ast_string, BabelGenerateOptions generate_options,
    absl::Duration timeout, Babel &babel) {
  MALDOCA_ASSIGN_OR_RETURN(
      BabelGenerateResult generate_result,
      babel.Generate(ast_string, generate_options, timeout));
  return JsSourceRepr{std::move(generate_result.source_code)};
}

// -----------------------------------------------------------------------------
// LIR -> HIR -> AST
// -----------------------------------------------------------------------------

absl::StatusOr<JsAstRepr> ToJsAstRepr::FromJsLirRepr(
    const JsLirRepr &lir_repr) {
  JsHirRepr hir = ToJsHirRepr::FromJsLirRepr(lir_repr);
  return ToJsAstRepr::FromJsHirRepr(hir);
}

// -----------------------------------------------------------------------------
// LIR -> HIR -> AST -> AST string
// -----------------------------------------------------------------------------

absl::StatusOr<JsAstStringRepr> ToJsAstStringRepr::FromJsLirRepr(
    const JsLirRepr &lir_repr) {
  MALDOCA_ASSIGN_OR_RETURN(JsAstRepr ast, ToJsAstRepr::FromJsLirRepr(lir_repr));
  return ToJsAstStringRepr::FromJsAstRepr(*ast.ast, ast.scopes);
}

// -----------------------------------------------------------------------------
// LIR -> HIR -> AST -> AST string -> Source
// -----------------------------------------------------------------------------

absl::StatusOr<JsSourceRepr> ToJsSourceRepr::FromJsLirRepr(
    const JsLirRepr &lir_repr, BabelGenerateOptions generate_options,
    absl::Duration timeout, Babel &babel) {
  MALDOCA_ASSIGN_OR_RETURN(JsAstStringRepr ast_string,
                   ToJsAstStringRepr::FromJsLirRepr(lir_repr));
  return ToJsSourceRepr::FromJsAstStringRepr(ast_string.ast_string,
                                             generate_options, timeout, babel);
}

// -----------------------------------------------------------------------------
// HIR -> AST -> AST string
// -----------------------------------------------------------------------------

absl::StatusOr<JsAstStringRepr> ToJsAstStringRepr::FromJsHirRepr(
    const JsHirRepr &hir_repr) {
  MALDOCA_ASSIGN_OR_RETURN(JsAstRepr ast, ToJsAstRepr::FromJsHirRepr(hir_repr));
  return ToJsAstStringRepr::FromJsAstRepr(*ast.ast, ast.scopes);
}

// -----------------------------------------------------------------------------
// HIR -> AST -> AST string -> Source
// -----------------------------------------------------------------------------

absl::StatusOr<JsSourceRepr> ToJsSourceRepr::FromJsHirRepr(
    const JsHirRepr &hir_repr, BabelGenerateOptions generate_options,
    absl::Duration timeout, Babel &babel) {
  MALDOCA_ASSIGN_OR_RETURN(JsAstStringRepr ast_string,
                   ToJsAstStringRepr::FromJsHirRepr(hir_repr));
  return ToJsSourceRepr::FromJsAstStringRepr(ast_string.ast_string,
                                             generate_options, timeout, babel);
}

// -----------------------------------------------------------------------------
// AST -> AST string -> Source
// -----------------------------------------------------------------------------

absl::StatusOr<JsSourceRepr> ToJsSourceRepr::FromJsAstRepr(
    const JsFile &ast, BabelGenerateOptions generate_options,
    absl::Duration timeout, Babel &babel) {
  BabelScopes dummy_scopes;
  MALDOCA_ASSIGN_OR_RETURN(JsAstStringRepr ast_string,
                   ToJsAstStringRepr::FromJsAstRepr(ast, dummy_scopes));
  return ToJsSourceRepr::FromJsAstStringRepr(ast_string.ast_string,
                                             generate_options, timeout, babel);
}

}  // namespace maldoca
