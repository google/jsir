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

#include "maldoca/js/ir/conversion/ast_to_jsir.h"

#include <functional>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/ir/ir.h"
#include "maldoca/js/ir/trivia.h"

namespace maldoca {

JsirCommentAttrInterface AstToJsir::VisitCommentAttr(mlir::OpBuilder& builder,
                                                     const JsComment* node) {
  JsirLocationAttr loc = nullptr;
  if (node->loc().has_value()) {
    loc = GetJsirLocationAttr(builder.getContext(), node->loc().value(),
                              node->start(), node->end(),
                              /*scope_uid=*/std::nullopt);
  }

  switch (node->comment_type()) {
    case JsCommentType::kCommentLine:
      return JsirCommentLineAttr::get(
          builder.getContext(), loc,
          mlir::StringAttr::get(builder.getContext(), node->value()));
    case JsCommentType::kCommentBlock:
      return JsirCommentBlockAttr::get(
          builder.getContext(), loc,
          mlir::StringAttr::get(builder.getContext(), node->value()));
  }
}

JsirInterpreterDirectiveAttr AstToJsir::VisitInterpreterDirectiveAttr(
    mlir::OpBuilder& builder, const JsInterpreterDirective* node) {
  auto loc = GetJsirTriviaAttr(builder.getContext(), *node);
  mlir::StringAttr mlir_value = builder.getStringAttr(node->value());
  return JsirInterpreterDirectiveAttr::get(builder.getContext(), loc,
                                           mlir_value);
}

JsirDirectiveLiteralExtraAttr AstToJsir::VisitDirectiveLiteralExtraAttr(
    mlir::OpBuilder& builder, const JsDirectiveLiteralExtra* node) {
  mlir::StringAttr mlir_raw = builder.getStringAttr(node->raw());
  mlir::StringAttr mlir_raw_value = builder.getStringAttr(node->raw_value());
  return JsirDirectiveLiteralExtraAttr::get(builder.getContext(), mlir_raw,
                                            mlir_raw_value);
}

JsirIdentifierAttr AstToJsir::VisitIdentifierAttr(mlir::OpBuilder& builder,
                                                  const JsIdentifier* node) {
  auto loc = GetJsirTriviaAttr(builder.getContext(), *node);
  mlir::StringAttr mlir_name = builder.getStringAttr(node->name());
  return JsirIdentifierAttr::get(builder.getContext(), loc, mlir_name);
}

JsirPrivateNameAttr AstToJsir::VisitPrivateNameAttr(mlir::OpBuilder& builder,
                                                    const JsPrivateName* node) {
  auto loc = GetJsirTriviaAttr(builder.getContext(), *node);
  JsirIdentifierAttr mlir_id = VisitIdentifierAttr(builder, node->id());
  return JsirPrivateNameAttr::get(builder.getContext(), loc, mlir_id);
}

JsirRegExpLiteralExtraAttr AstToJsir::VisitRegExpLiteralExtraAttr(
    mlir::OpBuilder& builder, const JsRegExpLiteralExtra* node) {
  mlir::StringAttr mlir_raw = builder.getStringAttr(node->raw());
  return JsirRegExpLiteralExtraAttr::get(builder.getContext(), mlir_raw);
}

JsirStringLiteralExtraAttr AstToJsir::VisitStringLiteralExtraAttr(
    mlir::OpBuilder& builder, const JsStringLiteralExtra* node) {
  mlir::StringAttr mlir_raw = builder.getStringAttr(node->raw());
  mlir::StringAttr mlir_raw_value = builder.getStringAttr(node->raw_value());
  return JsirStringLiteralExtraAttr::get(builder.getContext(), mlir_raw,
                                         mlir_raw_value);
}

JsirStringLiteralAttr AstToJsir::VisitStringLiteralAttr(
    mlir::OpBuilder& builder, const JsStringLiteral* node) {
  auto loc = GetJsirTriviaAttr(builder.getContext(), *node);
  mlir::StringAttr mlir_value = builder.getStringAttr(node->value());
  JsirStringLiteralExtraAttr mlir_extra;
  if (node->extra().has_value()) {
    mlir_extra = VisitStringLiteralExtraAttr(builder, node->extra().value());
  }
  return JsirStringLiteralAttr::get(builder.getContext(), loc, mlir_value,
                                    mlir_extra);
}

JsirNumericLiteralExtraAttr AstToJsir::VisitNumericLiteralExtraAttr(
    mlir::OpBuilder& builder, const JsNumericLiteralExtra* node) {
  mlir::StringAttr mlir_raw = builder.getStringAttr(node->raw());
  mlir::FloatAttr mlir_raw_value = builder.getF64FloatAttr(node->raw_value());
  return JsirNumericLiteralExtraAttr::get(builder.getContext(), mlir_raw,
                                          mlir_raw_value);
}

JsirNumericLiteralAttr AstToJsir::VisitNumericLiteralAttr(
    mlir::OpBuilder& builder, const JsNumericLiteral* node) {
  auto loc = GetJsirTriviaAttr(builder.getContext(), *node);
  mlir::FloatAttr mlir_value = builder.getF64FloatAttr(node->value());
  JsirNumericLiteralExtraAttr mlir_extra;
  if (node->extra().has_value()) {
    mlir_extra = VisitNumericLiteralExtraAttr(builder, node->extra().value());
  }
  return JsirNumericLiteralAttr::get(builder.getContext(), loc, mlir_value,
                                     mlir_extra);
}

JsirBigIntLiteralExtraAttr AstToJsir::VisitBigIntLiteralExtraAttr(
    mlir::OpBuilder& builder, const JsBigIntLiteralExtra* node) {
  mlir::StringAttr mlir_raw = builder.getStringAttr(node->raw());
  mlir::StringAttr mlir_raw_value = builder.getStringAttr(node->raw_value());
  return JsirBigIntLiteralExtraAttr::get(builder.getContext(), mlir_raw,
                                         mlir_raw_value);
}

JsirBigIntLiteralAttr AstToJsir::VisitBigIntLiteralAttr(
    mlir::OpBuilder& builder, const JsBigIntLiteral* node) {
  mlir::StringAttr mlir_value = builder.getStringAttr(node->value());
  JsirBigIntLiteralExtraAttr mlir_extra;
  if (node->extra().has_value()) {
    mlir_extra = VisitBigIntLiteralExtraAttr(builder, node->extra().value());
  }
  return JsirBigIntLiteralAttr::get(builder.getContext(), mlir_value,
                                    mlir_extra);
}

JsirImportSpecifierAttr AstToJsir::VisitImportSpecifierAttr(
    mlir::OpBuilder& builder, const JsImportSpecifier* node) {
  auto loc = GetJsirTriviaAttr(builder.getContext(), *node);
  mlir::Attribute mlir_imported;
  if (std::holds_alternative<const JsIdentifier*>(node->imported())) {
    auto* imported = std::get<const JsIdentifier*>(node->imported());
    mlir_imported = VisitIdentifierAttr(builder, imported);
  } else if (std::holds_alternative<const JsStringLiteral*>(node->imported())) {
    auto* imported = std::get<const JsStringLiteral*>(node->imported());
    mlir_imported = VisitStringLiteralAttr(builder, imported);
  } else {
    LOG(FATAL) << "Unreachable code.";
  }
  JsirIdentifierAttr mlir_local = VisitIdentifierAttr(builder, node->local());
  return JsirImportSpecifierAttr::get(builder.getContext(), loc, mlir_imported,
                                      mlir_local);
}

JsirImportDefaultSpecifierAttr AstToJsir::VisitImportDefaultSpecifierAttr(
    mlir::OpBuilder& builder, const JsImportDefaultSpecifier* node) {
  auto loc = GetJsirTriviaAttr(builder.getContext(), *node);
  JsirIdentifierAttr mlir_local = VisitIdentifierAttr(builder, node->local());
  return JsirImportDefaultSpecifierAttr::get(builder.getContext(), loc,
                                             mlir_local);
}

JsirImportNamespaceSpecifierAttr AstToJsir::VisitImportNamespaceSpecifierAttr(
    mlir::OpBuilder& builder, const JsImportNamespaceSpecifier* node) {
  auto loc = GetJsirTriviaAttr(builder.getContext(), *node);
  JsirIdentifierAttr mlir_local = VisitIdentifierAttr(builder, node->local());
  return JsirImportNamespaceSpecifierAttr::get(builder.getContext(), loc,
                                               mlir_local);
}

JsirImportAttributeAttr AstToJsir::VisitImportAttributeAttr(
    mlir::OpBuilder& builder, const JsImportAttribute* node) {
  JsirIdentifierAttr mlir_key = VisitIdentifierAttr(builder, node->key());
  JsirStringLiteralAttr mlir_value =
      VisitStringLiteralAttr(builder, node->value());
  return JsirImportAttributeAttr::get(builder.getContext(), mlir_key,
                                      mlir_value);
}

void AstToJsir::AppendNewBlockAndPopulate(mlir::OpBuilder& builder,
                                          mlir::Region& region,
                                          std::function<void()> populate) {
  // Save insertion point.
  // Will revert at the end.
  mlir::OpBuilder::InsertionGuard insertion_guard(builder);

  // Insert new block and point builder to it.
  mlir::Block& block = region.emplaceBlock();
  builder.setInsertionPointToStart(&block);

  populate();
}

}  // namespace maldoca
