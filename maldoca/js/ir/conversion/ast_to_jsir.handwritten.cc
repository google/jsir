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

JshirBreakStatementOp AstToJsir::VisitBreakStatement(
    mlir::OpBuilder& builder, const JsBreakStatement* node) {
  JsirIdentifierAttr mlir_label;
  if (node->label().has_value()) {
    mlir_label = VisitIdentifierAttr(builder, node->label().value());
  }
  return CreateStmt<JshirBreakStatementOp>(builder, node, mlir_label);
}

JshirContinueStatementOp AstToJsir::VisitContinueStatement(
    mlir::OpBuilder& builder, const JsContinueStatement* node) {
  JsirIdentifierAttr mlir_label;
  if (node->label().has_value()) {
    mlir_label = VisitIdentifierAttr(builder, node->label().value());
  }
  return CreateStmt<JshirContinueStatementOp>(builder, node, mlir_label);
}

JsirObjectExpressionOp AstToJsir::VisitObjectExpression(
    mlir::OpBuilder& builder, const JsObjectExpression* node) {
  auto op = CreateExpr<JsirObjectExpressionOp>(builder, node);
  mlir::Region& mlir_properties_region = op.getRegion();
  AppendNewBlockAndPopulate(builder, mlir_properties_region, [&] {
    std::vector<mlir::Value> mlir_properties;
    for (const auto& property : *node->properties_()) {
      mlir::Value mlir_property;
      switch (property.index()) {
        case 0: {
          const JsObjectProperty* property_object_property =
              std::get<0>(property).get();
          mlir_property =
              VisitObjectProperty(builder, property_object_property);
          break;
        }
        case 1: {
          const JsObjectMethod* property_object_method =
              std::get<1>(property).get();
          mlir_property = VisitObjectMethod(builder, property_object_method);
          break;
        }
        case 2: {
          const JsSpreadElement* property_spread_element =
              std::get<2>(property).get();
          mlir_property = VisitSpreadElement(builder, property_spread_element);
          break;
        }
        default:
          LOG(FATAL) << "Unreachable code.";
      }
      mlir_properties.push_back(mlir_property);
    }
    CreateStmt<JsirExprsRegionEndOp>(builder, nullptr, mlir_properties);
  });
  return op;
}

JsirParenthesizedExpressionOp AstToJsir::VisitParenthesizedExpression(
    mlir::OpBuilder& builder, const JsParenthesizedExpression* node) {
  mlir::Value mlir_expression = VisitExpression(builder, node->expression());
  return CreateExpr<JsirParenthesizedExpressionOp>(builder, node,
                                                   mlir_expression);
}

JsirParenthesizedExpressionRefOp AstToJsir::VisitParenthesizedExpressionRef(
    mlir::OpBuilder& builder, const JsParenthesizedExpression* node) {
  mlir::Value mlir_expression = [&]() -> mlir::Value {
    if (auto* lval = dynamic_cast<const JsLVal*>(node->expression())) {
      return VisitLValRef(builder, lval);
    } else {
      // TODO(b/293174026): Disallow this.
      mlir::emitError(GetJsirTriviaAttr(builder.getContext(), *node),
                      "lvalue expected");
      return VisitExpression(builder, node->expression());
    }
  }();
  return CreateExpr<JsirParenthesizedExpressionRefOp>(builder, node,
                                                      mlir_expression);
}

JsirClassPrivateMethodOp AstToJsir::VisitClassPrivateMethod(
    mlir::OpBuilder& builder, const JsClassPrivateMethod* node) {
  JsirIdentifierAttr mlir_id;
  if (node->id().has_value()) {
    mlir_id = VisitIdentifierAttr(builder, node->id().value());
  }
  mlir::BoolAttr mlir_generator = builder.getBoolAttr(node->generator());
  mlir::BoolAttr mlir_async = builder.getBoolAttr(node->async());
  JsirPrivateNameAttr mlir_key = VisitPrivateNameAttr(builder, node->key());
  mlir::StringAttr mlir_kind = builder.getStringAttr(node->kind());
  mlir::BoolAttr mlir_static = builder.getBoolAttr(node->static_());
  auto op = CreateStmt<JsirClassPrivateMethodOp>(
      builder, node, mlir_id, mlir_generator, mlir_async, mlir_key, mlir_kind,
      mlir_static);
  mlir::Region& mlir_params_region = op.getParams();
  AppendNewBlockAndPopulate(builder, mlir_params_region, [&] {
    std::vector<mlir::Value> mlir_params;
    for (const auto& param : *node->params()) {
      mlir::Value mlir_param = VisitPatternRef(builder, param.get());
      mlir_params.push_back(mlir_param);
    }
    CreateStmt<JsirExprsRegionEndOp>(builder, nullptr, mlir_params);
  });
  mlir::Region& mlir_body_region = op.getBody();
  AppendNewBlockAndPopulate(builder, mlir_body_region, [&] {
    VisitBlockStatement(builder, node->body());
  });
  return op;
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

JsirExportSpecifierOp AstToJsir::VisitExportSpecifier(
    mlir::OpBuilder& builder, const JsExportSpecifier* node) {
  mlir::Attribute mlir_exported;
  if (std::holds_alternative<const JsIdentifier*>(node->exported())) {
    auto* exported = std::get<const JsIdentifier*>(node->exported());
    mlir_exported = VisitIdentifierAttr(builder, exported);
  } else if (std::holds_alternative<const JsStringLiteral*>(
                 node->exported())) {
    auto* exported = std::get<const JsStringLiteral*>(node->exported());
    mlir_exported = VisitStringLiteralAttr(builder, exported);
  } else {
    LOG(FATAL) << "Unreachable code.";
  }

  // An identifier `local` refers to the exported binding, so it lowers to an
  // lvalue (`jsir.identifier_ref`). A string literal `local` (only valid in
  // re-exports) has no lvalue form, so it stays `jsir.string_literal`.
  mlir::Value mlir_local;
  if (node->local().has_value()) {
    auto local = node->local().value();
    if (std::holds_alternative<const JsIdentifier*>(local)) {
      mlir_local =
          VisitIdentifierRef(builder, std::get<const JsIdentifier*>(local));
    } else if (std::holds_alternative<const JsStringLiteral*>(local)) {
      mlir_local =
          VisitStringLiteral(builder, std::get<const JsStringLiteral*>(local));
    } else {
      LOG(FATAL) << "Unreachable code.";
    }
  }
  return CreateStmt<JsirExportSpecifierOp>(builder, node, mlir_exported,
                                           mlir_local);
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
