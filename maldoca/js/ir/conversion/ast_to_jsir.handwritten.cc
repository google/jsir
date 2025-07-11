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
#include "absl/types/optional.h"
#include "absl/types/variant.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/ir/ir.h"
#include "maldoca/js/ir/trivia.h"

namespace maldoca {

JsirCommentAttrInterface AstToJsir::VisitCommentAttr(const JsComment *node) {
  auto loc =
      GetJsirLocationAttr(builder_.getContext(), node->loc(), node->start(),
                          node->end(), /*scope_uid=*/std::nullopt);
  switch (node->comment_type()) {
    case JsCommentType::kCommentLine:
      return JsirCommentLineAttr::get(
          builder_.getContext(), loc,
          mlir::StringAttr::get(builder_.getContext(), node->value()));
    case JsCommentType::kCommentBlock:
      return JsirCommentBlockAttr::get(
          builder_.getContext(), loc,
          mlir::StringAttr::get(builder_.getContext(), node->value()));
  }
}

JsirInterpreterDirectiveAttr AstToJsir::VisitInterpreterDirectiveAttr(
    const JsInterpreterDirective *node) {
  auto loc = GetJsirTriviaAttr(builder_.getContext(), *node);
  mlir::StringAttr mlir_value = builder_.getStringAttr(node->value());
  return JsirInterpreterDirectiveAttr::get(builder_.getContext(), loc,
                                           mlir_value);
}

JsirDirectiveLiteralExtraAttr AstToJsir::VisitDirectiveLiteralExtraAttr(
    const JsDirectiveLiteralExtra *node) {
  mlir::StringAttr mlir_raw = builder_.getStringAttr(node->raw());
  mlir::StringAttr mlir_raw_value = builder_.getStringAttr(node->raw_value());
  return JsirDirectiveLiteralExtraAttr::get(builder_.getContext(), mlir_raw,
                                            mlir_raw_value);
}

JsirIdentifierAttr AstToJsir::VisitIdentifierAttr(const JsIdentifier *node) {
  auto loc = GetJsirTriviaAttr(builder_.getContext(), *node);
  mlir::StringAttr mlir_name = builder_.getStringAttr(node->name());
  return JsirIdentifierAttr::get(builder_.getContext(), loc, mlir_name);
}

JsirPrivateNameAttr AstToJsir::VisitPrivateNameAttr(const JsPrivateName *node) {
  auto loc = GetJsirTriviaAttr(builder_.getContext(), *node);
  JsirIdentifierAttr mlir_id = VisitIdentifierAttr(node->id());
  return JsirPrivateNameAttr::get(builder_.getContext(), loc, mlir_id);
}

JsirRegExpLiteralExtraAttr AstToJsir::VisitRegExpLiteralExtraAttr(
    const JsRegExpLiteralExtra *node) {
  mlir::StringAttr mlir_raw = builder_.getStringAttr(node->raw());
  return JsirRegExpLiteralExtraAttr::get(builder_.getContext(), mlir_raw);
}

JsirStringLiteralExtraAttr AstToJsir::VisitStringLiteralExtraAttr(
    const JsStringLiteralExtra *node) {
  mlir::StringAttr mlir_raw = builder_.getStringAttr(node->raw());
  mlir::StringAttr mlir_raw_value = builder_.getStringAttr(node->raw_value());
  return JsirStringLiteralExtraAttr::get(builder_.getContext(), mlir_raw,
                                         mlir_raw_value);
}

JsirStringLiteralAttr AstToJsir::VisitStringLiteralAttr(
    const JsStringLiteral *node) {
  auto loc = GetJsirTriviaAttr(builder_.getContext(), *node);
  mlir::StringAttr mlir_value = builder_.getStringAttr(node->value());
  JsirStringLiteralExtraAttr mlir_extra;
  if (node->extra().has_value()) {
    mlir_extra = VisitStringLiteralExtraAttr(node->extra().value());
  }
  return JsirStringLiteralAttr::get(builder_.getContext(), loc, mlir_value,
                                    mlir_extra);
}

JsirNumericLiteralExtraAttr AstToJsir::VisitNumericLiteralExtraAttr(
    const JsNumericLiteralExtra *node) {
  mlir::StringAttr mlir_raw = builder_.getStringAttr(node->raw());
  mlir::FloatAttr mlir_raw_value = builder_.getF64FloatAttr(node->raw_value());
  return JsirNumericLiteralExtraAttr::get(builder_.getContext(), mlir_raw,
                                          mlir_raw_value);
}

JsirNumericLiteralAttr AstToJsir::VisitNumericLiteralAttr(
    const JsNumericLiteral *node) {
  auto loc = GetJsirTriviaAttr(builder_.getContext(), *node);
  mlir::FloatAttr mlir_value = builder_.getF64FloatAttr(node->value());
  JsirNumericLiteralExtraAttr mlir_extra;
  if (node->extra().has_value()) {
    mlir_extra = VisitNumericLiteralExtraAttr(node->extra().value());
  }
  return JsirNumericLiteralAttr::get(builder_.getContext(), loc, mlir_value,
                                     mlir_extra);
}

JsirBigIntLiteralExtraAttr AstToJsir::VisitBigIntLiteralExtraAttr(
    const JsBigIntLiteralExtra *node) {
  mlir::StringAttr mlir_raw = builder_.getStringAttr(node->raw());
  mlir::StringAttr mlir_raw_value = builder_.getStringAttr(node->raw_value());
  return JsirBigIntLiteralExtraAttr::get(builder_.getContext(), mlir_raw,
                                         mlir_raw_value);
}

JsirBigIntLiteralAttr AstToJsir::VisitBigIntLiteralAttr(
    const JsBigIntLiteral *node) {
  mlir::StringAttr mlir_value = builder_.getStringAttr(node->value());
  JsirBigIntLiteralExtraAttr mlir_extra;
  if (node->extra().has_value()) {
    mlir_extra = VisitBigIntLiteralExtraAttr(node->extra().value());
  }
  return JsirBigIntLiteralAttr::get(builder_.getContext(), mlir_value,
                                    mlir_extra);
}

JshirBreakStatementOp AstToJsir::VisitBreakStatement(
    const JsBreakStatement *node) {
  JsirIdentifierAttr mlir_label;
  if (node->label().has_value()) {
    mlir_label = VisitIdentifierAttr(node->label().value());
  }
  return CreateStmt<JshirBreakStatementOp>(node, mlir_label);
}

JshirContinueStatementOp AstToJsir::VisitContinueStatement(
    const JsContinueStatement *node) {
  JsirIdentifierAttr mlir_label;
  if (node->label().has_value()) {
    mlir_label = VisitIdentifierAttr(node->label().value());
  }
  return CreateStmt<JshirContinueStatementOp>(node, mlir_label);
}

JshirForStatementOp AstToJsir::VisitForStatement(const JsForStatement *node) {
  auto op = CreateStmt<JshirForStatementOp>(node);
  mlir::Region &init_region = op.getInit();
  if (node->init().has_value()) {
    AppendNewBlockAndPopulate(init_region, [&] {
      auto init = node->init().value();
      if (std::holds_alternative<const JsVariableDeclaration *>(init)) {
        auto *init_variable_declaration =
            std::get<const JsVariableDeclaration *>(init);
        VisitVariableDeclaration(init_variable_declaration);
      } else if (std::holds_alternative<const JsExpression *>(init)) {
        auto *init_expression = std::get<const JsExpression *>(init);
        mlir::Value mlir_init = VisitExpression(init_expression);
        CreateStmt<JsirExprRegionEndOp>(init_expression, mlir_init);
      }
    });
  }
  mlir::Region &test_region = op.getTest();
  if (node->test().has_value()) {
    const JsExpression *test = node->test().value();
    AppendNewBlockAndPopulate(test_region, [&] {
      mlir::Value mlir_test = VisitExpression(test);
      CreateStmt<JsirExprRegionEndOp>(test, mlir_test);
    });
  }
  mlir::Region &update_region = op.getUpdate();
  if (node->update().has_value()) {
    const JsExpression *update = node->update().value();
    AppendNewBlockAndPopulate(update_region, [&] {
      mlir::Value mlir_update = VisitExpression(update);
      CreateStmt<JsirExprRegionEndOp>(update, mlir_update);
    });
  }
  mlir::Region &body_region = op.getBody();
  AppendNewBlockAndPopulate(body_region, [&] { VisitStatement(node->body()); });
  return op;
}

struct ForInOfLeft {
  std::optional<JsirForInOfDeclarationAttr> declaration_attr;
  const JsLVal *lval;
};

static absl::StatusOr<ForInOfLeft> GetForInOfLeft(
    mlir::MLIRContext *context,
    std::variant<const JsVariableDeclaration *, const JsLVal *> left) {
  if (std::holds_alternative<const JsLVal *>(left)) {
    auto *left_lval = std::get<const JsLVal *>(left);

    return ForInOfLeft{
        .declaration_attr = std::nullopt,
        .lval = left_lval,
    };
  }

  CHECK(std::holds_alternative<const JsVariableDeclaration *>(left))
      << "Exhausted std::variant case.";
  auto *left_declaration = std::get<const JsVariableDeclaration *>(left);

  auto *declarators = left_declaration->declarations();
  if (auto num_declarators = declarators->size(); num_declarators != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Expected exactly 1 declarator, got ", num_declarators, "."));
  }
  JsVariableDeclarator *declarator = declarators->front().get();

  return ForInOfLeft{
      .declaration_attr = JsirForInOfDeclarationAttr::get(
          context,
          /*declaration_loc=*/GetJsirTriviaAttr(context, *left_declaration),
          /*declarator_loc=*/GetJsirTriviaAttr(context, *declarator),
          /*kind=*/mlir::StringAttr::get(context, left_declaration->kind())),
      .lval = declarator->id(),
  };
}

JshirForInStatementOp AstToJsir::VisitForInStatement(
    const JsForInStatement *node) {
  auto left = GetForInOfLeft(builder_.getContext(), node->left());
  if (!left.ok()) {
    mlir::emitError(GetJsirTriviaAttr(builder_.getContext(), *node),
                    left.status().ToString());
    return nullptr;
  }

  mlir::Value mlir_left = VisitLValRef(left->lval);

  mlir::Value mlir_right = VisitExpression(node->right());

  auto op = CreateStmt<JshirForInStatementOp>(
      node, left->declaration_attr.value_or(nullptr), mlir_left, mlir_right);

  mlir::Region &body_region = op.getBody();
  AppendNewBlockAndPopulate(body_region, [&] { VisitStatement(node->body()); });
  return op;
}

JshirForOfStatementOp AstToJsir::VisitForOfStatement(
    const JsForOfStatement *node) {
  auto left = GetForInOfLeft(builder_.getContext(), node->left());
  if (!left.ok()) {
    mlir::emitError(GetJsirTriviaAttr(builder_.getContext(), *node),
                    left.status().ToString());
    return nullptr;
  }

  mlir::BoolAttr mlir_await = builder_.getBoolAttr(node->await());

  mlir::Value mlir_left = VisitLValRef(left->lval);

  mlir::Value mlir_right = VisitExpression(node->right());

  auto op = CreateStmt<JshirForOfStatementOp>(
      node, left->declaration_attr.value_or(nullptr), mlir_left, mlir_right,
      mlir_await);

  mlir::Region &body_region = op.getBody();
  AppendNewBlockAndPopulate(body_region, [&] { VisitStatement(node->body()); });
  return op;
}

JsirArrowFunctionExpressionOp AstToJsir::VisitArrowFunctionExpression(
    const JsArrowFunctionExpression *node) {
  mlir::Value mlir_id;
  if (node->id().has_value()) {
    mlir_id = VisitIdentifierRef(node->id().value());
  }
  std::vector<mlir::Value> mlir_params;
  for (const auto &param : *node->params()) {
    mlir::Value mlir_param = VisitPatternRef(param.get());
    mlir_params.push_back(mlir_param);
  }
  mlir::BoolAttr mlir_generator = builder_.getBoolAttr(node->generator());
  mlir::BoolAttr mlir_async = builder_.getBoolAttr(node->async());
  auto op = CreateExpr<JsirArrowFunctionExpressionOp>(
      node, mlir_id, mlir_params, mlir_generator, mlir_async);
  mlir::Region &body_region = op.getBody();
  AppendNewBlockAndPopulate(body_region, [&] {
    if (std::holds_alternative<const JsBlockStatement *>(node->body())) {
      auto *body = std::get<const JsBlockStatement *>(node->body());
      VisitBlockStatement(body);
    } else if (std::holds_alternative<const JsExpression *>(node->body())) {
      auto *body = std::get<const JsExpression *>(node->body());
      mlir::Value mlir_body = VisitExpression(body);
      CreateStmt<JsirExprRegionEndOp>(body, mlir_body);
    } else {
      LOG(FATAL) << "Unreachable code.";
    }
  });
  return op;
}

JsirObjectPropertyOp AstToJsir::VisitObjectProperty(
    const JsObjectProperty *node) {
  auto mlir_key = GetObjectPropertyKey(node->key(), node->computed());
  mlir::BoolAttr mlir_shorthand = builder_.getBoolAttr(node->shorthand());
  CHECK(std::holds_alternative<const JsExpression *>(node->value()));
  mlir::Value mlir_value =
      VisitExpression(std::get<const JsExpression *>(node->value()));
  return CreateExpr<JsirObjectPropertyOp>(
      node, mlir_key.literal, mlir_key.computed, mlir_shorthand, mlir_value);
}

JsirObjectPropertyRefOp AstToJsir::VisitObjectPropertyRef(
    const JsObjectProperty *node) {
  auto mlir_key = GetObjectPropertyKey(node->key(), node->computed());
  mlir::BoolAttr mlir_shorthand = builder_.getBoolAttr(node->shorthand());
  const JsPattern *value_pattern;
  if (std::holds_alternative<const JsExpression *>(node->value())) {
    const auto *value_expression =
        std::get<const JsExpression *>(node->value());
    CHECK(value_pattern = dynamic_cast<const JsPattern *>(value_expression));
  } else {
    value_pattern = std::get<const JsPattern *>(node->value());
  }
  mlir::Value mlir_value = VisitPatternRef(value_pattern);
  return CreateExpr<JsirObjectPropertyRefOp>(
      node, mlir_key.literal, mlir_key.computed, mlir_shorthand, mlir_value);
}

JsirObjectMethodOp AstToJsir::VisitObjectMethod(const JsObjectMethod *node) {
  auto mlir_key = GetObjectPropertyKey(node->key(), node->computed());
  JsirIdentifierAttr mlir_id;
  if (node->id().has_value()) {
    mlir_id = VisitIdentifierAttr(node->id().value());
  }
  std::vector<mlir::Value> mlir_params;
  for (const auto &param : *node->params()) {
    mlir::Value mlir_param = VisitPatternRef(param.get());
    mlir_params.push_back(mlir_param);
  }
  mlir::BoolAttr mlir_generator = builder_.getBoolAttr(node->generator());
  mlir::BoolAttr mlir_async = builder_.getBoolAttr(node->async());
  mlir::StringAttr mlir_kind = builder_.getStringAttr(node->kind());
  auto op = CreateExpr<JsirObjectMethodOp>(
      node, mlir_key.literal, mlir_key.computed, mlir_id, mlir_params,
      mlir_generator, mlir_async, mlir_kind);
  mlir::Region &mlir_body_region = op.getBody();
  AppendNewBlockAndPopulate(mlir_body_region,
                            [&] { VisitBlockStatement(node->body()); });
  return op;
}

JsirObjectExpressionOp AstToJsir::VisitObjectExpression(
    const JsObjectExpression *node) {
  auto op = CreateExpr<JsirObjectExpressionOp>(node);
  mlir::Region &mlir_properties_region = op.getRegion();
  AppendNewBlockAndPopulate(mlir_properties_region, [&] {
    std::vector<mlir::Value> mlir_properties;
    for (const auto &property : *node->properties_()) {
      mlir::Value mlir_property;
      switch (property.index()) {
        case 0: {
          const JsObjectProperty *property_object_property =
              std::get<0>(property).get();
          mlir_property = VisitObjectProperty(property_object_property);
          break;
        }
        case 1: {
          const JsObjectMethod *property_object_method =
              std::get<1>(property).get();
          mlir_property = VisitObjectMethod(property_object_method);
          break;
        }
        case 2: {
          const JsSpreadElement *property_spread_element =
              std::get<2>(property).get();
          mlir_property = VisitSpreadElement(property_spread_element);
          break;
        }
        default:
          LOG(FATAL) << "Unreachable code.";
      }
      mlir_properties.push_back(mlir_property);
    }
    CreateStmt<JsirExprsRegionEndOp>(node, mlir_properties);
  });
  return op;
}

mlir::Value AstToJsir::VisitMemberExpressionObject(
    std::variant<const JsExpression *, const JsSuper *> object) {
  if (std::holds_alternative<const JsExpression *>(object)) {
    auto *object_expression = std::get<const JsExpression *>(object);
    return VisitExpression(object_expression);
  } else if (std::holds_alternative<const JsSuper *>(object)) {
    auto *object_super = std::get<const JsSuper *>(object);
    return VisitSuper(object_super);
  } else {
    LOG(FATAL) << "Unreachable code.";
  }
}

AstToJsir::MemberExpressionProperty AstToJsir::VisitMemberExpressionProperty(
    std::variant<const JsExpression *, const JsPrivateName *> property,
    bool computed) {
  if (computed) {
    // The op corresponds to a computed (`a[b]`) member expression and
    // `property` is an `Expression`.
    CHECK(std::holds_alternative<const JsExpression *>(property));
    auto *property_expression = std::get<const JsExpression *>(property);
    mlir::Value mlir_property = VisitExpression(property_expression);
    return {.literal = nullptr, .computed = mlir_property};
  } else {
    mlir::Attribute mlir_property;
    // The op corresponds to a static (`a.b`) member expression and `property`
    // is an `Identifier` or a `PrivateName`.
    if (std::holds_alternative<const JsExpression *>(property)) {
      auto *property_expression = std::get<const JsExpression *>(property);
      auto *property_identifier =
          dynamic_cast<const JsIdentifier *>(property_expression);
      CHECK(property_identifier != nullptr)
          << "If computed == false, then `property` can only be Identifier or "
             "PrivateName.";
      mlir_property = VisitIdentifierAttr(property_identifier);
    } else if (std::holds_alternative<const JsPrivateName *>(property)) {
      auto *property_private_name = std::get<const JsPrivateName *>(property);
      mlir_property = VisitPrivateNameAttr(property_private_name);
    }
    return {.literal = mlir_property, .computed = nullptr};
  }
}

JsirMemberExpressionOp AstToJsir::VisitMemberExpression(
    const JsMemberExpression *node) {
  mlir::Value mlir_object = VisitMemberExpressionObject(node->object());
  MemberExpressionProperty mlir_property =
      VisitMemberExpressionProperty(node->property(), node->computed());
  return CreateExpr<JsirMemberExpressionOp>(
      node, mlir_object, mlir_property.literal, mlir_property.computed);
}

JsirMemberExpressionRefOp AstToJsir::VisitMemberExpressionRef(
    const JsMemberExpression *node) {
  mlir::Value mlir_object = VisitMemberExpressionObject(node->object());
  MemberExpressionProperty mlir_property =
      VisitMemberExpressionProperty(node->property(), node->computed());
  return CreateExpr<JsirMemberExpressionRefOp>(
      node, mlir_object, mlir_property.literal, mlir_property.computed);
}

JsirOptionalMemberExpressionOp AstToJsir::VisitOptionalMemberExpression(
    const JsOptionalMemberExpression *node) {
  mlir::Value mlir_object = VisitMemberExpressionObject(node->object());
  MemberExpressionProperty mlir_property =
      VisitMemberExpressionProperty(node->property(), node->computed());
  mlir::BoolAttr mlir_optional = builder_.getBoolAttr(node->optional());
  return CreateExpr<JsirOptionalMemberExpressionOp>(
      node, mlir_object, mlir_property.literal, mlir_property.computed,
      mlir_optional);
}

JsirParenthesizedExpressionOp AstToJsir::VisitParenthesizedExpression(
    const JsParenthesizedExpression *node) {
  mlir::Value mlir_expression = VisitExpression(node->expression());
  return CreateExpr<JsirParenthesizedExpressionOp>(node, mlir_expression);
}

JsirParenthesizedExpressionRefOp AstToJsir::VisitParenthesizedExpressionRef(
    const JsParenthesizedExpression *node) {
  mlir::Value mlir_expression = [&]() -> mlir::Value {
    if (auto *lval = dynamic_cast<const JsLVal *>(node->expression())) {
      return VisitLValRef(lval);
    } else {
      // TODO(b/293174026): Disallow this.
      mlir::emitError(GetJsirTriviaAttr(builder_.getContext(), *node),
                      "lvalue expected");
      return VisitExpression(node->expression());
    }
  }();
  return CreateExpr<JsirParenthesizedExpressionRefOp>(node, mlir_expression);
}

JsirClassMethodOp AstToJsir::VisitClassMethod(const JsClassMethod *node) {
  JsirIdentifierAttr mlir_id;
  if (node->id().has_value()) {
    mlir_id = VisitIdentifierAttr(node->id().value());
  }
  std::vector<mlir::Value> mlir_params;
  for (const auto &param : *node->params()) {
    mlir::Value mlir_param = VisitPatternRef(param.get());
    mlir_params.push_back(mlir_param);
  }
  mlir::BoolAttr mlir_generator = builder_.getBoolAttr(node->generator());
  mlir::BoolAttr mlir_async = builder_.getBoolAttr(node->async());
  auto mlir_key = GetObjectPropertyKey(node->key(), node->computed());
  mlir::StringAttr mlir_kind = builder_.getStringAttr(node->kind());
  mlir::BoolAttr mlir_static = builder_.getBoolAttr(node->static_());
  auto op = CreateStmt<JsirClassMethodOp>(
      node, mlir_id, mlir_params, mlir_generator, mlir_async, mlir_key.literal,
      mlir_key.computed, mlir_kind, mlir_static);
  mlir::Region &mlir_body_region = op.getBody();
  AppendNewBlockAndPopulate(mlir_body_region,
                            [&] { VisitBlockStatement(node->body()); });
  return op;
}

JsirClassPrivateMethodOp AstToJsir::VisitClassPrivateMethod(
    const JsClassPrivateMethod *node) {
  JsirIdentifierAttr mlir_id;
  if (node->id().has_value()) {
    mlir_id = VisitIdentifierAttr(node->id().value());
  }
  std::vector<mlir::Value> mlir_params;
  for (const auto &param : *node->params()) {
    mlir::Value mlir_param = VisitPatternRef(param.get());
    mlir_params.push_back(mlir_param);
  }
  mlir::BoolAttr mlir_generator = builder_.getBoolAttr(node->generator());
  mlir::BoolAttr mlir_async = builder_.getBoolAttr(node->async());
  JsirPrivateNameAttr mlir_key = VisitPrivateNameAttr(node->key());
  mlir::StringAttr mlir_kind = builder_.getStringAttr(node->kind());
  mlir::BoolAttr mlir_static = builder_.getBoolAttr(node->static_());
  auto op = CreateStmt<JsirClassPrivateMethodOp>(
      node, mlir_id, mlir_params, mlir_generator, mlir_async, mlir_key,
      mlir_kind, mlir_static);
  mlir::Region &mlir_body_region = op.getBody();
  AppendNewBlockAndPopulate(mlir_body_region,
                            [&] { VisitBlockStatement(node->body()); });
  return op;
}

JsirClassPropertyOp AstToJsir::VisitClassProperty(const JsClassProperty *node) {
  auto mlir_key = GetObjectPropertyKey(node->key(), node->computed());
  mlir::BoolAttr mlir_static = builder_.getBoolAttr(node->static_());
  auto op = CreateStmt<JsirClassPropertyOp>(node, mlir_key.literal,
                                            mlir_key.computed, mlir_static);
  if (node->value().has_value()) {
    AppendNewBlockAndPopulate(op.getValue(), [&] {
      mlir::Value mlir_value = VisitExpression(node->value().value());
      CreateStmt<JsirExprRegionEndOp>(node->value().value(), mlir_value);
    });
  }
  return op;
}

JsirImportSpecifierAttr AstToJsir::VisitImportSpecifierAttr(
    const JsImportSpecifier *node) {
  auto loc = GetJsirTriviaAttr(builder_.getContext(), *node);
  mlir::Attribute mlir_imported;
  if (std::holds_alternative<const JsIdentifier *>(node->imported())) {
    auto *imported = std::get<const JsIdentifier *>(node->imported());
    mlir_imported = VisitIdentifierAttr(imported);
  } else if (std::holds_alternative<const JsStringLiteral *>(
                 node->imported())) {
    auto *imported = std::get<const JsStringLiteral *>(node->imported());
    mlir_imported = VisitStringLiteralAttr(imported);
  } else {
    LOG(FATAL) << "Unreachable code.";
  }
  JsirIdentifierAttr mlir_local = VisitIdentifierAttr(node->local());
  return JsirImportSpecifierAttr::get(builder_.getContext(), loc, mlir_imported,
                                      mlir_local);
}

JsirImportDefaultSpecifierAttr AstToJsir::VisitImportDefaultSpecifierAttr(
    const JsImportDefaultSpecifier *node) {
  auto loc = GetJsirTriviaAttr(builder_.getContext(), *node);
  JsirIdentifierAttr mlir_local = VisitIdentifierAttr(node->local());
  return JsirImportDefaultSpecifierAttr::get(builder_.getContext(), loc,
                                             mlir_local);
}

JsirImportNamespaceSpecifierAttr AstToJsir::VisitImportNamespaceSpecifierAttr(
    const JsImportNamespaceSpecifier *node) {
  auto loc = GetJsirTriviaAttr(builder_.getContext(), *node);
  JsirIdentifierAttr mlir_local = VisitIdentifierAttr(node->local());
  return JsirImportNamespaceSpecifierAttr::get(builder_.getContext(), loc,
                                               mlir_local);
}

JsirImportAttributeAttr AstToJsir::VisitImportAttributeAttr(
    const JsImportAttribute *node) {
  JsirIdentifierAttr mlir_key = VisitIdentifierAttr(node->key());
  JsirStringLiteralAttr mlir_value = VisitStringLiteralAttr(node->value());
  return JsirImportAttributeAttr::get(builder_.getContext(), mlir_key,
                                      mlir_value);
}

JsirExportSpecifierAttr AstToJsir::VisitExportSpecifierAttr(
    const JsExportSpecifier *node) {
  auto loc = GetJsirTriviaAttr(builder_.getContext(), *node);
  mlir::Attribute mlir_exported;
  if (std::holds_alternative<const JsIdentifier *>(node->exported())) {
    auto *exported = std::get<const JsIdentifier *>(node->exported());
    mlir_exported = VisitIdentifierAttr(exported);
  } else if (std::holds_alternative<const JsStringLiteral *>(
                 node->exported())) {
    auto *exported = std::get<const JsStringLiteral *>(node->exported());
    mlir_exported = VisitStringLiteralAttr(exported);
  } else {
    LOG(FATAL) << "Unreachable code.";
  }
  mlir::Attribute mlir_local;
  if (node->local().has_value()) {
    auto local_variant = node->local().value();
    if (std::holds_alternative<const JsIdentifier *>(local_variant)) {
      auto *local = std::get<const JsIdentifier *>(local_variant);
      mlir_local = VisitIdentifierAttr(local);
    } else if (std::holds_alternative<const JsStringLiteral *>(local_variant)) {
      auto *local = std::get<const JsStringLiteral *>(local_variant);
      mlir_local = VisitStringLiteralAttr(local);
    } else {
      LOG(FATAL) << "Unreachable code.";
    }
  }
  return JsirExportSpecifierAttr::get(builder_.getContext(), loc, mlir_exported,
                                      mlir_local);
}

JsirExportDefaultDeclarationOp AstToJsir::VisitExportDefaultDeclaration(
    const JsExportDefaultDeclaration *node) {
  auto op = CreateStmt<JsirExportDefaultDeclarationOp>(node);
  mlir::Region &mlir_declaration_region = op.getDeclaration();
  AppendNewBlockAndPopulate(mlir_declaration_region, [&] {
    if (std::holds_alternative<const JsFunctionDeclaration *>(
            node->declaration())) {
      auto *declaration =
          std::get<const JsFunctionDeclaration *>(node->declaration());
      VisitFunctionDeclaration(declaration);
    } else if (std::holds_alternative<const JsClassDeclaration *>(
                   node->declaration())) {
      auto *declaration =
          std::get<const JsClassDeclaration *>(node->declaration());
      VisitClassDeclaration(declaration);
    } else if (std::holds_alternative<const JsExpression *>(
                   node->declaration())) {
      auto *declaration = std::get<const JsExpression *>(node->declaration());
      mlir::Value mlir_declaration = VisitExpression(declaration);
      CreateStmt<JsirExprRegionEndOp>(declaration, mlir_declaration);
    } else {
      LOG(FATAL) << "Unreachable code.";
    }
  });
  return op;
}

void AstToJsir::AppendNewBlockAndPopulate(mlir::Region &region,
                                          std::function<void()> populate) {
  // Save insertion point.
  // Will revert at the end.
  mlir::OpBuilder::InsertionGuard insertion_guard(builder_);

  // Insert new block and point builder to it.
  mlir::Block &block = region.emplaceBlock();
  builder_.setInsertionPointToStart(&block);

  populate();
}
AstToJsir::ObjectPropertyKey AstToJsir::GetObjectPropertyKey(
    const JsExpression *node, bool computed) {
  if (!computed) {
    mlir::Attribute attr;
    if (auto *identifier = dynamic_cast<const JsIdentifier *>(node)) {
      attr = VisitIdentifierAttr(identifier);
    } else if (auto *string_literal =
                   dynamic_cast<const JsStringLiteral *>(node)) {
      attr = VisitStringLiteralAttr(string_literal);
    } else if (auto *numeric_literal =
                   dynamic_cast<const JsNumericLiteral *>(node)) {
      attr = VisitNumericLiteralAttr(numeric_literal);
    } else if (auto *big_int_literal =
                   dynamic_cast<const JsBigIntLiteral *>(node)) {
      attr = VisitBigIntLiteralAttr(big_int_literal);
    } else {
      LOG(FATAL) << "Invalid property name.";
    }
    return ObjectPropertyKey{.literal = attr, .computed = nullptr};
  } else {
    JsirExpressionOpInterface op = VisitExpression(node);
    return ObjectPropertyKey{.literal = nullptr, .computed = op};
  }
}

}  // namespace maldoca
