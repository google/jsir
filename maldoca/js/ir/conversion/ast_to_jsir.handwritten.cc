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

JshirForStatementOp AstToJsir::VisitForStatement(mlir::OpBuilder& builder,
                                                 const JsForStatement* node) {
  auto op = CreateStmt<JshirForStatementOp>(builder, node);
  mlir::Region& init_region = op.getInit();
  if (node->init().has_value()) {
    AppendNewBlockAndPopulate(builder, init_region, [&] {
      auto init = node->init().value();
      if (std::holds_alternative<const JsVariableDeclaration*>(init)) {
        auto* init_variable_declaration =
            std::get<const JsVariableDeclaration*>(init);
        VisitVariableDeclaration(builder, init_variable_declaration);
      } else if (std::holds_alternative<const JsExpression*>(init)) {
        auto* init_expression = std::get<const JsExpression*>(init);
        mlir::Value mlir_init = VisitExpression(builder, init_expression);
        CreateStmt<JsirExprRegionEndOp>(builder, nullptr, mlir_init);
      }
    });
  }
  mlir::Region& test_region = op.getTest();
  if (node->test().has_value()) {
    const JsExpression* test = node->test().value();
    AppendNewBlockAndPopulate(builder, test_region, [&] {
      mlir::Value mlir_test = VisitExpression(builder, test);
      CreateStmt<JsirExprRegionEndOp>(builder, nullptr, mlir_test);
    });
  }
  mlir::Region& update_region = op.getUpdate();
  if (node->update().has_value()) {
    const JsExpression* update = node->update().value();
    AppendNewBlockAndPopulate(builder, update_region, [&] {
      mlir::Value mlir_update = VisitExpression(builder, update);
      CreateStmt<JsirExprRegionEndOp>(builder, nullptr, mlir_update);
    });
  }
  mlir::Region& body_region = op.getBody();
  AppendNewBlockAndPopulate(builder, body_region,
                            [&] { VisitStatement(builder, node->body()); });
  return op;
}

struct ForInOfLeft {
  std::optional<JsirForInOfDeclarationAttr> declaration_attr;
  const JsLVal* lval;
};

static absl::StatusOr<ForInOfLeft> GetForInOfLeft(
    mlir::MLIRContext* context,
    std::variant<const JsVariableDeclaration*, const JsLVal*> left) {
  if (std::holds_alternative<const JsLVal*>(left)) {
    auto* left_lval = std::get<const JsLVal*>(left);

    return ForInOfLeft{
        .declaration_attr = std::nullopt,
        .lval = left_lval,
    };
  }

  CHECK(std::holds_alternative<const JsVariableDeclaration*>(left))
      << "Exhausted std::variant case.";
  auto* left_declaration = std::get<const JsVariableDeclaration*>(left);

  auto* declarators = left_declaration->declarations();
  if (auto num_declarators = declarators->size(); num_declarators != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Expected exactly 1 declarator, got ", num_declarators, "."));
  }
  JsVariableDeclarator* declarator = declarators->front().get();

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
    mlir::OpBuilder& builder, const JsForInStatement* node) {
  auto left = GetForInOfLeft(builder.getContext(), node->left());
  if (!left.ok()) {
    mlir::emitError(GetJsirTriviaAttr(builder.getContext(), *node),
                    left.status().ToString());
    return nullptr;
  }

  mlir::Value mlir_left = VisitLValRef(builder, left->lval);

  mlir::Value mlir_right = VisitExpression(builder, node->right());

  auto op = CreateStmt<JshirForInStatementOp>(
      builder, node, left->declaration_attr.value_or(nullptr), mlir_left,
      mlir_right);

  mlir::Region& body_region = op.getBody();
  AppendNewBlockAndPopulate(builder, body_region,
                            [&] { VisitStatement(builder, node->body()); });
  return op;
}

JshirForOfStatementOp AstToJsir::VisitForOfStatement(
    mlir::OpBuilder& builder, const JsForOfStatement* node) {
  auto left = GetForInOfLeft(builder.getContext(), node->left());
  if (!left.ok()) {
    mlir::emitError(GetJsirTriviaAttr(builder.getContext(), *node),
                    left.status().ToString());
    return nullptr;
  }

  mlir::BoolAttr mlir_await = builder.getBoolAttr(node->await());

  mlir::Value mlir_left = VisitLValRef(builder, left->lval);

  mlir::Value mlir_right = VisitExpression(builder, node->right());

  auto op = CreateStmt<JshirForOfStatementOp>(
      builder, node, left->declaration_attr.value_or(nullptr), mlir_left,
      mlir_right, mlir_await);

  mlir::Region& body_region = op.getBody();
  AppendNewBlockAndPopulate(builder, body_region,
                            [&] { VisitStatement(builder, node->body()); });
  return op;
}

JsirArrowFunctionExpressionOp AstToJsir::VisitArrowFunctionExpression(
    mlir::OpBuilder& builder, const JsArrowFunctionExpression* node) {
  mlir::Value mlir_id;
  if (node->id().has_value()) {
    mlir_id = VisitIdentifierRef(builder, node->id().value());
  }
  std::vector<mlir::Value> mlir_params;
  for (const auto& param : *node->params()) {
    mlir::Value mlir_param = VisitPatternRef(builder, param.get());
    mlir_params.push_back(mlir_param);
  }
  mlir::BoolAttr mlir_generator = builder.getBoolAttr(node->generator());
  mlir::BoolAttr mlir_async = builder.getBoolAttr(node->async());
  auto op = CreateExpr<JsirArrowFunctionExpressionOp>(
      builder, node, mlir_id, mlir_params, mlir_generator, mlir_async);
  mlir::Region& body_region = op.getBody();
  AppendNewBlockAndPopulate(builder, body_region, [&] {
    if (std::holds_alternative<const JsBlockStatement*>(node->body())) {
      auto* body = std::get<const JsBlockStatement*>(node->body());
      VisitBlockStatement(builder, body);
    } else if (std::holds_alternative<const JsExpression*>(node->body())) {
      auto* body = std::get<const JsExpression*>(node->body());
      mlir::Value mlir_body = VisitExpression(builder, body);
      CreateStmt<JsirExprRegionEndOp>(builder, nullptr, mlir_body);
    } else {
      LOG(FATAL) << "Unreachable code.";
    }
  });
  return op;
}

JsirObjectPropertyOp AstToJsir::VisitObjectProperty(
    mlir::OpBuilder& builder, const JsObjectProperty* node) {
  auto mlir_key = GetObjectPropertyKey(builder, node->key(), node->computed());
  mlir::BoolAttr mlir_shorthand = builder.getBoolAttr(node->shorthand());
  CHECK(std::holds_alternative<const JsExpression*>(node->value()));
  mlir::Value mlir_value =
      VisitExpression(builder, std::get<const JsExpression*>(node->value()));
  return CreateExpr<JsirObjectPropertyOp>(builder, node, mlir_key.literal,
                                          mlir_key.computed, mlir_shorthand,
                                          mlir_value);
}

JsirObjectPropertyRefOp AstToJsir::VisitObjectPropertyRef(
    mlir::OpBuilder& builder, const JsObjectProperty* node) {
  auto mlir_key = GetObjectPropertyKey(builder, node->key(), node->computed());
  mlir::BoolAttr mlir_shorthand = builder.getBoolAttr(node->shorthand());
  const JsPattern* value_pattern;
  if (std::holds_alternative<const JsExpression*>(node->value())) {
    const auto* value_expression = std::get<const JsExpression*>(node->value());
    CHECK(value_pattern = dynamic_cast<const JsPattern*>(value_expression));
  } else {
    value_pattern = std::get<const JsPattern*>(node->value());
  }
  mlir::Value mlir_value = VisitPatternRef(builder, value_pattern);
  return CreateExpr<JsirObjectPropertyRefOp>(builder, node, mlir_key.literal,
                                             mlir_key.computed, mlir_shorthand,
                                             mlir_value);
}

JsirObjectMethodOp AstToJsir::VisitObjectMethod(mlir::OpBuilder& builder,
                                                const JsObjectMethod* node) {
  auto mlir_key = GetObjectPropertyKey(builder, node->key(), node->computed());
  JsirIdentifierAttr mlir_id;
  if (node->id().has_value()) {
    mlir_id = VisitIdentifierAttr(builder, node->id().value());
  }
  std::vector<mlir::Value> mlir_params;
  for (const auto& param : *node->params()) {
    mlir::Value mlir_param = VisitPatternRef(builder, param.get());
    mlir_params.push_back(mlir_param);
  }
  mlir::BoolAttr mlir_generator = builder.getBoolAttr(node->generator());
  mlir::BoolAttr mlir_async = builder.getBoolAttr(node->async());
  mlir::StringAttr mlir_kind = builder.getStringAttr(node->kind());
  auto op = CreateExpr<JsirObjectMethodOp>(
      builder, node, mlir_key.literal, mlir_key.computed, mlir_id, mlir_params,
      mlir_generator, mlir_async, mlir_kind);
  mlir::Region& mlir_body_region = op.getBody();
  AppendNewBlockAndPopulate(builder, mlir_body_region, [&] {
    VisitBlockStatement(builder, node->body());
  });
  return op;
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

mlir::Value AstToJsir::VisitMemberExpressionObject(
    mlir::OpBuilder& builder,
    std::variant<const JsExpression*, const JsSuper*> object) {
  if (std::holds_alternative<const JsExpression*>(object)) {
    auto* object_expression = std::get<const JsExpression*>(object);
    return VisitExpression(builder, object_expression);
  } else if (std::holds_alternative<const JsSuper*>(object)) {
    auto* object_super = std::get<const JsSuper*>(object);
    return VisitSuper(builder, object_super);
  } else {
    LOG(FATAL) << "Unreachable code.";
  }
}

AstToJsir::MemberExpressionProperty AstToJsir::VisitMemberExpressionProperty(
    mlir::OpBuilder& builder,
    std::variant<const JsExpression*, const JsPrivateName*> property,
    bool computed) {
  if (computed) {
    // The op corresponds to a computed (`a[b]`) member expression and
    // `property` is an `Expression`.
    CHECK(std::holds_alternative<const JsExpression*>(property));
    auto* property_expression = std::get<const JsExpression*>(property);
    mlir::Value mlir_property = VisitExpression(builder, property_expression);
    return {.literal = nullptr, .computed = mlir_property};
  } else {
    mlir::Attribute mlir_property;
    // The op corresponds to a static (`a.b`) member expression and `property`
    // is an `Identifier` or a `PrivateName`.
    if (std::holds_alternative<const JsExpression*>(property)) {
      auto* property_expression = std::get<const JsExpression*>(property);
      auto* property_identifier =
          dynamic_cast<const JsIdentifier*>(property_expression);
      CHECK(property_identifier != nullptr)
          << "If computed == false, then `property` can only be Identifier or "
             "PrivateName.";
      mlir_property = VisitIdentifierAttr(builder, property_identifier);
    } else if (std::holds_alternative<const JsPrivateName*>(property)) {
      auto* property_private_name = std::get<const JsPrivateName*>(property);
      mlir_property = VisitPrivateNameAttr(builder, property_private_name);
    }
    return {.literal = mlir_property, .computed = nullptr};
  }
}

JsirMemberExpressionOp AstToJsir::VisitMemberExpression(
    mlir::OpBuilder& builder, const JsMemberExpression* node) {
  mlir::Value mlir_object =
      VisitMemberExpressionObject(builder, node->object());
  MemberExpressionProperty mlir_property = VisitMemberExpressionProperty(
      builder, node->property(), node->computed());
  return CreateExpr<JsirMemberExpressionOp>(builder, node, mlir_object,
                                            mlir_property.literal,
                                            mlir_property.computed);
}

JsirMemberExpressionRefOp AstToJsir::VisitMemberExpressionRef(
    mlir::OpBuilder& builder, const JsMemberExpression* node) {
  mlir::Value mlir_object =
      VisitMemberExpressionObject(builder, node->object());
  MemberExpressionProperty mlir_property = VisitMemberExpressionProperty(
      builder, node->property(), node->computed());
  return CreateExpr<JsirMemberExpressionRefOp>(builder, node, mlir_object,
                                               mlir_property.literal,
                                               mlir_property.computed);
}

JsirOptionalMemberExpressionOp AstToJsir::VisitOptionalMemberExpression(
    mlir::OpBuilder& builder, const JsOptionalMemberExpression* node) {
  mlir::Value mlir_object =
      VisitMemberExpressionObject(builder, node->object());
  MemberExpressionProperty mlir_property = VisitMemberExpressionProperty(
      builder, node->property(), node->computed());
  mlir::BoolAttr mlir_optional = builder.getBoolAttr(node->optional());
  return CreateExpr<JsirOptionalMemberExpressionOp>(
      builder, node, mlir_object, mlir_property.literal, mlir_property.computed,
      mlir_optional);
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

JsirClassMethodOp AstToJsir::VisitClassMethod(mlir::OpBuilder& builder,
                                              const JsClassMethod* node) {
  JsirIdentifierAttr mlir_id;
  if (node->id().has_value()) {
    mlir_id = VisitIdentifierAttr(builder, node->id().value());
  }
  std::vector<mlir::Value> mlir_params;
  for (const auto& param : *node->params()) {
    mlir::Value mlir_param = VisitPatternRef(builder, param.get());
    mlir_params.push_back(mlir_param);
  }
  mlir::BoolAttr mlir_generator = builder.getBoolAttr(node->generator());
  mlir::BoolAttr mlir_async = builder.getBoolAttr(node->async());
  auto mlir_key = GetObjectPropertyKey(builder, node->key(), node->computed());
  mlir::StringAttr mlir_kind = builder.getStringAttr(node->kind());
  mlir::BoolAttr mlir_static = builder.getBoolAttr(node->static_());
  auto op = CreateStmt<JsirClassMethodOp>(
      builder, node, mlir_id, mlir_params, mlir_generator, mlir_async,
      mlir_key.literal, mlir_key.computed, mlir_kind, mlir_static);
  mlir::Region& mlir_body_region = op.getBody();
  AppendNewBlockAndPopulate(builder, mlir_body_region, [&] {
    VisitBlockStatement(builder, node->body());
  });
  return op;
}

JsirClassPrivateMethodOp AstToJsir::VisitClassPrivateMethod(
    mlir::OpBuilder& builder, const JsClassPrivateMethod* node) {
  JsirIdentifierAttr mlir_id;
  if (node->id().has_value()) {
    mlir_id = VisitIdentifierAttr(builder, node->id().value());
  }
  std::vector<mlir::Value> mlir_params;
  for (const auto& param : *node->params()) {
    mlir::Value mlir_param = VisitPatternRef(builder, param.get());
    mlir_params.push_back(mlir_param);
  }
  mlir::BoolAttr mlir_generator = builder.getBoolAttr(node->generator());
  mlir::BoolAttr mlir_async = builder.getBoolAttr(node->async());
  JsirPrivateNameAttr mlir_key = VisitPrivateNameAttr(builder, node->key());
  mlir::StringAttr mlir_kind = builder.getStringAttr(node->kind());
  mlir::BoolAttr mlir_static = builder.getBoolAttr(node->static_());
  auto op = CreateStmt<JsirClassPrivateMethodOp>(
      builder, node, mlir_id, mlir_params, mlir_generator, mlir_async, mlir_key,
      mlir_kind, mlir_static);
  mlir::Region& mlir_body_region = op.getBody();
  AppendNewBlockAndPopulate(builder, mlir_body_region, [&] {
    VisitBlockStatement(builder, node->body());
  });
  return op;
}

JsirClassPropertyOp AstToJsir::VisitClassProperty(mlir::OpBuilder& builder,
                                                  const JsClassProperty* node) {
  auto mlir_key = GetObjectPropertyKey(builder, node->key(), node->computed());
  mlir::BoolAttr mlir_static = builder.getBoolAttr(node->static_());
  auto op = CreateStmt<JsirClassPropertyOp>(builder, node, mlir_key.literal,
                                            mlir_key.computed, mlir_static);
  if (node->value().has_value()) {
    AppendNewBlockAndPopulate(builder, op.getValue(), [&] {
      mlir::Value mlir_value = VisitExpression(builder, node->value().value());
      CreateStmt<JsirExprRegionEndOp>(builder, nullptr,
                                      mlir_value);
    });
  }
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

JsirExportSpecifierAttr AstToJsir::VisitExportSpecifierAttr(
    mlir::OpBuilder& builder, const JsExportSpecifier* node) {
  auto loc = GetJsirTriviaAttr(builder.getContext(), *node);
  mlir::Attribute mlir_exported;
  if (std::holds_alternative<const JsIdentifier*>(node->exported())) {
    auto* exported = std::get<const JsIdentifier*>(node->exported());
    mlir_exported = VisitIdentifierAttr(builder, exported);
  } else if (std::holds_alternative<const JsStringLiteral*>(node->exported())) {
    auto* exported = std::get<const JsStringLiteral*>(node->exported());
    mlir_exported = VisitStringLiteralAttr(builder, exported);
  } else {
    LOG(FATAL) << "Unreachable code.";
  }
  mlir::Attribute mlir_local;
  if (node->local().has_value()) {
    auto local_variant = node->local().value();
    if (std::holds_alternative<const JsIdentifier*>(local_variant)) {
      auto* local = std::get<const JsIdentifier*>(local_variant);
      mlir_local = VisitIdentifierAttr(builder, local);
    } else if (std::holds_alternative<const JsStringLiteral*>(local_variant)) {
      auto* local = std::get<const JsStringLiteral*>(local_variant);
      mlir_local = VisitStringLiteralAttr(builder, local);
    } else {
      LOG(FATAL) << "Unreachable code.";
    }
  }
  return JsirExportSpecifierAttr::get(builder.getContext(), loc, mlir_exported,
                                      mlir_local);
}

JsirExportDefaultDeclarationOp AstToJsir::VisitExportDefaultDeclaration(
    mlir::OpBuilder& builder, const JsExportDefaultDeclaration* node) {
  auto op = CreateStmt<JsirExportDefaultDeclarationOp>(builder, node);
  mlir::Region& mlir_declaration_region = op.getDeclaration();
  AppendNewBlockAndPopulate(builder, mlir_declaration_region, [&] {
    if (std::holds_alternative<const JsFunctionDeclaration*>(
            node->declaration())) {
      auto* declaration =
          std::get<const JsFunctionDeclaration*>(node->declaration());
      VisitFunctionDeclaration(builder, declaration);
    } else if (std::holds_alternative<const JsClassDeclaration*>(
                   node->declaration())) {
      auto* declaration =
          std::get<const JsClassDeclaration*>(node->declaration());
      VisitClassDeclaration(builder, declaration);
    } else if (std::holds_alternative<const JsExpression*>(
                   node->declaration())) {
      auto* declaration = std::get<const JsExpression*>(node->declaration());
      mlir::Value mlir_declaration = VisitExpression(builder, declaration);
      CreateStmt<JsirExprRegionEndOp>(builder, nullptr, mlir_declaration);
    } else {
      LOG(FATAL) << "Unreachable code.";
    }
  });
  return op;
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
AstToJsir::ObjectPropertyKey AstToJsir::GetObjectPropertyKey(
    mlir::OpBuilder& builder, const JsExpression* node, bool computed) {
  if (!computed) {
    mlir::Attribute attr;
    if (auto* identifier = dynamic_cast<const JsIdentifier*>(node)) {
      attr = VisitIdentifierAttr(builder, identifier);
    } else if (auto* string_literal =
                   dynamic_cast<const JsStringLiteral*>(node)) {
      attr = VisitStringLiteralAttr(builder, string_literal);
    } else if (auto* numeric_literal =
                   dynamic_cast<const JsNumericLiteral*>(node)) {
      attr = VisitNumericLiteralAttr(builder, numeric_literal);
    } else if (auto* big_int_literal =
                   dynamic_cast<const JsBigIntLiteral*>(node)) {
      attr = VisitBigIntLiteralAttr(builder, big_int_literal);
    } else {
      LOG(FATAL) << "Invalid property name.";
    }
    return ObjectPropertyKey{.literal = attr, .computed = nullptr};
  } else {
    JsirExpressionOpInterface op = VisitExpression(builder, node);
    return ObjectPropertyKey{.literal = nullptr, .computed = op};
  }
}

}  // namespace maldoca
