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

#include "maldoca/js/ir/conversion/jsir_to_ast.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "maldoca/base/status_macros.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/ir/cast.h"
#include "maldoca/js/ir/ir.h"
#include "maldoca/js/ir/trivia.h"

namespace maldoca {

absl::StatusOr<std::unique_ptr<JsInterpreterDirective>>
JsirToAst::VisitInterpreterDirectiveAttr(JsirInterpreterDirectiveAttr attr) {
  return Create<JsInterpreterDirective>(attr, attr.getValue().str());
}

absl::StatusOr<std::unique_ptr<JsDirectiveLiteralExtra>>
JsirToAst::VisitDirectiveLiteralExtraAttr(JsirDirectiveLiteralExtraAttr attr) {
  return Create<JsDirectiveLiteralExtra>(attr, attr.getRaw().str(),
                                         attr.getRawValue().str());
}

absl::StatusOr<std::unique_ptr<JsIdentifier>> JsirToAst::VisitIdentifierAttr(
    JsirIdentifierAttr attr) {
  return Create<JsIdentifier>(attr, attr.getName().str());
}

absl::StatusOr<std::unique_ptr<JsPrivateName>> JsirToAst::VisitPrivateNameAttr(
    JsirPrivateNameAttr attr) {
  MALDOCA_ASSIGN_OR_RETURN(auto id, VisitIdentifierAttr(attr.getId()));
  return Create<JsPrivateName>(attr, std::move(id));
}

absl::StatusOr<std::unique_ptr<JsRegExpLiteralExtra>>
JsirToAst::VisitRegExpLiteralExtraAttr(JsirRegExpLiteralExtraAttr attr) {
  return Create<JsRegExpLiteralExtra>(attr, attr.getRaw().str());
}

absl::StatusOr<std::unique_ptr<JsStringLiteralExtra>>
JsirToAst::VisitStringLiteralExtraAttr(JsirStringLiteralExtraAttr attr) {
  return Create<JsStringLiteralExtra>(attr, attr.getRaw().str(),
                                      attr.getRawValue().str());
}

absl::StatusOr<std::unique_ptr<JsStringLiteral>>
JsirToAst::VisitStringLiteralAttr(JsirStringLiteralAttr attr) {
  MALDOCA_ASSIGN_OR_RETURN(auto extra,
                           VisitStringLiteralExtraAttr(attr.getExtra()));
  return Create<JsStringLiteral>(attr, attr.getValue().str(), std::move(extra));
}

absl::StatusOr<std::unique_ptr<JsNumericLiteralExtra>>
JsirToAst::VisitNumericLiteralExtraAttr(JsirNumericLiteralExtraAttr attr) {
  return Create<JsNumericLiteralExtra>(attr, attr.getRaw().str(),
                                       attr.getRawValue().getValueAsDouble());
}

absl::StatusOr<std::unique_ptr<JsNumericLiteral>>
JsirToAst::VisitNumericLiteralAttr(JsirNumericLiteralAttr attr) {
  MALDOCA_ASSIGN_OR_RETURN(auto extra,
                           VisitNumericLiteralExtraAttr(attr.getExtra()));
  return Create<JsNumericLiteral>(attr, attr.getValue().getValueAsDouble(),
                                  std::move(extra));
}

absl::StatusOr<std::unique_ptr<JsBigIntLiteralExtra>>
JsirToAst::VisitBigIntLiteralExtraAttr(JsirBigIntLiteralExtraAttr attr) {
  return Create<JsBigIntLiteralExtra>(attr, attr.getRaw().str(),
                                      attr.getRawValue().str());
}

absl::StatusOr<std::unique_ptr<JsBigIntLiteral>>
JsirToAst::VisitBigIntLiteralAttr(JsirBigIntLiteralAttr attr) {
  MALDOCA_ASSIGN_OR_RETURN(auto extra,
                           VisitBigIntLiteralExtraAttr(attr.getExtra()));
  return Create<JsBigIntLiteral>(attr, attr.getValue().str(), std::move(extra));
}

absl::StatusOr<std::unique_ptr<JsBreakStatement>>
JsirToAst::VisitBreakStatement(JshirBreakStatementOp op) {
  std::optional<std::unique_ptr<JsIdentifier>> label;
  if (op.getLabel().has_value()) {
    MALDOCA_ASSIGN_OR_RETURN(label, VisitIdentifierAttr(op.getLabel().value()));
  }
  return Create<JsBreakStatement>(op, std::move(label));
}

absl::StatusOr<std::unique_ptr<JsContinueStatement>>
JsirToAst::VisitContinueStatement(JshirContinueStatementOp op) {
  std::optional<std::unique_ptr<JsIdentifier>> label;
  if (op.getLabel().has_value()) {
    MALDOCA_ASSIGN_OR_RETURN(label, VisitIdentifierAttr(op.getLabel().value()));
  }
  return Create<JsContinueStatement>(op, std::move(label));
}

absl::StatusOr<std::unique_ptr<JsForStatement>> JsirToAst::VisitForStatement(
    maldoca::JshirForStatementOp op) {
  std::optional<std::variant<std::unique_ptr<JsVariableDeclaration>,
                             std::unique_ptr<JsExpression>>>
      init;
  if (!op.getInit().empty()) {
    MALDOCA_ASSIGN_OR_RETURN(
        init, VisitStmtOrExprRegion(op.getInit(),
                                    &JsirToAst::VisitVariableDeclaration,
                                    &JsirToAst::VisitExpression));
  }
  std::optional<std::unique_ptr<JsExpression>> test;
  if (!op.getTest().empty()) {
    MALDOCA_ASSIGN_OR_RETURN(
        test, VisitExprRegion(op.getTest(), &JsirToAst::VisitExpression));
  }
  std::optional<std::unique_ptr<JsExpression>> update;
  if (!op.getUpdate().empty()) {
    MALDOCA_ASSIGN_OR_RETURN(
        update, VisitExprRegion(op.getUpdate(), &JsirToAst::VisitExpression));
  }
  MALDOCA_ASSIGN_OR_RETURN(
      auto body, VisitStmtRegion(op.getBody(), &JsirToAst::VisitStatement));
  return Create<JsForStatement>(op, std::move(init), std::move(test),
                                std::move(update), std::move(body));
}

absl::StatusOr<JsirToAst::JsForInOfStatementFields>
JsirToAst::VisitForInOfStatement(
    std::optional<JsirForInOfDeclarationAttr> left_declaration,
    mlir::Value left_lval_value, mlir::Value right_value,
    mlir::Region &body_region) {
  MALDOCA_ASSIGN_OR_RETURN(auto left_lval_op,
                           Cast<JsirLValRefOpInterface>(left_lval_value));
  MALDOCA_ASSIGN_OR_RETURN(auto left_lval, VisitLValRef(left_lval_op));

  std::variant<std::unique_ptr<JsVariableDeclaration>, std::unique_ptr<JsLVal>>
      left;
  if (!left_declaration.has_value()) {
    left = std::move(left_lval);
  } else {
    auto declarator = CreateJsNodeWithTrivia<JsVariableDeclarator>(
        left_declaration->getDeclaratorLoc(), std::move(left_lval),
        /*init=*/std::nullopt);

    std::vector<std::unique_ptr<JsVariableDeclarator>> declarations;
    declarations.push_back(std::move(declarator));

    std::string kind;
    left = CreateJsNodeWithTrivia<JsVariableDeclaration>(
        left_declaration->getDeclarationLoc(), std::move(declarations),
        left_declaration->getKind().str());
  }

  MALDOCA_ASSIGN_OR_RETURN(auto right_op,
                           Cast<JsirExpressionOpInterface>(right_value));
  MALDOCA_ASSIGN_OR_RETURN(auto right, VisitExpression(right_op));

  MALDOCA_ASSIGN_OR_RETURN(
      auto body, VisitStmtRegion(body_region, &JsirToAst::VisitStatement));

  return JsForInOfStatementFields{std::move(left), std::move(right),
                                  std::move(body)};
}

absl::StatusOr<std::unique_ptr<JsForInStatement>>
JsirToAst::VisitForInStatement(JshirForInStatementOp op) {
  MALDOCA_ASSIGN_OR_RETURN(
      auto fields,
      VisitForInOfStatement(op.getLeftDeclaration(), op.getLeftLval(),
                            op.getRight(), op.getBody()));
  return Create<JsForInStatement>(op, std::move(fields.left),
                                  std::move(fields.right),
                                  std::move(fields.body));
}

absl::StatusOr<std::unique_ptr<JsForOfStatement>>
JsirToAst::VisitForOfStatement(JshirForOfStatementOp op) {
  MALDOCA_ASSIGN_OR_RETURN(
      auto fields,
      VisitForInOfStatement(op.getLeftDeclaration(), op.getLeftLval(),
                            op.getRight(), op.getBody()));
  return Create<JsForOfStatement>(op, std::move(fields.left),
                                  std::move(fields.right),
                                  std::move(fields.body), op.getAwait());
}

absl::StatusOr<std::unique_ptr<JsArrowFunctionExpression>>
JsirToAst::VisitArrowFunctionExpression(JsirArrowFunctionExpressionOp op) {
  std::optional<std::unique_ptr<JsIdentifier>> id;
  if (op.getId() != nullptr) {
    MALDOCA_ASSIGN_OR_RETURN(auto mlir_id,
                             Cast<JsirIdentifierRefOp>(op.getId()));
    MALDOCA_ASSIGN_OR_RETURN(auto id, VisitIdentifierRef(mlir_id));
  }
  std::vector<std::unique_ptr<JsPattern>> params;
  for (mlir::Value mlir_param_value : op.getParams()) {
    MALDOCA_ASSIGN_OR_RETURN(auto mlir_param,
                             Cast<JsirPatternRefOpInterface>(mlir_param_value));
    MALDOCA_ASSIGN_OR_RETURN(auto param, VisitPatternRef(mlir_param));
    params.push_back(std::move(param));
  }
  bool generator = op.getGenerator();
  bool async = op.getAsync();
  MALDOCA_ASSIGN_OR_RETURN(
      auto body,
      VisitStmtOrExprRegion(op.getBody(), &JsirToAst::VisitBlockStatement,
                            &JsirToAst::VisitExpression));
  return Create<JsArrowFunctionExpression>(op, std::move(id), std::move(params),
                                           generator, async, std::move(body));
}

absl::StatusOr<JsirToAst::ObjectPropertyKey> JsirToAst::GetObjectPropertyKey(
    mlir::Value computed_key, std::optional<mlir::Attribute> literal_key) {
  if (computed_key != nullptr) {
    MALDOCA_ASSIGN_OR_RETURN(auto mlir_computed_key,
                             Cast<JsirExpressionOpInterface>(computed_key));
    MALDOCA_ASSIGN_OR_RETURN(auto key, VisitExpression(mlir_computed_key));
    return ObjectPropertyKey{.key = std::move(key), .computed = true};
  } else if (literal_key.has_value()) {
    mlir::Attribute mlir_literal_key_attr = literal_key.value();
    std::unique_ptr<JsExpression> key;
    if (auto mlir_literal_key =
            mlir::dyn_cast<JsirIdentifierAttr>(mlir_literal_key_attr)) {
      MALDOCA_ASSIGN_OR_RETURN(key, VisitIdentifierAttr(mlir_literal_key));
    } else if (auto mlir_literal_key = mlir::dyn_cast<JsirStringLiteralAttr>(
                   mlir_literal_key_attr)) {
      MALDOCA_ASSIGN_OR_RETURN(key, VisitStringLiteralAttr(mlir_literal_key));
    } else if (auto mlir_literal_key = mlir::dyn_cast<JsirNumericLiteralAttr>(
                   mlir_literal_key_attr)) {
      MALDOCA_ASSIGN_OR_RETURN(key, VisitNumericLiteralAttr(mlir_literal_key));
    } else if (auto mlir_literal_key = mlir::dyn_cast<JsirBigIntLiteralAttr>(
                   mlir_literal_key_attr)) {
      MALDOCA_ASSIGN_OR_RETURN(key, VisitBigIntLiteralAttr(mlir_literal_key));
    } else {
      return absl::InvalidArgumentError(
          "literal_key must be Identifier or StringLiteral or "
          "NumericLiteral.");
    }
    return ObjectPropertyKey{.key = std::move(key), .computed = false};
  } else {
    return absl::InvalidArgumentError(
        "One of computed_key and literal_key must be set.");
  }
}

absl::StatusOr<std::unique_ptr<JsObjectProperty>>
JsirToAst::VisitObjectProperty(JsirObjectPropertyOp op) {
  MALDOCA_ASSIGN_OR_RETURN(
      auto object_property_key,
      GetObjectPropertyKey(op.getComputedKey(), op.getLiteralKey()));

  bool shorthand = op.getShorthand();

  MALDOCA_ASSIGN_OR_RETURN(auto mlir_value,
                           Cast<JsirExpressionOpInterface>(op.getValue()));
  MALDOCA_ASSIGN_OR_RETURN(auto value, VisitExpression(mlir_value));

  return Create<JsObjectProperty>(op, std::move(object_property_key.key),
                                  object_property_key.computed, shorthand,
                                  std::move(value));
}

absl::StatusOr<std::unique_ptr<JsObjectProperty>>
JsirToAst::VisitObjectPropertyRef(JsirObjectPropertyRefOp op) {
  MALDOCA_ASSIGN_OR_RETURN(
      auto object_property_key,
      GetObjectPropertyKey(op.getComputedKey(), op.getLiteralKey()));

  bool shorthand = op.getShorthand();

  MALDOCA_ASSIGN_OR_RETURN(auto mlir_value,
                           Cast<JsirPatternRefOpInterface>(op.getValue()));
  MALDOCA_ASSIGN_OR_RETURN(auto value_pattern, VisitPatternRef(mlir_value));
  std::variant<std::unique_ptr<JsExpression>, std::unique_ptr<JsPattern>> value;
  if (dynamic_cast<JsExpression *>(value_pattern.get()) != nullptr) {
    value =
        absl::WrapUnique(dynamic_cast<JsExpression *>(value_pattern.release()));
  } else {
    value = std::move(value_pattern);
  }

  return Create<JsObjectProperty>(op, std::move(object_property_key.key),
                                  object_property_key.computed, shorthand,
                                  std::move(value));
}

absl::StatusOr<std::unique_ptr<JsObjectMethod>> JsirToAst::VisitObjectMethod(
    JsirObjectMethodOp op) {
  MALDOCA_ASSIGN_OR_RETURN(
      auto object_property_key,
      GetObjectPropertyKey(op.getComputedKey(), op.getLiteralKey()));

  std::optional<std::unique_ptr<JsIdentifier>> id;
  if (op.getId().has_value()) {
    MALDOCA_ASSIGN_OR_RETURN(id, VisitIdentifierAttr(*op.getId()));
  }

  std::vector<std::unique_ptr<JsPattern>> params;
  for (mlir::Value mlir_param_value : op.getParams()) {
    MALDOCA_ASSIGN_OR_RETURN(auto mlir_param,
                             Cast<JsirPatternRefOpInterface>(mlir_param_value));
    MALDOCA_ASSIGN_OR_RETURN(auto param, VisitPatternRef(mlir_param));
    params.push_back(std::move(param));
  }

  bool generator = op.getGenerator();
  bool async = op.getAsync();

  MALDOCA_ASSIGN_OR_RETURN(
      auto body,
      VisitStmtRegion(op.getBody(), &JsirToAst::VisitBlockStatement));

  auto kind = op.getKind().str();

  return Create<JsObjectMethod>(op, std::move(object_property_key.key),
                                object_property_key.computed, std::move(id),
                                std::move(params), generator, async,
                                std::move(body), std::move(kind));
}

absl::StatusOr<std::unique_ptr<JsObjectExpression>>
JsirToAst::VisitObjectExpression(JsirObjectExpressionOp op) {
  MALDOCA_ASSIGN_OR_RETURN(auto mlir_properties_values,
                           GetExprsRegionValues(op.getRegion()));
  std::vector<std::variant<std::unique_ptr<JsObjectProperty>,
                           std::unique_ptr<JsObjectMethod>,
                           std::unique_ptr<JsSpreadElement>>>
      properties;
  for (mlir::Value mlir_property_value : mlir_properties_values) {
    std::variant<std::unique_ptr<JsObjectProperty>,
                 std::unique_ptr<JsObjectMethod>,
                 std::unique_ptr<JsSpreadElement>>
        property;
    if (auto mlir_property = llvm::dyn_cast<JsirObjectPropertyOp>(
            mlir_property_value.getDefiningOp())) {
      MALDOCA_ASSIGN_OR_RETURN(property, VisitObjectProperty(mlir_property));
    } else if (auto mlir_property = llvm::dyn_cast<JsirObjectMethodOp>(
                   mlir_property_value.getDefiningOp())) {
      MALDOCA_ASSIGN_OR_RETURN(property, VisitObjectMethod(mlir_property));
    } else if (auto mlir_property = llvm::dyn_cast<JsirSpreadElementOp>(
                   mlir_property_value.getDefiningOp())) {
      MALDOCA_ASSIGN_OR_RETURN(property, VisitSpreadElement(mlir_property));
    } else {
      return absl::InvalidArgumentError(
          "properties must be ObjectProperty or ObjectMethod or "
          "SpreadElement.");
    }
    properties.push_back(std::move(property));
  }
  return Create<JsObjectExpression>(op, std::move(properties));
}

absl::StatusOr<JsirToAst::MemberExpressionProperty>
JsirToAst::GetMemberExpressionProperty(
    mlir::Value computed_property,
    std::optional<mlir::Attribute> literal_property) {
  if (computed_property != nullptr) {
    MALDOCA_ASSIGN_OR_RETURN(
        auto mlir_property, Cast<JsirExpressionOpInterface>(computed_property));
    MALDOCA_ASSIGN_OR_RETURN(auto property, VisitExpression(mlir_property));
    return MemberExpressionProperty{
        .property = std::move(property),
        .computed = true,
    };
  } else if (literal_property.has_value()) {
    std::variant<std::unique_ptr<JsExpression>, std::unique_ptr<JsPrivateName>>
        property;
    if (auto mlir_literal_property =
            mlir::dyn_cast<JsirIdentifierAttr>(literal_property.value())) {
      MALDOCA_ASSIGN_OR_RETURN(property,
                               VisitIdentifierAttr(mlir_literal_property));
    } else if (auto mlir_literal_property = mlir::dyn_cast<JsirPrivateNameAttr>(
                   literal_property.value())) {
      MALDOCA_ASSIGN_OR_RETURN(property,
                               VisitPrivateNameAttr(mlir_literal_property));
    } else {
      return absl::InvalidArgumentError(
          "literal_property must be Identifier or PrivateName.");
    }
    return MemberExpressionProperty{
        .property = std::move(property),
        .computed = false,
    };
  } else {
    return absl::InvalidArgumentError(
        "One of computed_property and literal_property must exist.");
  }
}

absl::StatusOr<
    std::variant<std::unique_ptr<JsExpression>, std::unique_ptr<JsSuper>>>
JsirToAst::GetMemberExpressionObject(mlir::Value object) {
  if (auto mlir_object =
          llvm::dyn_cast<JsirExpressionOpInterface>(object.getDefiningOp())) {
    return VisitExpression(mlir_object);
  } else if (auto mlir_object =
                 llvm::dyn_cast<JsirSuperOp>(object.getDefiningOp())) {
    return VisitSuper(mlir_object);
  } else {
    return absl::InvalidArgumentError("object must be Expression or Super.");
  }
}

absl::StatusOr<std::unique_ptr<JsMemberExpression>>
JsirToAst::VisitMemberExpression(JsirMemberExpressionOp op) {
  MALDOCA_ASSIGN_OR_RETURN(auto object,
                           GetMemberExpressionObject(op.getObject()));
  MALDOCA_ASSIGN_OR_RETURN(
      auto member_expression_property,
      GetMemberExpressionProperty(op.getComputedProperty(),
                                  op.getLiteralProperty()));
  return Create<JsMemberExpression>(
      op, std::move(object), std::move(member_expression_property.property),
      member_expression_property.computed);
}

absl::StatusOr<std::unique_ptr<JsMemberExpression>>
JsirToAst::VisitMemberExpressionRef(JsirMemberExpressionRefOp op) {
  MALDOCA_ASSIGN_OR_RETURN(auto object,
                           GetMemberExpressionObject(op.getObject()));
  MALDOCA_ASSIGN_OR_RETURN(
      auto member_expression_property,
      GetMemberExpressionProperty(op.getComputedProperty(),
                                  op.getLiteralProperty()));
  return Create<JsMemberExpression>(
      op, std::move(object), std::move(member_expression_property.property),
      member_expression_property.computed);
}

absl::StatusOr<std::unique_ptr<JsOptionalMemberExpression>>
JsirToAst::VisitOptionalMemberExpression(JsirOptionalMemberExpressionOp op) {
  MALDOCA_ASSIGN_OR_RETURN(auto mlir_object,
                           Cast<JsirExpressionOpInterface>(op.getObject()));
  MALDOCA_ASSIGN_OR_RETURN(auto object, VisitExpression(mlir_object));

  MALDOCA_ASSIGN_OR_RETURN(
      auto member_expression_property,
      GetMemberExpressionProperty(op.getComputedProperty(),
                                  op.getLiteralProperty()));
  bool optional = op.getOptional();
  return Create<JsOptionalMemberExpression>(
      op, std::move(object), std::move(member_expression_property.property),
      member_expression_property.computed, optional);
}

absl::StatusOr<std::unique_ptr<JsParenthesizedExpression>>
JsirToAst::VisitParenthesizedExpression(JsirParenthesizedExpressionOp op) {
  MALDOCA_ASSIGN_OR_RETURN(auto expression_op,
                           Cast<JsirExpressionOpInterface>(op.getExpression()));
  MALDOCA_ASSIGN_OR_RETURN(std::unique_ptr<JsExpression> expression,
                           VisitExpression(expression_op));
  return Create<JsParenthesizedExpression>(op, std::move(expression));
}

absl::StatusOr<std::unique_ptr<JsParenthesizedExpression>>
JsirToAst::VisitParenthesizedExpressionRef(
    JsirParenthesizedExpressionRefOp op) {
  mlir::Operation *expression_op = op.getExpression().getDefiningOp();
  std::unique_ptr<JsExpression> expression;
  if (auto lval_op = llvm::dyn_cast<JsirLValRefOpInterface>(expression_op)) {
    MALDOCA_ASSIGN_OR_RETURN(std::unique_ptr<JsLVal> lval,
                             VisitLValRef(lval_op));

    // Convert JsLVal to JsExpression
    if (dynamic_cast<JsExpression *>(lval.get())) {
      expression =
          absl::WrapUnique(dynamic_cast<JsExpression *>(lval.release()));
    } else {
      return absl::InvalidArgumentError("cannot convert LVal to Expression");
    }

  } else {
    MALDOCA_ASSIGN_OR_RETURN(auto rval_op,
                             Cast<JsirExpressionOpInterface>(expression_op));

    MALDOCA_ASSIGN_OR_RETURN(expression, VisitExpression(rval_op));
  }

  return Create<JsParenthesizedExpression>(op, std::move(expression));
}

absl::StatusOr<std::unique_ptr<JsClassMethod>> JsirToAst::VisitClassMethod(
    JsirClassMethodOp op) {
  std::optional<std::unique_ptr<JsIdentifier>> id;
  if (op.getId().has_value()) {
    MALDOCA_ASSIGN_OR_RETURN(id, VisitIdentifierAttr(*op.getId()));
  }

  std::vector<std::unique_ptr<JsPattern>> params;
  for (mlir::Value mlir_param_value : op.getParams()) {
    MALDOCA_ASSIGN_OR_RETURN(auto mlir_param,
                             Cast<JsirPatternRefOpInterface>(mlir_param_value));
    MALDOCA_ASSIGN_OR_RETURN(auto param, VisitPatternRef(mlir_param));
    params.push_back(std::move(param));
  }

  bool generator = op.getGenerator();
  bool async = op.getAsync();

  MALDOCA_ASSIGN_OR_RETURN(
      auto body,
      VisitStmtRegion(op.getBody(), &JsirToAst::VisitBlockStatement));

  MALDOCA_ASSIGN_OR_RETURN(
      auto object_property_key,
      GetObjectPropertyKey(op.getComputedKey(), op.getLiteralKey()));

  std::string kind = op.getKind().str();
  bool static_ = op.getStatic_();

  return Create<JsClassMethod>(
      op, std::move(id), std::move(params), generator, async, std::move(body),
      std::move(object_property_key.key), std::move(kind),
      object_property_key.computed, static_);
}

absl::StatusOr<std::unique_ptr<JsClassPrivateMethod>>
JsirToAst::VisitClassPrivateMethod(JsirClassPrivateMethodOp op) {
  std::optional<std::unique_ptr<JsIdentifier>> id;
  if (op.getId().has_value()) {
    MALDOCA_ASSIGN_OR_RETURN(id, VisitIdentifierAttr(*op.getId()));
  }

  std::vector<std::unique_ptr<JsPattern>> params;
  for (mlir::Value mlir_param_value : op.getParams()) {
    MALDOCA_ASSIGN_OR_RETURN(auto mlir_param,
                             Cast<JsirPatternRefOpInterface>(mlir_param_value));
    MALDOCA_ASSIGN_OR_RETURN(auto param, VisitPatternRef(mlir_param));
    params.push_back(std::move(param));
  }

  bool generator = op.getGenerator();
  bool async = op.getAsync();

  MALDOCA_ASSIGN_OR_RETURN(
      auto body,
      VisitStmtRegion(op.getBody(), &JsirToAst::VisitBlockStatement));

  MALDOCA_ASSIGN_OR_RETURN(auto key, VisitPrivateNameAttr(op.getKey()));

  std::string kind = op.getKind().str();
  bool static_ = op.getStatic_();

  return Create<JsClassPrivateMethod>(
      op, std::move(id), std::move(params), generator, async, std::move(body),
      std::move(key), std::move(kind), static_, /*computed=*/absl::nullopt);
}

absl::StatusOr<std::unique_ptr<JsClassProperty>> JsirToAst::VisitClassProperty(
    JsirClassPropertyOp op) {
  MALDOCA_ASSIGN_OR_RETURN(
      auto object_property_key,
      GetObjectPropertyKey(op.getComputedKey(), op.getLiteralKey()));
  bool static_ = op.getStatic_();
  std::optional<std::unique_ptr<JsExpression>> value;
  if (!op.getValue().empty()) {
    MALDOCA_ASSIGN_OR_RETURN(
        value, VisitExprRegion(op.getValue(), &JsirToAst::VisitExpression));
  }
  return Create<JsClassProperty>(op, std::move(object_property_key.key),
                                 std::move(value), static_,
                                 object_property_key.computed);
}

absl::StatusOr<std::variant<std::unique_ptr<JsIdentifier>,
                            std::unique_ptr<JsStringLiteral>>>
JsirToAst::GetIdentifierOrStringLiteral(mlir::Attribute attr) {
  if (auto identifier = mlir::dyn_cast<JsirIdentifierAttr>(attr)) {
    return VisitIdentifierAttr(identifier);
  } else if (auto string_literal =
                 mlir::dyn_cast<JsirStringLiteralAttr>(attr)) {
    return VisitStringLiteralAttr(string_literal);
  } else {
    return absl::InvalidArgumentError("Must be Identifier or StringLiteral.");
  }
}

absl::StatusOr<std::unique_ptr<JsImportSpecifier>>
JsirToAst::VisitImportSpecifierAttr(JsirImportSpecifierAttr attr) {
  MALDOCA_ASSIGN_OR_RETURN(auto imported,
                           GetIdentifierOrStringLiteral(attr.getImported()));
  MALDOCA_ASSIGN_OR_RETURN(auto local, VisitIdentifierAttr(attr.getLocal()));
  return Create<JsImportSpecifier>(attr, std::move(imported), std::move(local));
}

absl::StatusOr<std::unique_ptr<JsImportDefaultSpecifier>>
JsirToAst::VisitImportDefaultSpecifierAttr(
    JsirImportDefaultSpecifierAttr attr) {
  MALDOCA_ASSIGN_OR_RETURN(auto local, VisitIdentifierAttr(attr.getLocal()));
  return Create<JsImportDefaultSpecifier>(attr, std::move(local));
}

absl::StatusOr<std::unique_ptr<JsImportNamespaceSpecifier>>
JsirToAst::VisitImportNamespaceSpecifierAttr(
    JsirImportNamespaceSpecifierAttr attr) {
  MALDOCA_ASSIGN_OR_RETURN(auto local, VisitIdentifierAttr(attr.getLocal()));
  return Create<JsImportNamespaceSpecifier>(attr, std::move(local));
}

absl::StatusOr<std::unique_ptr<JsImportAttribute>>
JsirToAst::VisitImportAttributeAttr(JsirImportAttributeAttr attr) {
  MALDOCA_ASSIGN_OR_RETURN(auto key, VisitIdentifierAttr(attr.getKey()));
  MALDOCA_ASSIGN_OR_RETURN(auto value, VisitStringLiteralAttr(attr.getValue()));
  return Create<JsImportAttribute>(attr, std::move(key), std::move(value));
}

absl::StatusOr<std::unique_ptr<JsExportSpecifier>>
JsirToAst::VisitExportSpecifierAttr(JsirExportSpecifierAttr attr) {
  MALDOCA_ASSIGN_OR_RETURN(auto exported,
                           GetIdentifierOrStringLiteral(attr.getExported()));
  MALDOCA_ASSIGN_OR_RETURN(auto local,
                           GetIdentifierOrStringLiteral(attr.getLocal()));
  return Create<JsExportSpecifier>(attr, std::move(exported), std::move(local));
}

absl::StatusOr<std::unique_ptr<JsExportDefaultDeclaration>>
JsirToAst::VisitExportDefaultDeclaration(JsirExportDefaultDeclarationOp op) {
  MALDOCA_ASSIGN_OR_RETURN(mlir::Operation * mlir_declaration,
                           GetStmtRegionOperation(op.getDeclaration()));
  std::variant<std::unique_ptr<JsFunctionDeclaration>,
               std::unique_ptr<JsClassDeclaration>,
               std::unique_ptr<JsExpression>>
      declaration;
  if (auto mlir_declaration_op =
          llvm::dyn_cast<JsirFunctionDeclarationOp>(mlir_declaration)) {
    MALDOCA_ASSIGN_OR_RETURN(declaration,
                             VisitFunctionDeclaration(mlir_declaration_op));
  } else if (auto mlir_declaration_op =
                 llvm::dyn_cast<JsirClassDeclarationOp>(mlir_declaration)) {
    MALDOCA_ASSIGN_OR_RETURN(declaration,
                             VisitClassDeclaration(mlir_declaration_op));
  } else if (auto mlir_declaration_region_end_op =
                 llvm::dyn_cast<JsirExprRegionEndOp>(mlir_declaration)) {
    auto mlir_declaration_op = llvm::dyn_cast<JsirExpressionOpInterface>(
        mlir_declaration_region_end_op.getArgument().getDefiningOp());
    MALDOCA_ASSIGN_OR_RETURN(declaration, VisitExpression(mlir_declaration_op));
  } else {
    return absl::InvalidArgumentError(
        "Invalid JsirExportDefaultDeclarationOp::declaration()");
  }
  return Create<JsExportDefaultDeclaration>(op, std::move(declaration));
}

absl::StatusOr<std::unique_ptr<JsComment>> JsirToAst::VisitCommentAttr(
    JsirCommentAttrInterface attr) {
  if (auto comment_line_attr = mlir::dyn_cast<JsirCommentLineAttr>(attr)) {
    JsirLocationAttr mlir_loc = comment_line_attr.getLoc();

    auto start = JsirPositionAttr2JsPosition(mlir_loc.getStart());
    auto end = JsirPositionAttr2JsPosition(mlir_loc.getEnd());
    std::optional<std::string> identifier_name;
    if (mlir::StringAttr mlir_identifier_name = mlir_loc.getIdentifierName()) {
      identifier_name = mlir_identifier_name.str();
    }
    auto loc = std::make_unique<JsSourceLocation>(
        std::move(start), std::move(end), std::move(identifier_name));

    return std::make_unique<JsCommentLine>(
        std::move(loc), comment_line_attr.getValue().str(),
        *mlir_loc.getStartIndex(), *mlir_loc.getEndIndex());

  } else if (auto comment_block_attr =
                 mlir::dyn_cast<JsirCommentBlockAttr>(attr)) {
    JsirLocationAttr mlir_loc = comment_block_attr.getLoc();

    auto start = JsirPositionAttr2JsPosition(mlir_loc.getStart());
    auto end = JsirPositionAttr2JsPosition(mlir_loc.getEnd());
    std::optional<std::string> identifier_name;
    if (mlir::StringAttr mlir_identifier_name = mlir_loc.getIdentifierName()) {
      identifier_name = mlir_identifier_name.str();
    }
    auto loc = std::make_unique<JsSourceLocation>(
        std::move(start), std::move(end), std::move(identifier_name));

    return std::make_unique<JsCommentBlock>(
        std::move(loc), comment_block_attr.getValue().str(),
        *mlir_loc.getStartIndex(), *mlir_loc.getEndIndex());

  } else {
    return absl::InvalidArgumentError("Must be CommentLine or CommentBlock.");
  }
}

}  // namespace maldoca
