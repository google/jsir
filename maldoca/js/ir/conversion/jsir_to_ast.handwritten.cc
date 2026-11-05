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
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
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
  ABSL_ASSIGN_OR_RETURN(auto id, VisitIdentifierAttr(attr.getId()));
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
  std::optional<std::unique_ptr<JsStringLiteralExtra>> extra;
  if (attr.getExtra() != nullptr) {
    ABSL_ASSIGN_OR_RETURN(extra, VisitStringLiteralExtraAttr(attr.getExtra()));
  }
  return Create<JsStringLiteral>(attr, attr.getValue().str(), std::move(extra));
}

absl::StatusOr<std::unique_ptr<JsNumericLiteralExtra>>
JsirToAst::VisitNumericLiteralExtraAttr(JsirNumericLiteralExtraAttr attr) {
  return Create<JsNumericLiteralExtra>(attr, attr.getRaw().str(),
                                       attr.getRawValue().getValueAsDouble());
}

absl::StatusOr<std::unique_ptr<JsNumericLiteral>>
JsirToAst::VisitNumericLiteralAttr(JsirNumericLiteralAttr attr) {
  std::optional<std::unique_ptr<JsNumericLiteralExtra>> extra;
  if (attr.getExtra() != nullptr) {
    ABSL_ASSIGN_OR_RETURN(extra, VisitNumericLiteralExtraAttr(attr.getExtra()));
  }
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
  std::optional<std::unique_ptr<JsBigIntLiteralExtra>> extra;
  if (attr.getExtra() != nullptr) {
    ABSL_ASSIGN_OR_RETURN(extra, VisitBigIntLiteralExtraAttr(attr.getExtra()));
  }
  return Create<JsBigIntLiteral>(attr, attr.getValue().str(), std::move(extra));
}

absl::StatusOr<std::unique_ptr<JsBreakStatement>>
JsirToAst::VisitBreakStatement(JshirBreakStatementOp op) {
  std::optional<std::unique_ptr<JsIdentifier>> label;
  if (op.getLabel().has_value()) {
    ABSL_ASSIGN_OR_RETURN(label, VisitIdentifierAttr(op.getLabel().value()));
  }
  return Create<JsBreakStatement>(op, std::move(label));
}

absl::StatusOr<std::unique_ptr<JsContinueStatement>>
JsirToAst::VisitContinueStatement(JshirContinueStatementOp op) {
  std::optional<std::unique_ptr<JsIdentifier>> label;
  if (op.getLabel().has_value()) {
    ABSL_ASSIGN_OR_RETURN(label, VisitIdentifierAttr(op.getLabel().value()));
  }
  return Create<JsContinueStatement>(op, std::move(label));
}

absl::StatusOr<std::unique_ptr<JsObjectExpression>>
JsirToAst::VisitObjectExpression(JsirObjectExpressionOp op) {
  ABSL_ASSIGN_OR_RETURN(auto mlir_properties_values,
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
      ABSL_ASSIGN_OR_RETURN(property, VisitObjectProperty(mlir_property));
    } else if (auto mlir_property = llvm::dyn_cast<JsirObjectMethodOp>(
                   mlir_property_value.getDefiningOp())) {
      ABSL_ASSIGN_OR_RETURN(property, VisitObjectMethod(mlir_property));
    } else if (auto mlir_property = llvm::dyn_cast<JsirSpreadElementOp>(
                   mlir_property_value.getDefiningOp())) {
      ABSL_ASSIGN_OR_RETURN(property, VisitSpreadElement(mlir_property));
    } else {
      return absl::InvalidArgumentError(
          "properties must be ObjectProperty or ObjectMethod or "
          "SpreadElement.");
    }
    properties.push_back(std::move(property));
  }
  return Create<JsObjectExpression>(op, std::move(properties));
}

absl::StatusOr<std::unique_ptr<JsParenthesizedExpression>>
JsirToAst::VisitParenthesizedExpression(JsirParenthesizedExpressionOp op) {
  ABSL_ASSIGN_OR_RETURN(auto expression_op,
                        Cast<JsirExpressionOpInterface>(op.getExpression()));
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<JsExpression> expression,
                        VisitExpression(expression_op));
  return Create<JsParenthesizedExpression>(op, std::move(expression));
}

absl::StatusOr<std::unique_ptr<JsParenthesizedExpression>>
JsirToAst::VisitParenthesizedExpressionRef(
    JsirParenthesizedExpressionRefOp op) {
  mlir::Operation* expression_op = op.getExpression().getDefiningOp();
  std::unique_ptr<JsExpression> expression;
  if (auto lval_op = llvm::dyn_cast<JsirLValRefOpInterface>(expression_op)) {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<JsLVal> lval, VisitLValRef(lval_op));

    // Convert JsLVal to JsExpression
    if (dynamic_cast<JsExpression*>(lval.get())) {
      expression =
          absl::WrapUnique(dynamic_cast<JsExpression*>(lval.release()));
    } else {
      return absl::InvalidArgumentError("cannot convert LVal to Expression");
    }

  } else {
    ABSL_ASSIGN_OR_RETURN(auto rval_op,
                          Cast<JsirExpressionOpInterface>(expression_op));

    ABSL_ASSIGN_OR_RETURN(expression, VisitExpression(rval_op));
  }

  return Create<JsParenthesizedExpression>(op, std::move(expression));
}

absl::StatusOr<std::unique_ptr<JsClassPrivateMethod>>
JsirToAst::VisitClassPrivateMethod(JsirClassPrivateMethodOp op) {
  std::optional<std::unique_ptr<JsIdentifier>> id;
  if (op.getId().has_value()) {
    ABSL_ASSIGN_OR_RETURN(id, VisitIdentifierAttr(*op.getId()));
  }

  std::vector<std::unique_ptr<JsPattern>> params;
  ABSL_ASSIGN_OR_RETURN(mlir::ValueRange mlir_params,
                        GetExprsRegionValues(op.getParams()));
  for (mlir::Value mlir_param_value : mlir_params) {
    ABSL_ASSIGN_OR_RETURN(auto mlir_param,
                          Cast<JsirPatternRefOpInterface>(mlir_param_value));
    ABSL_ASSIGN_OR_RETURN(auto param, VisitPatternRef(mlir_param));
    params.push_back(std::move(param));
  }

  bool generator = op.getGenerator();
  bool async = op.getAsync();

  ABSL_ASSIGN_OR_RETURN(
      auto body,
      VisitStmtRegion(op.getBody(), &JsirToAst::VisitBlockStatement));

  ABSL_ASSIGN_OR_RETURN(auto key, VisitPrivateNameAttr(op.getKey()));

  std::string kind = op.getKind().str();
  bool static_ = op.getStatic_();

  return Create<JsClassPrivateMethod>(
      op, std::move(id), std::move(params), generator, async, std::move(body),
      std::move(key), std::move(kind), static_, /*computed=*/std::nullopt);
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
  ABSL_ASSIGN_OR_RETURN(auto imported,
                        GetIdentifierOrStringLiteral(attr.getImported()));
  ABSL_ASSIGN_OR_RETURN(auto local, VisitIdentifierAttr(attr.getLocal()));
  return Create<JsImportSpecifier>(attr, std::move(imported), std::move(local));
}

absl::StatusOr<std::unique_ptr<JsImportDefaultSpecifier>>
JsirToAst::VisitImportDefaultSpecifierAttr(
    JsirImportDefaultSpecifierAttr attr) {
  ABSL_ASSIGN_OR_RETURN(auto local, VisitIdentifierAttr(attr.getLocal()));
  return Create<JsImportDefaultSpecifier>(attr, std::move(local));
}

absl::StatusOr<std::unique_ptr<JsImportNamespaceSpecifier>>
JsirToAst::VisitImportNamespaceSpecifierAttr(
    JsirImportNamespaceSpecifierAttr attr) {
  ABSL_ASSIGN_OR_RETURN(auto local, VisitIdentifierAttr(attr.getLocal()));
  return Create<JsImportNamespaceSpecifier>(attr, std::move(local));
}

absl::StatusOr<std::unique_ptr<JsImportAttribute>>
JsirToAst::VisitImportAttributeAttr(JsirImportAttributeAttr attr) {
  ABSL_ASSIGN_OR_RETURN(auto key, VisitIdentifierAttr(attr.getKey()));
  ABSL_ASSIGN_OR_RETURN(auto value, VisitStringLiteralAttr(attr.getValue()));
  return Create<JsImportAttribute>(attr, std::move(key), std::move(value));
}

absl::StatusOr<std::unique_ptr<JsExportSpecifier>>
JsirToAst::VisitExportSpecifier(JsirExportSpecifierOp op) {
  ABSL_ASSIGN_OR_RETURN(auto exported,
                        GetIdentifierOrStringLiteral(op.getExportedAttr()));

  std::optional<std::variant<std::unique_ptr<JsIdentifier>,
                             std::unique_ptr<JsStringLiteral>>>
      local;
  if (mlir::Value mlir_local = op.getLocal(); mlir_local != nullptr) {
    mlir::Operation* mlir_local_op = mlir_local.getDefiningOp();
    if (auto identifier_ref =
            llvm::dyn_cast_or_null<JsirIdentifierRefOp>(mlir_local_op)) {
      ABSL_ASSIGN_OR_RETURN(local, VisitIdentifierRef(identifier_ref));
    } else if (auto string_literal =
                   llvm::dyn_cast_or_null<JsirStringLiteralOp>(
                       mlir_local_op)) {
      ABSL_ASSIGN_OR_RETURN(local, VisitStringLiteral(string_literal));
    } else if (!llvm::isa_and_nonnull<JsirNoneOp>(mlir_local_op)) {
      return absl::InvalidArgumentError(
          "JsirExportSpecifierOp::local must be defined by "
          "JsirIdentifierRefOp or JsirStringLiteralOp.");
    }
  }
  return Create<JsExportSpecifier>(op, std::move(exported), std::move(local));
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
