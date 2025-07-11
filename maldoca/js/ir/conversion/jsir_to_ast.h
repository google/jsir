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

#ifndef MALDOCA_JS_IR_CONVERSION_JSIR_TO_AST_H_
#define MALDOCA_JS_IR_CONVERSION_JSIR_TO_AST_H_

#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "llvm/Support/Casting.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "maldoca/base/status_macros.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/ir/ir.h"
#include "maldoca/js/ir/trivia.h"

namespace maldoca {

class JsirToAst {
 public:
  explicit JsirToAst() = default;

// Example:
//
// absl::StatusOr<std::unique_ptr<JsFile>> VisitFile(JsirFileOp op);
#define DECLARE_CIR_OP_VISIT_FUNCTION(TYPE) \
  absl::StatusOr<std::unique_ptr<Js##TYPE>> Visit##TYPE(Jsir##TYPE##Op op);

// Example:
//
// absl::StatusOr<std::unique_ptr<JsBlockStatement>>
// VisitBlockStatement(JshirBlockStatementOp op);
#define DECLARE_HIR_OP_VISIT_FUNCTION(TYPE) \
  absl::StatusOr<std::unique_ptr<Js##TYPE>> Visit##TYPE(Jshir##TYPE##Op op);

// Example:
//
// absl::StatusOr<std::unique_ptr<JsIdentifier>>
// VisitIdentifierRef(JsirIdentifierRefOp op);
#define DECLARE_REF_OP_VISIT_FUNCTION(TYPE)                   \
  absl::StatusOr<std::unique_ptr<Js##TYPE>> Visit##TYPE##Ref( \
      Jsir##TYPE##RefOp op);

// Example:
//
// absl::StatusOr<std::unique_ptr<JsIdentifier>>
// VisitIdentifierAttr(JsirIdentifierAttr attr);
#define DECLARE_ATTRIB_VISIT_FUNCTION(TYPE)                    \
  absl::StatusOr<std::unique_ptr<Js##TYPE>> Visit##TYPE##Attr( \
      Jsir##TYPE##Attr attr);

  FOR_EACH_JSIR_CLASS(DECLARE_CIR_OP_VISIT_FUNCTION,
                      DECLARE_HIR_OP_VISIT_FUNCTION,
                      /*LIR_OP=*/JSIR_CLASS_IGNORE,
                      DECLARE_REF_OP_VISIT_FUNCTION,
                      DECLARE_ATTRIB_VISIT_FUNCTION)

#undef DECLARE_CIR_OP_VISIT_FUNCTION
#undef DECLARE_REF_OP_VISIT_FUNCTION
#undef DECLARE_HIR_OP_VISIT_FUNCTION
#undef DECLARE_ATTRIB_VISIT_FUNCTION

  absl::StatusOr<std::unique_ptr<JsStatement>> VisitStatement(
      JsirStatementOpInterface op);

  absl::StatusOr<std::unique_ptr<JsModuleDeclaration>> VisitModuleDeclaration(
      JsirModuleDeclarationOpInterface op);

  absl::StatusOr<std::unique_ptr<JsExpression>> VisitExpression(
      JsirExpressionOpInterface op);

  absl::StatusOr<std::unique_ptr<JsPattern>> VisitPatternRef(
      JsirPatternRefOpInterface op);

  absl::StatusOr<std::unique_ptr<JsLVal>> VisitLValRef(
      JsirLValRefOpInterface op);

  absl::StatusOr<std::unique_ptr<JsLiteral>> VisitLiteral(
      JsirLiteralOpInterface op);

  absl::StatusOr<std::unique_ptr<JsDeclaration>> VisitDeclaration(
      JsirDeclarationOpInterface op);

  struct JsForInOfStatementFields {
    std::variant<std::unique_ptr<JsVariableDeclaration>,
                 std::unique_ptr<JsLVal>>
        left;
    std::unique_ptr<JsExpression> right;
    std::unique_ptr<JsStatement> body;
  };

  absl::StatusOr<JsForInOfStatementFields> VisitForInOfStatement(
      std::optional<JsirForInOfDeclarationAttr> left_declaration,
      mlir::Value left_lval, mlir::Value right, mlir::Region &body_region);

  absl::StatusOr<std::unique_ptr<JsModuleSpecifier>> VisitModuleSpecifierAttr(
      JsirModuleSpecifierAttrInterface attr);

  absl::StatusOr<std::unique_ptr<JsComment>> VisitCommentAttr(
      JsirCommentAttrInterface attr);

 private:
  template <typename NodeT, typename IrT,
            typename = std::enable_if_t<std::is_base_of_v<JsNode, NodeT>>,
            typename... Args>
  std::unique_ptr<NodeT> Create(IrT op, Args &&...args) {
    CHECK(op != nullptr) << "Op cannot be null.";
    JsTrivia trivia = GetJsTrivia(op);
    return CreateJsNodeWithTrivia<NodeT>(std::move(trivia),
                                         std::forward<Args>(args)...);
  }

  template <typename NodeT, typename IrT,
            typename = std::enable_if_t<!std::is_base_of_v<JsNode, NodeT>>,
            typename = void,  // deduplication
            typename... Args>
  std::unique_ptr<NodeT> Create(IrT op, Args &&...args) {
    CHECK(op != nullptr) << "Op cannot be null.";
    return absl::make_unique<NodeT>(std::forward<Args>(args)...);
  }

  absl::StatusOr<std::vector<std::unique_ptr<JsStatement>>>
  VisitStatementRegion(mlir::Region &region) {
    if (!region.hasOneBlock()) {
      return absl::InvalidArgumentError(
          "Region should have exactly one block.");
    }

    mlir::Block &block = region.getBlocks().front();
    std::vector<std::unique_ptr<JsStatement>> statements;
    for (mlir::Operation &op : block) {
      if (!llvm::isa<JsirStatementOpInterface>(op)) {
        continue;
      }
      auto statement_op = llvm::cast<JsirStatementOpInterface>(op);
      MALDOCA_ASSIGN_OR_RETURN(auto statement, VisitStatement(statement_op));
      statements.push_back(std::move(statement));
    }

    return statements;
  }

  struct ObjectPropertyKey {
    std::unique_ptr<JsExpression> key;
    bool computed;
  };

  absl::StatusOr<ObjectPropertyKey> GetObjectPropertyKey(
      mlir::Value computed_key, std::optional<mlir::Attribute> literal_key);

  struct MemberExpressionProperty {
    std::variant<std::unique_ptr<JsExpression>, std::unique_ptr<JsPrivateName>>
        property;
    bool computed;
  };

  absl::StatusOr<MemberExpressionProperty> GetMemberExpressionProperty(
      mlir::Value computed_property,
      std::optional<mlir::Attribute> literal_property);

  absl::StatusOr<
      std::variant<std::unique_ptr<JsExpression>, std::unique_ptr<JsSuper>>>
      GetMemberExpressionObject(mlir::Value);

  absl::StatusOr<std::variant<std::unique_ptr<JsIdentifier>,
                              std::unique_ptr<JsStringLiteral>>>
  GetIdentifierOrStringLiteral(mlir::Attribute attr);

  template <typename NodeType, typename OpType>
  using VisitFunc =
      absl::StatusOr<std::unique_ptr<NodeType>> (JsirToAst::*)(OpType);

  template <typename NodeT, typename OpT>
  absl::StatusOr<std::unique_ptr<NodeT>> VisitExprRegion(
      mlir::Region &region, VisitFunc<NodeT, OpT> visit) {
    MALDOCA_ASSIGN_OR_RETURN(mlir::Value mlir_value,
                             GetExprRegionValue(region));
    auto mlir_op = llvm::dyn_cast<OpT>(mlir_value.getDefiningOp());
    if (mlir_op == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrCat("Must be ", __PRETTY_FUNCTION__));
    }
    return (this->*visit)(mlir_op);
  }

  absl::StatusOr<mlir::Value> GetExprRegionValue(mlir::Region &region) {
    if (!region.hasOneBlock()) {
      return absl::InvalidArgumentError(
          "Region should have exactly one block.");
    }
    mlir::Block &block = region.front();
    if (block.empty()) {
      return absl::InvalidArgumentError("Block cannot be empty.");
    }
    auto expr_region_end = llvm::dyn_cast<JsirExprRegionEndOp>(block.back());
    if (expr_region_end == nullptr) {
      return absl::InvalidArgumentError(
          "Block should end with JsirExprRegionEndOp.");
    }
    return expr_region_end.getArgument();
  }

  absl::StatusOr<mlir::ValueRange> GetExprsRegionValues(mlir::Region &region) {
    if (!region.hasOneBlock()) {
      return absl::InvalidArgumentError(
          "Region should have exactly one block.");
    }
    mlir::Block &block = region.front();
    if (block.empty()) {
      return absl::InvalidArgumentError("Block cannot be empty.");
    }
    auto exprs_region_end = llvm::dyn_cast<JsirExprsRegionEndOp>(block.back());
    if (exprs_region_end == nullptr) {
      return absl::InvalidArgumentError(
          "Block should end with JsirExprsRegionEndOp.");
    }
    return exprs_region_end.getArguments();
  }

  absl::StatusOr<mlir::Operation *> GetStmtRegionOperation(
      mlir::Region &region) {
    if (!region.hasOneBlock()) {
      return absl::InvalidArgumentError(
          "Region should have exactly one block.");
    }
    mlir::Block &block = region.front();
    if (block.empty()) {
      return absl::InvalidArgumentError("Block cannot be empty.");
    }
    return &block.back();
  }

  template <typename NodeT, typename OpT>
  absl::StatusOr<std::unique_ptr<NodeT>> VisitStmtRegion(
      mlir::Region &region, VisitFunc<NodeT, OpT> visit) {
    MALDOCA_ASSIGN_OR_RETURN(mlir::Operation * mlir_operation,
                             GetStmtRegionOperation(region));
    auto mlir_op = llvm::dyn_cast<OpT>(mlir_operation);
    if (mlir_op == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrCat("Must be ", __PRETTY_FUNCTION__));
    }
    return (this->*visit)(mlir_op);
  }

  absl::StatusOr<mlir::Block *> GetStmtsRegionBlock(mlir::Region &region) {
    if (!region.hasOneBlock()) {
      return absl::InvalidArgumentError(
          "Region should have exactly one block.");
    }
    mlir::Block &block = region.front();
    return &block;
  }

  template <typename Stmt, typename StmtOp, typename Expr, typename ExprOp>
  absl::StatusOr<std::variant<std::unique_ptr<Stmt>, std::unique_ptr<Expr>>>
  VisitStmtOrExprRegion(mlir::Region &region,
                        VisitFunc<Stmt, StmtOp> visit_stmt,
                        VisitFunc<Expr, ExprOp> visit_expr) {
    MALDOCA_ASSIGN_OR_RETURN(auto mlir_end, GetStmtRegionOperation(region));
    if (auto mlir_stmt_op = llvm::dyn_cast<StmtOp>(mlir_end)) {
      return (this->*visit_stmt)(mlir_stmt_op);
    } else if (auto mlir_end_op =
                   llvm::dyn_cast<JsirExprRegionEndOp>(mlir_end)) {
      auto mlir_expr_value = mlir_end_op.getArgument();
      auto mlir_expr_op =
          llvm::dyn_cast<ExprOp>(mlir_expr_value.getDefiningOp());
      return (this->*visit_expr)(mlir_expr_op);
    } else {
      return absl::InvalidArgumentError("Invalid op type.");
    }
  }
};

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_CONVERSION_JSIR_TO_AST_H_
