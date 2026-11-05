// Copyright 2026 Google LLC
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

#ifndef MALDOCA_ASTGEN_TEST_SPLIT_CONVERSION_AST_TO_SIR_H_
#define MALDOCA_ASTGEN_TEST_SPLIT_CONVERSION_AST_TO_SIR_H_

#include <functional>
#include <utility>

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Region.h"
#include "maldoca/astgen/test/split/ast.generated.h"
#include "maldoca/astgen/test/split/ir.h"

namespace maldoca {

class AstToSir {
 public:
  static SirExpressionOpInterface VisitExpression(mlir::OpBuilder& builder,
                                                  const SExpression* node);

  static SirPatternRefOpInterface VisitPatternRef(mlir::OpBuilder& builder,
                                                  const SPattern* node);

  static SirStatementOpInterface VisitStatement(mlir::OpBuilder& builder,
                                                const SStatement* node);

  static SirIdentifierOp VisitIdentifier(mlir::OpBuilder& builder,
                                         const SIdentifier* node);

  static SirIdentifierRefOp VisitIdentifierRef(mlir::OpBuilder& builder,
                                               const SIdentifier* node);

  static SirIdentifierAttr VisitIdentifierAttr(mlir::OpBuilder& builder,
                                               const SIdentifier* node) {
    return SirIdentifierAttr::get(builder.getContext(),
                                  builder.getStringAttr(node->name()));
  }

  static SirCallOp VisitCall(mlir::OpBuilder& builder, const SCall* node);

  static SirDeclarationOp VisitDeclaration(mlir::OpBuilder& builder,
                                           const SDeclaration* node);

  static SirExpressionStatementOp VisitExpressionStatement(
      mlir::OpBuilder& builder, const SExpressionStatement* node);

  static SirMemberOp VisitMember(mlir::OpBuilder& builder, const SMember* node);

  static SirPropertyOp VisitProperty(mlir::OpBuilder& builder,
                                     const SProperty* node);

  static SirPropertyRefOp VisitPropertyRef(mlir::OpBuilder& builder,
                                           const SProperty* node);

  static SirLoopOp VisitLoop(mlir::OpBuilder& builder, const SLoop* node);

 private:
  template <typename Op, typename... Args>
  static Op CreateExpr(mlir::OpBuilder& builder, const void* node,
                       Args&&... args) {
    return Op::create(builder, builder.getUnknownLoc(),
                      std::forward<Args>(args)...);
  }

  template <typename Op, typename... Args>
  static Op CreateStmt(mlir::OpBuilder& builder, const void* node,
                       Args&&... args) {
    return Op::create(builder, builder.getUnknownLoc(), mlir::TypeRange(),
                      std::forward<Args>(args)...);
  }

  static void AppendNewBlockAndPopulate(mlir::OpBuilder& builder,
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
};

}  // namespace maldoca

#endif  // MALDOCA_ASTGEN_TEST_SPLIT_CONVERSION_AST_TO_SIR_H_
