#ifndef MALDOCA_ASTGEN_TEST_LAMBDA_CONVERSION_AST_TO_LAIR_H_
#define MALDOCA_ASTGEN_TEST_LAMBDA_CONVERSION_AST_TO_LAIR_H_

#include <functional>

#include "mlir/IR/Builders.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Region.h"
#include "maldoca/astgen/test/lambda/ast.generated.h"
#include "maldoca/astgen/test/lambda/ir.h"

namespace maldoca {

class AstToLair {
 public:
  static LairExpressionOpInterface VisitExpression(mlir::OpBuilder& builder,
                                                   const LaExpression* node);

  static LairVariableOp VisitVariable(mlir::OpBuilder& builder,
                                      const LaVariable* node);

  static LairVariableRefOp VisitVariableRef(mlir::OpBuilder& builder,
                                            const LaVariable* node);

  static LairFunctionDefinitionOp VisitFunctionDefinition(
      mlir::OpBuilder& builder, const LaFunctionDefinition* node);

  static LairFunctionCallOp VisitFunctionCall(mlir::OpBuilder& builder,
                                              const LaFunctionCall* node);

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

#endif  // MALDOCA_ASTGEN_TEST_LAMBDA_CONVERSION_AST_TO_LAIR_H_
