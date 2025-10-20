#ifndef MALDOCA_ASTGEN_TEST_LAMBDA_CONVERSION_AST_TO_LAIR_H_
#define MALDOCA_ASTGEN_TEST_LAMBDA_CONVERSION_AST_TO_LAIR_H_

#include <functional>
#include <optional>

#include "mlir/IR/Builders.h"
#include "absl/cleanup/cleanup.h"
#include "maldoca/astgen/test/lambda/ast.generated.h"
#include "maldoca/astgen/test/lambda/ir.h"

namespace maldoca {

class AstToLair {
 public:
  explicit AstToLair(mlir::OpBuilder &builder) : builder_(builder) {}

  LairExpressionOpInterface VisitExpression(const LaExpression *node);

  LairVariableOp VisitVariable(const LaVariable *node);

  LairVariableRefOp VisitVariableRef(const LaVariable *node);

  LairFunctionDefinitionOp VisitFunctionDefinition(
      const LaFunctionDefinition *node);

  LairFunctionCallOp VisitFunctionCall(const LaFunctionCall *node);

 private:
  template <typename Op, typename JsNode, typename... Args>
  Op CreateExpr(const JsNode *node, Args &&...args) {
    return builder_.create<Op>(builder_.getUnknownLoc(),
                               std::forward<Args>(args)...);
  }

  template <typename Op, typename JsNode, typename... Args>
  Op CreateStmt(const JsNode *node, Args &&...args) {
    return builder_.create<Op>(builder_.getUnknownLoc(), mlir::TypeRange(),
                               std::forward<Args>(args)...);
  }

  void AppendNewBlockAndPopulate(mlir::Region &region,
                                 std::function<void()> populate) {
    // Save insertion point.
    // Will revert at the end.
    mlir::OpBuilder::InsertionGuard insertion_guard(builder_);

    // Insert new block and point builder to it.
    mlir::Block &block = region.emplaceBlock();
    builder_.setInsertionPointToStart(&block);

    populate();
  }

  mlir::OpBuilder &builder_;
};

}  // namespace maldoca

#endif  // MALDOCA_ASTGEN_TEST_LAMBDA_CONVERSION_AST_TO_LAIR_H_
