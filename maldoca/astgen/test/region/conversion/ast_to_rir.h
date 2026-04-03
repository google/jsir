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

#ifndef MALDOCA_ASTGEN_TEST_REGION_CONVERSION_AST_TO_RIR_H_
#define MALDOCA_ASTGEN_TEST_REGION_CONVERSION_AST_TO_RIR_H_

#include <functional>
#include <optional>

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Region.h"
#include "maldoca/astgen/test/region/ast.generated.h"
#include "maldoca/astgen/test/region/ir.h"

namespace maldoca {

class AstToRir {
 public:
  static RirExprOp VisitExpr(mlir::OpBuilder& builder, const RExpr* node);

  static RirStmtOp VisitStmt(mlir::OpBuilder& builder, const RStmt* node);

  static RirNodeOp VisitNode(mlir::OpBuilder& builder, const RNode* node);

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

#endif  // MALDOCA_ASTGEN_TEST_REGION_CONVERSION_AST_TO_RIR_H_
