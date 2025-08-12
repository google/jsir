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

#ifndef MALDOCA_ASTGEN_TEST_REGION_CONVERSION_RIR_TO_AST_H_
#define MALDOCA_ASTGEN_TEST_REGION_CONVERSION_RIR_TO_AST_H_

#include <memory>
#include <utility>

#include "llvm/Support/Casting.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "maldoca/astgen/test/region/ast.generated.h"
#include "maldoca/astgen/test/region/ir.h"

namespace maldoca {

class RirToAst {
 public:
  absl::StatusOr<std::unique_ptr<RExpr>> VisitExpr(RirExprOp op);

  absl::StatusOr<std::unique_ptr<RStmt>> VisitStmt(RirStmtOp op);

  absl::StatusOr<std::unique_ptr<RNode>> VisitNode(RirNodeOp op);

 private:
  template <typename T, typename... Args>
  std::unique_ptr<T> Create(mlir::Operation *op, Args &&...args) {
    return absl::make_unique<T>(std::forward<Args>(args)...);
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
    auto expr_region_end = llvm::dyn_cast<RirExprRegionEndOp>(block.back());
    if (expr_region_end == nullptr) {
      return absl::InvalidArgumentError(
          "Block should end with RirExprRegionEndOp.");
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
    auto exprs_region_end = llvm::dyn_cast<RirExprsRegionEndOp>(block.back());
    if (exprs_region_end == nullptr) {
      return absl::InvalidArgumentError(
          "Block should end with RirExprsRegionEndOp.");
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

  absl::StatusOr<mlir::Block *> GetStmtsRegionBlock(mlir::Region &region) {
    if (!region.hasOneBlock()) {
      return absl::InvalidArgumentError(
          "Region should have exactly one block.");
    }
    mlir::Block &block = region.front();
    return &block;
  }
};

}  // namespace maldoca

#endif  // MALDOCA_ASTGEN_TEST_REGION_CONVERSION_RIR_TO_AST_H_
