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

#ifndef MALDOCA_JS_IR_JSIR_UTILS_H_
#define MALDOCA_JS_IR_JSIR_UTILS_H_

#include <vector>

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "maldoca/base/status_macros.h"
#include "maldoca/js/ir/cast.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {

// ============================================================================
//  Functions to extract content from regions
// ============================================================================

// Extracts operation from a region.
absl::StatusOr<mlir::Operation *> GetStmtRegionOperation(mlir::Region &region);

// Extracts operation from a region and converts it to OpT.
template <typename OpT>
absl::StatusOr<OpT> GetStmtRegionOp(mlir::Region &region);

// Extracts Value from a region.
absl::StatusOr<mlir::Value> GetExprRegionValue(mlir::Region &region);

// Extracts Value from a region and converts it to OpT.
template <typename OpT>
absl::StatusOr<OpT> GetExprRegionOp(mlir::Region &region);

// Extracts Block from a region.
absl::StatusOr<mlir::Block *> GetStmtsRegionBlock(mlir::Region &region);

// Extracts the last operation from a region of expressions.
absl::StatusOr<JsirExprsRegionEndOp> GetExprsRegionEndOp(mlir::Region &region);

// Extracts ValueRange from a region.
absl::StatusOr<mlir::ValueRange> GetExprsRegionValues(mlir::Region &region);

template <typename OpT>
absl::StatusOr<OpT> GetStmtRegionOp(mlir::Region &region) {
  MALDOCA_ASSIGN_OR_RETURN(mlir::Operation * op,
                           GetStmtRegionOperation(region));
  return Cast<OpT>(op);
}

template <typename OpT>
absl::StatusOr<OpT> GetExprRegionOp(mlir::Region &region) {
  MALDOCA_ASSIGN_OR_RETURN(mlir::Value value, GetExprRegionValue(region));
  return Cast<OpT>(value.getDefiningOp());
}

// ============================================================================
//  Operation-filtering functions
// ============================================================================

template <typename OpT>
auto FilterBlockOps(mlir::Block &block) {
  auto filter_range = llvm::make_filter_range(
      block, [](mlir::Operation& op) { return llvm::isa<OpT>(op); });

  return llvm::map_range(std::move(filter_range), [](mlir::Operation &op) {
    return llvm::cast<OpT>(op);
  });
}

// ============================================================================
//  Block-manipulation functions
// ============================================================================

// Infers if this block should contain a single statement.
bool IsStatementBlock(mlir::Block &block);

// TODO(tzx) Implement a standalone pass to add `jshir.block_statement`.
//
// We shouldn't require each individual pass to maintain the invariant that
// certain `mlir::Block`s should only contain a single statement - a
// `mlir::Block` should always allow multiple statements, and we should
// automatically add `jshir.block_statement`s when lifting JSHIR to AST.
void WrapBlockContentWithBlockStatement(mlir::Block &block);

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_JSIR_UTILS_H_
