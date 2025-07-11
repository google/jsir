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

// NOTE:
// This code is adapted from mlir/lib/Transforms/SCCP.cpp

#include "maldoca/js/ir/transforms/constant_propagation/pass.h"

#include <cassert>
#include <vector>

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/FoldUtils.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/ir/analyses/constant_propagation/analysis.h"
#include "maldoca/js/ir/analyses/dataflow_analysis.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {

// Replaces all uses of the given value with a constant if the corresponding
// lattice represents a constant. Returns success if the value was replaced,
// failure otherwise.
static mlir::LogicalResult ReplaceUsesWithConstant(
    JsirDialect *jsir_dialect, JsirConstantPropagationAnalysis &analysis,
    mlir::OpBuilder &builder, mlir::OperationFolder &folder,
    mlir::Value value) {
  // If the value is not used, then there is no need to create a substitute
  // constant op.
  if (value.getUses().empty()) {
    return mlir::success();
  }

  JsirStateRef<JsirConstantPropagationValue> state_ref =
      analysis.GetStateAt(value);
  if (state_ref.value().IsUninitialized() || state_ref.value().IsUnknown()) {
    return mlir::failure();
  }
  mlir::Attribute constant_attr = **state_ref.value();
  mlir::Value constant_value =
      folder.getOrCreateConstant(builder.getInsertionBlock(), jsir_dialect,
                                 constant_attr, value.getType());

  if (constant_value == nullptr) {
    return mlir::failure();
  }
  value.replaceAllUsesWith(constant_value);
  return mlir::success();
}

mlir::LogicalResult PerformConstantPropagation(mlir::Operation *op,
                                               const BabelScopes &scopes) {
  mlir::DataFlowSolver solver;

  auto *analysis = solver.load<JsirConstantPropagationAnalysis>(&scopes);

  mlir::LogicalResult result = solver.initializeAndRun(op);
  if (mlir::failed(result)) {
    return result;
  }

  return PerformConstantPropagation(op, *analysis);
}

mlir::LogicalResult PerformConstantPropagation(
    mlir::Operation *op, JsirConstantPropagationAnalysis &analysis) {
  mlir::MLIRContext *context = op->getContext();
  auto *jsir_dialect = context->getLoadedDialect<JsirDialect>();

  mlir::DominanceInfo dominance_info{op};

  // Initialize the worklist.
  //
  // In general, we want to traverse the program in "reverse" order, so that we
  // can discover dead code earlier and skip some transforms.
  //
  // Imagine that we have code like this with the knowledge that `%arg` holds
  // the constant 100:
  //
  // ```
  // ^BB(%arg):
  //   %r0 = %arg + %arg
  //   %r1 = %r0 + %r0
  // ```
  //
  // The constant propagation analysis would first calculate the values for both
  // %r0 and %r1: %r0 = 200, %r1 = 400.
  //
  // Now, when performing the transformation, if we traverse the program in
  // order, then the following transformations happen in order:
  // +---------------------+  +--------------------+  +----------------+
  // |                     |  |    %c200 = 200     |  |    %c200 = 200 |
  // |                     |  |    ~~~~~~~~~~~     |  |                |
  // |                     |  |                    |  |    %c400 = 400 |
  // |                     |  |                    |  |    ~~~~~~~~~~~ |
  // |                     |  |    ...             |  |    ...         |
  // | ^BB(%arg):          |->| ^BB(%arg):         |->| ^BB(%arg):     |
  // |   %r0 = %arg + %arg |  |    %r0 = %c200     |  |    %r0 = %c200 |
  // |                     |  |          ~~~~~     |  |                |
  // |   %r1 = %r0 + %r0   |  |    %r1 = %r0 + %r0 |  |    %r1 = %c400 |
  // |                     |  |                    |  |          ~~~~~ |
  // +---------------------+  +--------------------+  +----------------+
  //
  // However, if we traverse the program in reverse order, then by replacing
  // `%r0 + %r0` with `%c400`, we have eliminated all uses of `%r0`. At this
  // point, we can safely ignore the transformation on `%arg + %arg`.
  // +---------------------+  +------------------------+
  // |                     |  |    %c400 = 400         |
  // |                     |  |    ~~~~~~~~~~~         |
  // |                     |  |    ...                 |
  // | ^BB(%arg):          |->| ^BB(%arg):             |
  // |   %r0 = %arg + %arg |  |    %r0 = %arg + %arg   |
  // |   %r1 = %r0 + %r0   |  |    %r1 = %c400         |
  // |                     |  |          ~~~~~         |
  // +---------------------+  +------------------------+
  llvm::SmallVector<mlir::Block *> worklist;
  auto add_to_worklist = [&](mlir::MutableArrayRef<mlir::Region> regions) {
    for (mlir::Region &region : regions) {
      switch (region.getBlocks().size()) {
        case 0:
          break;
        case 1:
          worklist.push_back(&region.front());
          break;
        default: {
          auto &dom_tree = dominance_info.getDomTree(&region);
          auto *dom_root_node = dom_tree.getRootNode();
          for (auto *node : llvm::post_order(dom_root_node)) {
            worklist.push_back(node->getBlock());
          }
        }
      }
    }
  };

  // An operation folder used to create and unique constants.
  mlir::OperationFolder folder(context);
  mlir::OpBuilder builder(context);

  add_to_worklist(op->getRegions());
  while (!worklist.empty()) {
    mlir::Block *block = worklist.pop_back_val();

    for (mlir::Operation &op :
         llvm::make_early_inc_range(llvm::reverse(*block))) {
      builder.setInsertionPoint(&op);

      // Replace any result with constants.
      bool replaced_all = true;
      for (mlir::Value res : op.getResults()) {
        replaced_all &= mlir::succeeded(ReplaceUsesWithConstant(
            jsir_dialect, analysis, builder, folder, res));
      }

      // If all of the results of the operation were replaced, try to erase
      // the operation completely.
      if (replaced_all && mlir::wouldOpBeTriviallyDead(&op)) {
        assert(op.use_empty() && "expected all uses to be replaced");
        op.erase();
        continue;
      }

      // Add any the regions of this operation to the worklist.
      // Note that if previous transforms have caused the op to be dead, there
      // is no need to traverse its regions, and this logic is skipped.
      add_to_worklist(op.getRegions());
    }

    // Replace any block arguments with constants.
    builder.setInsertionPointToStart(block);
    for (mlir::BlockArgument arg : block->getArguments())
      (void)ReplaceUsesWithConstant(jsir_dialect, analysis, builder, folder,
                                    arg);
  }

  return mlir::success();
}

}  // namespace maldoca
