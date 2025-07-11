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

#include "maldoca/js/ir/transforms/split_declaration_statements/pass.h"

#include <vector>

#include "llvm/Support/Casting.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "absl/status/status.h"
#include "maldoca/base/status_macros.h"
#include "maldoca/js/ir/ir.h"
#include "maldoca/js/ir/jsir_utils.h"

namespace maldoca {

void GetDependencyOps(mlir::Operation *root,
                      std::vector<mlir::Operation *> &dependency_ops) {
  for (mlir::Value operand : root->getOperands()) {
    mlir::Operation *operand_op = operand.getDefiningOp();
    if (operand_op == nullptr) {
      continue;
    }
    GetDependencyOps(operand_op, dependency_ops);
  }
  dependency_ops.push_back(root);
}

std::vector<mlir::Operation *> GetDependencyOps(mlir::Operation *root) {
  std::vector<mlir::Operation *> dependency_ops;
  GetDependencyOps(root, dependency_ops);
  return dependency_ops;
}

bool BlockIsForHeader(mlir::Block &block) {
  auto for_op =
      llvm::dyn_cast_if_present<JshirForStatementOp>(block.getParentOp());
  if (for_op == nullptr) {
    return false;
  }

  auto *region = block.getParent();
  if (region == nullptr) {
    return false;
  }

  return &for_op.getInit() == region || &for_op.getTest() == region ||
         &for_op.getUpdate() == region;
}

void SplitDeclarationStatements(mlir::Operation *root) {
  std::vector<mlir::Block *> modified_blocks;
  root->walk([&](JsirVariableDeclarationOp declaration_op) {
    mlir::Block *parent_block = declaration_op->getBlock();

    if (BlockIsForHeader(*parent_block)) {
      // Skip
      return;
    }

    MALDOCA_ASSIGN_OR_RETURN(
        JsirExprsRegionEndOp declarators_op,
        GetExprsRegionEndOp(declaration_op.getDeclarations()),
        _.With([](const absl::Status &) { return; }));

    if (declarators_op.getArguments().size() <= 1) {
      return;
    }

    modified_blocks.push_back(parent_block);

    mlir::MLIRContext *context = root->getContext();
    mlir::OpBuilder builder{context};

    while (declarators_op.getArguments().size() > 1) {
      // jsir.variable_declaration() <{kind = "const"}> ({ <- declaration_op
      //   %0 = jsir.identifier_ref() <{name = "a"}>
      //   %1 = jsir.numeric_literal() <{value = 2}>
      //   %2 = jsir.variable_declarator(%0, %1)           <- declarator_op
      //   %3 = jsir.identifier_ref() <{name = "b"}>
      //   %4 = jsir.numeric_literal() <{value = 3}>
      //   %5 = jsir.variable_declarator(%3, %4)
      //   %6 = jsir.identifier_ref() <{name = "c"}>
      //   %7 = jsir.numeric_literal() <{value = 4}>
      //   %8 = jsir.variable_declarator(%6, %7)
      //   jsir.exprs_region_end(%2, %5, %8)               <- erase %2 here
      // })
      mlir::Value declarator_value = declarators_op.getArguments().front();
      JsirVariableDeclaratorOp declarator_op =
          declarator_value.getDefiningOp<JsirVariableDeclaratorOp>();
      if (declarator_op == nullptr) {
        break;
      }
      declarators_op.getArgumentsMutable().erase(0);

      // jsir.variable_declaration() <{kind = "const"}> ({ <- new_declaration_op
      // })
      // jsir.variable_declaration() <{kind = "const"}> ({ <- declaration_op
      //   %0 = jsir.identifier_ref() <{name = "a"}>
      //   %1 = jsir.numeric_literal() <{value = 2}>
      //   %2 = jsir.variable_declarator(%0, %1)
      //   %3 = jsir.identifier_ref() <{name = "b"}>
      //   %4 = jsir.numeric_literal() <{value = 3}>
      //   %5 = jsir.variable_declarator(%3, %4)
      //   %6 = jsir.identifier_ref() <{name = "c"}>
      //   %7 = jsir.numeric_literal() <{value = 4}>
      //   %8 = jsir.variable_declarator(%6, %7)
      //   jsir.exprs_region_end(%5, %8)
      // })
      builder.setInsertionPoint(declaration_op);
      auto new_declaration_op = JsirVariableDeclarationOp::create(
          builder, declaration_op.getLoc(), declaration_op.getKind());

      // jsir.variable_declaration() <{kind = "const"}> ({ <- new_declaration_op
      // })
      // jsir.variable_declaration() <{kind = "const"}> ({ <- declaration_op
      //   %0 = jsir.identifier_ref() <{name = "a"}>       <- dependency_ops
      //   %1 = jsir.numeric_literal() <{value = 2}>       <- dependency_ops
      //   %2 = jsir.variable_declarator(%0, %1)           <- dependency_ops
      //                                                      declarator_op
      //   %3 = jsir.identifier_ref() <{name = "b"}>
      //   %4 = jsir.numeric_literal() <{value = 3}>
      //   %5 = jsir.variable_declarator(%3, %4)
      //   %6 = jsir.identifier_ref() <{name = "c"}>
      //   %7 = jsir.numeric_literal() <{value = 4}>
      //   %8 = jsir.variable_declarator(%6, %7)
      //   jsir.exprs_region_end(%5, %8)
      // })
      std::vector<mlir::Operation *> dependency_ops =
          GetDependencyOps(declarator_op);

      // jsir.variable_declaration() <{kind = "const"}> ({ <- new_declaration_op
      //   %0 = jsir.identifier_ref() <{name = "a"}>
      //   %1 = jsir.numeric_literal() <{value = 2}>
      //   %2 = jsir.variable_declarator(%0, %1)
      // })
      // jsir.variable_declaration() <{kind = "const"}> ({ <- declaration_op
      //   %3 = jsir.identifier_ref() <{name = "b"}>
      //   %4 = jsir.numeric_literal() <{value = 3}>
      //   %5 = jsir.variable_declarator(%3, %4)
      //   %6 = jsir.identifier_ref() <{name = "c"}>
      //   %7 = jsir.numeric_literal() <{value = 4}>
      //   %8 = jsir.variable_declarator(%6, %7)
      //   jsir.exprs_region_end(%5, %8)
      // })
      mlir::Block &block = new_declaration_op.getDeclarations().emplaceBlock();
      for (mlir::Operation *dependency_op : dependency_ops) {
        dependency_op->moveBefore(&block, block.end());
      }

      builder.setInsertionPointToEnd(&block);
      JsirExprsRegionEndOp::create(builder, declarator_op->getLoc(),
                                   mlir::ValueRange({declarator_op}));
    }
  });

  for (mlir::Block *block : modified_blocks) {
    if (IsStatementBlock(*block)) {
      WrapBlockContentWithBlockStatement(*block);
    }
  }
}

}  // namespace maldoca
