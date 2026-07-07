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

#include "maldoca/js/ir/transforms/remove_directives/pass.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {

// #!/usr/bin/env babel-node  // (1) interpreter directive
//
// "use strict";              // (2) file-level directive
//
// function foo() {
//   "use strict";            // (3) block-level directive
//   a;
// }

void RemoveDirectives(mlir::Operation *root) {
  mlir::OpBuilder builder(root);

  root->walk([&](mlir::Operation* op) {
    llvm::TypeSwitch<mlir::Operation*, void>(op)
        .Case([&](JsirProgramOp op) {
          // Remove (1)
          op.removeInterpreterAttr();

          // Remove (2)
          for (mlir::Block &block : op.getDirectives()) {
            block.clear();
          }
        })

        .Case([&](JsirBlockStatementOp op) {
          // Remove (3)
          for (mlir::Block &block : op.getDirectives()) {
            block.clear();
          }
        })

        .Default([&](mlir::Operation* op) {});
  });
}

}  // namespace maldoca
