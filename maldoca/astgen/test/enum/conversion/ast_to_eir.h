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

#ifndef MALDOCA_ASTGEN_TEST_ENUM_CONVERSION_AST_TO_EIR_H_
#define MALDOCA_ASTGEN_TEST_ENUM_CONVERSION_AST_TO_EIR_H_

#include "mlir/IR/Builders.h"
#include "maldoca/astgen/test/enum/ast.generated.h"
#include "maldoca/astgen/test/enum/ir.h"

namespace maldoca {

class AstToEir {
 public:
  static EirNodeOp VisitNode(mlir::OpBuilder& builder, const ENode* node);

 private:
  template <typename Op, typename Node, typename... Args>
  static Op CreateExpr(mlir::OpBuilder& builder, const Node* node,
                       Args&&... args) {
    return Op::create(builder, builder.getUnknownLoc(),
                      EirAnyType::get(builder.getContext()),
                      std::forward<Args>(args)...);
  }
};

}  // namespace maldoca

#endif  // MALDOCA_ASTGEN_TEST_ENUM_CONVERSION_AST_TO_EIR_H_
