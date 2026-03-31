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

#ifndef MALDOCA_ASTGEN_TEST_VARIANT_CONVERSION_AST_TO_VIR_H_
#define MALDOCA_ASTGEN_TEST_VARIANT_CONVERSION_AST_TO_VIR_H_

#include "mlir/IR/Builders.h"
#include "maldoca/astgen/test/variant/ast.generated.h"
#include "maldoca/astgen/test/variant/ir.h"

namespace maldoca {

class AstToVir {
 public:
  static VirBaseClassOpInterface VisitBaseClass(mlir::OpBuilder& builder,
                                                const VBaseClass* node);

  static VirDerivedClass1Op VisitDerivedClass1(mlir::OpBuilder& builder,
                                               const VDerivedClass1* node);

  static VirDerivedClass2Op VisitDerivedClass2(mlir::OpBuilder& builder,
                                               const VDerivedClass2* node);

  static VirNodeOp VisitNode(mlir::OpBuilder& builder, const VNode* node);

 private:
  template <typename Op, typename VNode, typename... Args>
  static Op CreateExpr(mlir::OpBuilder& builder, const VNode* node,
                       Args&&... args) {
    return Op::create(builder, builder.getUnknownLoc(),
                      std::forward<Args>(args)...);
  }
};

}  // namespace maldoca

#endif  // MALDOCA_ASTGEN_TEST_VARIANT_CONVERSION_AST_TO_VIR_H_
