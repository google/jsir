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

#ifndef MALDOCA_ASTGEN_TEST_LIST_CONVERSION_AST_TO_LIIR_H_
#define MALDOCA_ASTGEN_TEST_LIST_CONVERSION_AST_TO_LIIR_H_

#include "mlir/IR/Builders.h"
#include "maldoca/astgen/test/list/ast.generated.h"
#include "maldoca/astgen/test/list/ir.h"

namespace maldoca {

class AstToLiir {
 public:
  static LiirClass1Op VisitClass1(mlir::OpBuilder& builder,
                                  const LiClass1* node);

  static LiirClass2Op VisitClass2(mlir::OpBuilder& builder,
                                  const LiClass2* node);

  static LiirSimpleListOp VisitSimpleList(mlir::OpBuilder& builder,
                                          const LiSimpleList* node);

  static LiirOptionalListOp VisitOptionalList(mlir::OpBuilder& builder,
                                              const LiOptionalList* node);

  static LiirListOfOptionalOp VisitListOfOptional(mlir::OpBuilder& builder,
                                                  const LiListOfOptional* node);

  static LiirListOfVariantOp VisitListOfVariant(mlir::OpBuilder& builder,
                                                const LiListOfVariant* node);

  static LiirOptionalListOfOptionalOp VisitOptionalListOfOptional(
      mlir::OpBuilder& builder, const LiOptionalListOfOptional* node);

  static LiirOptionalListOfVariantOp VisitOptionalListOfVariant(
      mlir::OpBuilder& builder, const LiOptionalListOfVariant* node);

  static LiirListOfOptionalVariantOp VisitListOfOptionalVariant(
      mlir::OpBuilder& builder, const LiListOfOptionalVariant* node);

  static LiirOptionalListOfOptionalVariantOp VisitOptionalListOfOptionalVariant(
      mlir::OpBuilder& builder, const LiOptionalListOfOptionalVariant* node);

 private:
  template <typename Op, typename Node, typename... Args>
  static Op CreateExpr(mlir::OpBuilder& builder, const Node* node,
                       Args&&... args) {
    return Op::create(builder, builder.getUnknownLoc(),
                      std::forward<Args>(args)...);
  }
};

}  // namespace maldoca

#endif  // MALDOCA_ASTGEN_TEST_LIST_CONVERSION_AST_TO_LIIR_H_
