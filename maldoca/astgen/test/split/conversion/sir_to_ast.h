// Copyright 2026 Google LLC
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

#ifndef MALDOCA_ASTGEN_TEST_SPLIT_CONVERSION_SIR_TO_AST_H_
#define MALDOCA_ASTGEN_TEST_SPLIT_CONVERSION_SIR_TO_AST_H_

#include <memory>
#include <utility>

#include "mlir/IR/Operation.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "maldoca/astgen/test/split/ast.generated.h"
#include "maldoca/astgen/test/split/ir.h"

namespace maldoca {

class SirToAst {
 public:
  static absl::StatusOr<std::unique_ptr<SExpression>> VisitExpression(
      SirExpressionOpInterface op);

  static absl::StatusOr<std::unique_ptr<SPattern>> VisitPatternRef(
      SirPatternRefOpInterface op);

  static absl::StatusOr<std::unique_ptr<SStatement>> VisitStatement(
      SirStatementOpInterface op);

  static absl::StatusOr<std::unique_ptr<SIdentifier>> VisitIdentifier(
      SirIdentifierOp op);

  static absl::StatusOr<std::unique_ptr<SIdentifier>> VisitIdentifierRef(
      SirIdentifierRefOp op);

  static absl::StatusOr<std::unique_ptr<SIdentifier>> VisitIdentifierAttr(
      SirIdentifierAttr attr) {
    return absl::make_unique<SIdentifier>(attr.getName().str());
  }

  static absl::StatusOr<std::unique_ptr<SCall>> VisitCall(SirCallOp op);

  static absl::StatusOr<std::unique_ptr<SDeclaration>> VisitDeclaration(
      SirDeclarationOp op);

  static absl::StatusOr<std::unique_ptr<SExpressionStatement>>
  VisitExpressionStatement(SirExpressionStatementOp op);

  static absl::StatusOr<std::unique_ptr<SMember>> VisitMember(SirMemberOp op);

  static absl::StatusOr<std::unique_ptr<SProperty>> VisitProperty(
      SirPropertyOp op);

  static absl::StatusOr<std::unique_ptr<SProperty>> VisitPropertyRef(
      SirPropertyRefOp op);

  static absl::StatusOr<std::unique_ptr<SLoop>> VisitLoop(SirLoopOp op);

 private:
  template <typename T, typename... Args>
  static std::unique_ptr<T> Create(mlir::Operation* op, Args&&... args) {
    return absl::make_unique<T>(std::forward<Args>(args)...);
  }
};

}  // namespace maldoca

#endif  // MALDOCA_ASTGEN_TEST_SPLIT_CONVERSION_SIR_TO_AST_H_
