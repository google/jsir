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

#ifndef MALDOCA_ASTGEN_TEST_LIST_CONVERSION_LIIR_TO_AST_H_
#define MALDOCA_ASTGEN_TEST_LIST_CONVERSION_LIIR_TO_AST_H_

#include <memory>

#include "mlir/IR/Operation.h"
#include "absl/status/statusor.h"
#include "maldoca/astgen/test/list/ast.generated.h"
#include "maldoca/astgen/test/list/ir.h"

namespace maldoca {

class LiirToAst {
 public:
  absl::StatusOr<std::unique_ptr<LiClass1>> VisitClass1(LiirClass1Op op);

  absl::StatusOr<std::unique_ptr<LiClass2>> VisitClass2(LiirClass2Op op);

  absl::StatusOr<std::unique_ptr<LiSimpleList>> VisitSimpleList(
      LiirSimpleListOp op);

  absl::StatusOr<std::unique_ptr<LiOptionalList>> VisitOptionalList(
      LiirOptionalListOp op);

  absl::StatusOr<std::unique_ptr<LiListOfOptional>> VisitListOfOptional(
      LiirListOfOptionalOp op);

  absl::StatusOr<std::unique_ptr<LiListOfVariant>> VisitListOfVariant(
      LiirListOfVariantOp op);

  absl::StatusOr<std::unique_ptr<LiOptionalListOfOptional>>
  VisitOptionalListOfOptional(LiirOptionalListOfOptionalOp op);

  absl::StatusOr<std::unique_ptr<LiOptionalListOfVariant>>
  VisitOptionalListOfVariant(LiirOptionalListOfVariantOp op);

  absl::StatusOr<std::unique_ptr<LiListOfOptionalVariant>>
  VisitListOfOptionalVariant(LiirListOfOptionalVariantOp op);

  absl::StatusOr<std::unique_ptr<LiOptionalListOfOptionalVariant>>
  VisitOptionalListOfOptionalVariant(LiirOptionalListOfOptionalVariantOp op);

 private:
  template <typename T, typename... Args>
  std::unique_ptr<T> Create(mlir::Operation *op, Args &&...args) {
    return absl::make_unique<T>(std::forward<Args>(args)...);
  }
};

}  // namespace maldoca

#endif  // MALDOCA_ASTGEN_TEST_LIST_CONVERSION_LIIR_TO_AST_H_
