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

#ifndef MALDOCA_ASTGEN_TEST_VARIANT_CONVERSION_VIR_TO_AST_H_
#define MALDOCA_ASTGEN_TEST_VARIANT_CONVERSION_VIR_TO_AST_H_

#include <memory>

#include "mlir/IR/Operation.h"
#include "absl/status/statusor.h"
#include "maldoca/astgen/test/variant/ast.generated.h"
#include "maldoca/astgen/test/variant/ir.h"

namespace maldoca {

class VirToAst {
 public:
  absl::StatusOr<std::unique_ptr<VBaseClass>> VisitBaseClass(
      VirBaseClassOpInterface op);

  absl::StatusOr<std::unique_ptr<VDerivedClass1>> VisitDerivedClass1(
      VirDerivedClass1Op op);

  absl::StatusOr<std::unique_ptr<VDerivedClass2>> VisitDerivedClass2(
      VirDerivedClass2Op op);

  absl::StatusOr<std::unique_ptr<VNode>> VisitNode(VirNodeOp op);

 private:
  template <typename T, typename... Args>
  std::unique_ptr<T> Create(mlir::Operation *op, Args &&...args) {
    return absl::make_unique<T>(std::forward<Args>(args)...);
  }
};

}  // namespace maldoca

#endif  // MALDOCA_ASTGEN_TEST_VARIANT_CONVERSION_VIR_TO_AST_H_
