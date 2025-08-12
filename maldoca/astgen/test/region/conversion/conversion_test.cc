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

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "absl/strings/ascii.h"
#include "absl/types/optional.h"
#include "absl/types/variant.h"
#include "maldoca/astgen/test/conversion_test_util.h"
#include "maldoca/astgen/test/region/ast.generated.h"
#include "maldoca/astgen/test/region/conversion/ast_to_rir.h"
#include "maldoca/astgen/test/region/conversion/rir_to_ast.h"
#include "maldoca/astgen/test/region/ir.h"

namespace maldoca {
namespace {

TEST(ConversionTest, ConversionTest) {
  constexpr char kAstJsonString[] = R"(
    {
      "expr": {},
      "exprs": [
        {}
      ],
      "optionalExpr": null,
      "optionalStmt": null,
      "stmt": {
        "expr": {}
      },
      "stmts": [
        {
          "expr": {}
        }
      ]
    }
  )";

  constexpr char kExpectedIr[] = R"(
module {
  %0 = "rir.node"() ({
    %1 = "rir.expr"() : () -> !rir.any
    "rir.expr_region_end"(%1) : (!rir.any) -> ()
  }, {
  }, {
    %1 = "rir.expr"() : () -> !rir.any
    "rir.exprs_region_end"(%1) : (!rir.any) -> ()
  }, {
    %1 = "rir.expr"() : () -> !rir.any
    "rir.stmt"(%1) : (!rir.any) -> ()
  }, {
  }, {
    %1 = "rir.expr"() : () -> !rir.any
    "rir.stmt"(%1) : (!rir.any) -> ()
  }) : () -> !rir.any
}
  )";

  TestIrConversion<RNode, RirNodeOp, RirDialect, AstToRir, RirToAst>({
      .ast_json_string = kAstJsonString,
      .ast_to_ir_visit = &AstToRir::VisitNode,
      .ir_to_ast_visit = &RirToAst::VisitNode,
      .expected_ir_dump = kExpectedIr,
  });
}

}  // namespace
}  // namespace maldoca
