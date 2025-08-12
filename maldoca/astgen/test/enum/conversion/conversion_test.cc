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

#include "gtest/gtest.h"
#include "maldoca/astgen/test/conversion_test_util.h"
#include "maldoca/astgen/test/enum/ast.generated.h"
#include "maldoca/astgen/test/enum/conversion/ast_to_eir.h"
#include "maldoca/astgen/test/enum/conversion/eir_to_ast.h"
#include "maldoca/astgen/test/enum/ir.h"

namespace maldoca {
namespace {

TEST(ConversionTest, Enum) {
  constexpr char kAstJsonString[] = R"(
    {
      "unaryOperator": "+",
      "escapedChar": "\\"
    }
  )";

  constexpr char kExpectedIrDump[] = R"(
module {
  %0 = "eir.node"() <{escaped_char = "\\", unary_operator = "+"}> : () -> !eir.any
}
  )";

  TestIrConversion<ENode, EirNodeOp, EirDialect, AstToEir, EirToAst>({
      .ast_json_string = kAstJsonString,
      .ast_to_ir_visit = &AstToEir::VisitNode,
      .ir_to_ast_visit = &EirToAst::VisitNode,
      .expected_ir_dump = kExpectedIrDump,
  });
}

}  // namespace
}  // namespace maldoca
