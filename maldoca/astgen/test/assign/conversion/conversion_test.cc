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
#include "maldoca/astgen/test/assign/ast.generated.h"
#include "maldoca/astgen/test/assign/conversion/air_to_ast.h"
#include "maldoca/astgen/test/assign/conversion/ast_to_air.h"
#include "maldoca/astgen/test/assign/ir.h"
#include "maldoca/astgen/test/conversion_test_util.h"

namespace maldoca {
namespace {

TEST(ConversionTest, SimpleAssignment) {
  // a = b
  constexpr char kAstJsonString[] = R"(
    {
      "type": "Assignment",
      "lhs": {
        "type": "Identifier",
        "name": "a"
      },
      "rhs": {
        "type": "Identifier",
        "name": "b"
      }
    }
  )";

  constexpr char kExpectedIrDump[] = R"(
module {
  %0 = "air.identifier_ref"() <{name = "a"}> : () -> !air.any
  %1 = "air.identifier"() <{name = "b"}> : () -> !air.any
  %2 = "air.assignment"(%0, %1) : (!air.any, !air.any) -> !air.any
}
  )";

  TestIrConversion<AAssignment, AirAssignmentOp, AirDialect, AstToAir,
                   AirToAst>({
      .ast_json_string = kAstJsonString,
      .ast_to_ir_visit = &AstToAir::VisitAssignment,
      .ir_to_ast_visit = &AirToAst::VisitAssignment,
      .expected_ir_dump = kExpectedIrDump,
  });
}

TEST(ConversionTest, ChainAssignment) {
  // a = (b = c)
  constexpr char kAstJsonString[] = R"(
    {
      "type": "Assignment",
      "lhs": {
        "type": "Identifier",
        "name": "a"
      },
      "rhs": {
        "type": "Assignment",
        "lhs": {
          "type": "Identifier",
          "name": "b"
        },
        "rhs": {
          "type": "Identifier",
          "name": "c"
        }
      }
    }
  )";

  constexpr char kExpectedIrDump[] = R"(
module {
  %0 = "air.identifier_ref"() <{name = "a"}> : () -> !air.any
  %1 = "air.identifier_ref"() <{name = "b"}> : () -> !air.any
  %2 = "air.identifier"() <{name = "c"}> : () -> !air.any
  %3 = "air.assignment"(%1, %2) : (!air.any, !air.any) -> !air.any
  %4 = "air.assignment"(%0, %3) : (!air.any, !air.any) -> !air.any
}
  )";

  TestIrConversion<AAssignment, AirAssignmentOp, AirDialect, AstToAir,
                   AirToAst>({
      .ast_json_string = kAstJsonString,
      .ast_to_ir_visit = &AstToAir::VisitAssignment,
      .ir_to_ast_visit = &AirToAst::VisitAssignment,
      .expected_ir_dump = kExpectedIrDump,
  });
}

}  // namespace
}  // namespace maldoca
