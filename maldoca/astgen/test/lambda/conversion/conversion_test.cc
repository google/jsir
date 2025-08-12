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
#include "maldoca/astgen/test/lambda/ast.generated.h"
#include "maldoca/astgen/test/lambda/conversion/ast_to_lair.h"
#include "maldoca/astgen/test/lambda/ir.h"

namespace maldoca {
namespace {

TEST(ConversionTest, SimpleFunctionDefinition) {
  // x => x
  constexpr char kAstJsonString[] = R"(
    {
      "body": {
        "identifier": "x",
        "type": "Variable"
      },
      "parameter": {
        "identifier": "x",
        "type": "Variable"
      },
      "type": "FunctionDefinition"
    }
  )";

  constexpr char kExpectedIrDump[] = R"(
module {
  %0 = "lair.function_definition"() ({
    %1 = "lair.variable_ref"() <{identifier = "x"}> : () -> !lair.any
    "lair.expr_region_end"(%1) : (!lair.any) -> ()
  }, {
    %1 = "lair.variable"() <{identifier = "x"}> : () -> !lair.any
    "lair.expr_region_end"(%1) : (!lair.any) -> ()
  }) : () -> !lair.any
}
  )";

  TestIrConversion<LaFunctionDefinition, LairFunctionDefinitionOp, LairDialect,
                   AstToLair>({
      .ast_json_string = kAstJsonString,
      .ast_to_ir_visit = &AstToLair::VisitFunctionDefinition,
      .expected_ir_dump = kExpectedIrDump,
  });
}

TEST(ConversionTest, FunctionCall) {
  // (x => x)(x)
  constexpr char kAstJsonString[] = R"(
    {
      "argument": {
        "identifier": "x",
        "type": "Variable"
      },
      "function": {
        "body": {
          "identifier": "x",
          "type": "Variable"
        },
        "parameter": {
          "identifier": "x",
          "type": "Variable"
        },
        "type": "FunctionDefinition"
      },
      "type": "FunctionCall"
    }
  )";

  const char kExpectedIrDump[] = R"(
module {
  %0 = "lair.function_definition"() ({
    %3 = "lair.variable_ref"() <{identifier = "x"}> : () -> !lair.any
    "lair.expr_region_end"(%3) : (!lair.any) -> ()
  }, {
    %3 = "lair.variable"() <{identifier = "x"}> : () -> !lair.any
    "lair.expr_region_end"(%3) : (!lair.any) -> ()
  }) : () -> !lair.any
  %1 = "lair.variable"() <{identifier = "x"}> : () -> !lair.any
  %2 = "lair.function_call"(%0, %1) : (!lair.any, !lair.any) -> !lair.any
}
  )";

  TestIrConversion<LaFunctionCall, LairFunctionCallOp, LairDialect, AstToLair>({
      .ast_json_string = kAstJsonString,
      .ast_to_ir_visit = &AstToLair::VisitFunctionCall,
      .expected_ir_dump = kExpectedIrDump,
  });
}

}  // namespace
}  // namespace maldoca
