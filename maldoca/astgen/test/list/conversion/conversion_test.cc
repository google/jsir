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
#include "maldoca/astgen/test/list/ast.generated.h"
#include "maldoca/astgen/test/list/conversion/ast_to_liir.h"
#include "maldoca/astgen/test/list/conversion/liir_to_ast.h"
#include "maldoca/astgen/test/list/ir.h"

namespace maldoca {
namespace {

TEST(ConversionTest, OptionalListWithNullopt) {
  constexpr char kAstJsonString[] = R"(
    {}
  )";

  constexpr char kExpectedIr[] = R"(
module {
  %0 = "liir.optional_list"() : () -> !liir.any
}
  )";

  TestIrConversion<LiOptionalList, LiirOptionalListOp, LiirDialect, AstToLiir,
                   LiirToAst>({
      .ast_json_string = kAstJsonString,
      .ast_to_ir_visit = &AstToLiir::VisitOptionalList,
      .ir_to_ast_visit = &LiirToAst::VisitOptionalList,
      .expected_ir_dump = kExpectedIr,
  });
}

TEST(ConversionTest, OptionalList) {
  constexpr char kAstJsonString[] = R"(
    {
      "strings": [
        "a",
        "b"
      ]
    }
  )";

  constexpr char kExpectedIr[] = R"(
module {
  %0 = "liir.optional_list"() <{strings = ["a", "b"]}> : () -> !liir.any
}
  )";

  TestIrConversion<LiOptionalList, LiirOptionalListOp, LiirDialect, AstToLiir,
                   LiirToAst>({
      .ast_json_string = kAstJsonString,
      .ast_to_ir_visit = &AstToLiir::VisitOptionalList,
      .ir_to_ast_visit = &LiirToAst::VisitOptionalList,
      .expected_ir_dump = kExpectedIr,
  });
}

TEST(ConversionTest, ListOfVariant) {
  constexpr char kAstJsonString[] = R"(
    {
      "variants": [
        true,
        "true"
      ],
      "operations": [
        {}
      ]
    }
  )";

  constexpr char kExpectedIr[] = R"(
module {
  %0 = "liir.class1"() : () -> !liir.any
  %1 = "liir.list_of_variant"(%0) <{variants = [true, "true"]}> : (!liir.any) -> !liir.any
}
  )";

  TestIrConversion<LiListOfVariant, LiirListOfVariantOp, LiirDialect, AstToLiir,
                   LiirToAst>({
      .ast_json_string = kAstJsonString,
      .ast_to_ir_visit = &AstToLiir::VisitListOfVariant,
      .ir_to_ast_visit = &LiirToAst::VisitListOfVariant,
      .expected_ir_dump = kExpectedIr,
  });
}

// Disabled because null attributes are not supported.
TEST(ConversionTest, DISABLED_ListOfOptional) {
  constexpr char kAstJsonString[] = R"(
    {
      "strings": [
        "true",
        null
      ],
      "operations": []
    }
  )";

  constexpr char kExpectedIr[] = R"(
"builtin.module"() ({
  %0 = "liir.list_of_optional"() <{strings = ["true", <<NULL ATTRIBUTE>>]}> : () -> !liir.any
}) : () -> ()
  )";

  TestIrConversion<LiListOfOptional, LiirListOfOptionalOp, LiirDialect,
                   AstToLiir, LiirToAst>({
      .ast_json_string = kAstJsonString,
      .ast_to_ir_visit = &AstToLiir::VisitListOfOptional,
      .ir_to_ast_visit = &LiirToAst::VisitListOfOptional,
      .expected_ir_dump = kExpectedIr,
  });
}

}  // namespace
}  // namespace maldoca
