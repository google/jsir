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
#include "maldoca/astgen/test/variant/ast.generated.h"
#include "maldoca/astgen/test/variant/conversion/ast_to_vir.h"
#include "maldoca/astgen/test/variant/conversion/vir_to_ast.h"
#include "maldoca/astgen/test/variant/ir.h"

namespace maldoca {
namespace {

TEST(ConversionTest, Double) {
  constexpr char kAstJsonString[] = R"(
    {
      "simpleVariantBuiltin": 1,
      "nullableVariantBuiltin": 1,
      "optionalVariantBuiltin": 1,
      "simpleVariantClass": {
        "type": "DerivedClass1"
      },
      "nullableVariantClass": {
        "type": "DerivedClass1"
      },
      "optionalVariantClass": {
        "type": "DerivedClass2"
}
    }
  )";

  constexpr char kExpectedIr[] = R"(
module {
  %0 = "vir.derived_class1"() : () -> !vir.any
  %1 = "vir.derived_class1"() : () -> !vir.any
  %2 = "vir.derived_class2"() : () -> !vir.any
  %3 = "vir.node"(%0, %1, %2) <{nullable_variant_builtin = 1.000000e+00 : f64, operandSegmentSizes = array<i32: 1, 1, 1>, optional_variant_builtin = 1.000000e+00 : f64, simple_variant_builtin = 1.000000e+00 : f64}> : (!vir.any, !vir.any, !vir.any) -> !vir.any
}
  )";

  TestIrConversion<VNode, VirNodeOp, VirDialect, AstToVir, VirToAst>({
      .ast_json_string = kAstJsonString,
      .ast_to_ir_visit = &AstToVir::VisitNode,
      .ir_to_ast_visit = &VirToAst::VisitNode,
      .expected_ir_dump = kExpectedIr,
  });
}

TEST(ConversionTest, String) {
  constexpr char kAstJsonString[] = R"(
    {
      "simpleVariantBuiltin": "1",
      "nullableVariantBuiltin": "1",
      "optionalVariantBuiltin": "1",
      "simpleVariantClass": {
        "type": "DerivedClass1"
      },
      "nullableVariantClass": {
        "type": "DerivedClass1"
      },
      "optionalVariantClass": {
        "type": "DerivedClass2"
      }
    }
  )";

  constexpr char kExpectedIr[] = R"(
module {
  %0 = "vir.derived_class1"() : () -> !vir.any
  %1 = "vir.derived_class1"() : () -> !vir.any
  %2 = "vir.derived_class2"() : () -> !vir.any
  %3 = "vir.node"(%0, %1, %2) <{nullable_variant_builtin = "1", operandSegmentSizes = array<i32: 1, 1, 1>, optional_variant_builtin = "1", simple_variant_builtin = "1"}> : (!vir.any, !vir.any, !vir.any) -> !vir.any
}
  )";

  TestIrConversion<VNode, VirNodeOp, VirDialect, AstToVir, VirToAst>({
      .ast_json_string = kAstJsonString,
      .ast_to_ir_visit = &AstToVir::VisitNode,
      .ir_to_ast_visit = &VirToAst::VisitNode,
      .expected_ir_dump = kExpectedIr,
  });
}

TEST(ConversionTest, Nullopt) {
  constexpr char kAstJsonString[] = R"(
    {
      "simpleVariantBuiltin": "1",
      "nullableVariantBuiltin": null,
      "simpleVariantClass": {
        "type": "DerivedClass1"
      },
      "nullableVariantClass": null
    }
  )";

  constexpr char kExpectedIr[] = R"(
module {
  %0 = "vir.derived_class1"() : () -> !vir.any
  %1 = "vir.node"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0>, simple_variant_builtin = "1"}> : (!vir.any) -> !vir.any
}
  )";

  TestIrConversion<VNode, VirNodeOp, VirDialect, AstToVir, VirToAst>({
      .ast_json_string = kAstJsonString,
      .ast_to_ir_visit = &AstToVir::VisitNode,
      .ir_to_ast_visit = &VirToAst::VisitNode,
      .expected_ir_dump = kExpectedIr,
  });
}

}  // namespace
}  // namespace maldoca
