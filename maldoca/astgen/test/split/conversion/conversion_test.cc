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

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "gtest/gtest.h"
#include "maldoca/astgen/test/conversion_test_util.h"
#include "maldoca/astgen/test/split/ast.generated.h"
#include "maldoca/astgen/test/split/conversion/ast_to_sir.h"
#include "maldoca/astgen/test/split/conversion/sir_to_ast.h"
#include "maldoca/astgen/test/split/ir.h"

namespace maldoca {
namespace {

// `when`/`unless`: a static member expression holds its property in the
// `literal_property` attribute.
TEST(ConversionTest, LiteralMember) {
  constexpr char kAstJsonString[] = R"(
    {
      "type": "Member",
      "object": {"type": "Identifier", "name": "a"},
      "property": {"type": "Identifier", "name": "b"},
      "computed": false
    }
  )";

  constexpr char kExpectedIr[] = R"(
module {
  %0 = "sir.identifier"() <{name = "a"}> : () -> !sir.any
  %1 = "sir.member"(%0) <{literal_property = #sir<identifier "b">}> : (!sir.any) -> !sir.any
}
  )";

  TestIrConversion<SMember, SirMemberOp, SirDialect, AstToSir>({
      .ast_json_string = kAstJsonString,
      .ast_to_ir_visit = AstToSir::VisitMember,
      .ir_to_ast_visit = &SirToAst::VisitMember,
      .expected_ir_dump = kExpectedIr,
  });
}

// `when`/`unless`: a computed member expression holds its property in the
// `computed_property` operand.
TEST(ConversionTest, ComputedMember) {
  constexpr char kAstJsonString[] = R"(
    {
      "type": "Member",
      "object": {"type": "Identifier", "name": "a"},
      "property": {
        "type": "Call",
        "callee": {"type": "Identifier", "name": "b"}
      },
      "computed": true
    }
  )";

  constexpr char kExpectedIr[] = R"(
module {
  %0 = "sir.identifier"() <{name = "a"}> : () -> !sir.any
  %1 = "sir.identifier"() <{name = "b"}> : () -> !sir.any
  %2 = "sir.call"(%1) : (!sir.any) -> !sir.any
  %3 = "sir.member"(%0, %2) : (!sir.any, !sir.any) -> !sir.any
}
  )";

  TestIrConversion<SMember, SirMemberOp, SirDialect, AstToSir>({
      .ast_json_string = kAstJsonString,
      .ast_to_ir_visit = AstToSir::VisitMember,
      .ir_to_ast_visit = &SirToAst::VisitMember,
      .expected_ir_dump = kExpectedIr,
  });
}

// A narrowed variant field: the rvalue op holds the `Expression` member.
TEST(ConversionTest, Property) {
  constexpr char kAstJsonString[] = R"(
    {
      "type": "Property",
      "value": {"type": "Identifier", "name": "a"}
    }
  )";

  constexpr char kExpectedIr[] = R"(
module {
  %0 = "sir.identifier"() <{name = "a"}> : () -> !sir.any
  %1 = "sir.property"(%0) : (!sir.any) -> !sir.any
}
  )";

  TestIrConversion<SProperty, SirPropertyOp, SirDialect, AstToSir>({
      .ast_json_string = kAstJsonString,
      .ast_to_ir_visit = AstToSir::VisitProperty,
      .ir_to_ast_visit = &SirToAst::VisitProperty,
      .expected_ir_dump = kExpectedIr,
  });
}

// A narrowed variant field: the lvalue op holds the `Pattern` member.
TEST(ConversionTest, PropertyRef) {
  constexpr char kAstJsonString[] = R"(
    {
      "type": "Property",
      "value": {"type": "Identifier", "name": "a"}
    }
  )";

  constexpr char kExpectedIr[] = R"(
module {
  %0 = "sir.identifier_ref"() <{name = "a"}> : () -> !sir.any
  %1 = "sir.property_ref"(%0) : (!sir.any) -> !sir.any
}
  )";

  TestIrConversion<SProperty, SirPropertyRefOp, SirDialect, AstToSir>({
      .ast_json_string = kAstJsonString,
      .ast_to_ir_visit = AstToSir::VisitPropertyRef,
      .ir_to_ast_visit = &SirToAst::VisitPropertyRef,
      .expected_ir_dump = kExpectedIr,
  });
}

// `split`: the declaration members are held in statement regions, and the
// expression/pattern members in an expression region and an operand.
TEST(ConversionTest, LoopWithDeclarationInitAndPatternLeft) {
  constexpr char kAstJsonString[] = R"(
    {
      "type": "Loop",
      "init": {
        "type": "Declaration",
        "id": {"type": "Identifier", "name": "i"}
      },
      "left": {"type": "Identifier", "name": "a"},
      "body": {
        "type": "ExpressionStatement",
        "expression": {"type": "Identifier", "name": "b"}
      }
    }
  )";

  constexpr char kExpectedIr[] = R"(
module {
  %0 = "sir.identifier_ref"() <{name = "a"}> : () -> !sir.any
  "sir.loop"(%0) ({
    "sir.declaration"() <{id = #sir<identifier "i">}> : () -> ()
  }, {
  }, {
  }, {
    %1 = "sir.identifier"() <{name = "b"}> : () -> !sir.any
    "sir.expression_statement"(%1) : (!sir.any) -> ()
  }) : (!sir.any) -> ()
}
  )";

  TestIrConversion<SLoop, SirLoopOp, SirDialect, AstToSir>({
      .ast_json_string = kAstJsonString,
      .ast_to_ir_visit = AstToSir::VisitLoop,
      .ir_to_ast_visit = &SirToAst::VisitLoop,
      .expected_ir_dump = kExpectedIr,
  });
}

TEST(ConversionTest, LoopWithExpressionInitAndDeclarationLeft) {
  constexpr char kAstJsonString[] = R"(
    {
      "type": "Loop",
      "init": {"type": "Identifier", "name": "i"},
      "left": {
        "type": "Declaration",
        "id": {"type": "Identifier", "name": "a"}
      },
      "body": {
        "type": "ExpressionStatement",
        "expression": {"type": "Identifier", "name": "b"}
      }
    }
  )";

  constexpr char kExpectedIr[] = R"(
module {
  "sir.loop"() ({
  }, {
    %0 = "sir.identifier"() <{name = "i"}> : () -> !sir.any
    "sir.expr_region_end"(%0) : (!sir.any) -> ()
  }, {
    "sir.declaration"() <{id = #sir<identifier "a">}> : () -> ()
  }, {
    %0 = "sir.identifier"() <{name = "b"}> : () -> !sir.any
    "sir.expression_statement"(%0) : (!sir.any) -> ()
  }) : () -> ()
}
  )";

  TestIrConversion<SLoop, SirLoopOp, SirDialect, AstToSir>({
      .ast_json_string = kAstJsonString,
      .ast_to_ir_visit = AstToSir::VisitLoop,
      .ir_to_ast_visit = &SirToAst::VisitLoop,
      .expected_ir_dump = kExpectedIr,
  });
}

TEST(ConversionTest, LoopWithoutInit) {
  constexpr char kAstJsonString[] = R"(
    {
      "type": "Loop",
      "init": null,
      "left": {"type": "Identifier", "name": "a"},
      "body": {
        "type": "ExpressionStatement",
        "expression": {"type": "Identifier", "name": "b"}
      }
    }
  )";

  constexpr char kExpectedIr[] = R"(
module {
  %0 = "sir.identifier_ref"() <{name = "a"}> : () -> !sir.any
  "sir.loop"(%0) ({
  }, {
  }, {
  }, {
    %1 = "sir.identifier"() <{name = "b"}> : () -> !sir.any
    "sir.expression_statement"(%1) : (!sir.any) -> ()
  }) : (!sir.any) -> ()
}
  )";

  TestIrConversion<SLoop, SirLoopOp, SirDialect, AstToSir>({
      .ast_json_string = kAstJsonString,
      .ast_to_ir_visit = AstToSir::VisitLoop,
      .ir_to_ast_visit = &SirToAst::VisitLoop,
      .expected_ir_dump = kExpectedIr,
  });
}

// `op_kind` on a class field: the lvalue op holds the `Pattern`.
TEST(ConversionTest, ParenRef) {
  constexpr char kAstJsonString[] = R"(
    {
      "type": "Paren",
      "expression": {"type": "Identifier", "name": "a"}
    }
  )";

  constexpr char kExpectedIr[] = R"(
module {
  %0 = "sir.identifier_ref"() <{name = "a"}> : () -> !sir.any
  %1 = "sir.paren_ref"(%0) : (!sir.any) -> !sir.any
}
  )";

  TestIrConversion<SParen, SirParenRefOp, SirDialect, AstToSir>({
      .ast_json_string = kAstJsonString,
      .ast_to_ir_visit = AstToSir::VisitParenRef,
      .ir_to_ast_visit = &SirToAst::VisitParenRef,
      .expected_ir_dump = kExpectedIr,
  });
}

// `op_kind` on a class field: the lvalue op falls back to holding an
// `Expression` if the AST holds a node that isn't a `Pattern`.
TEST(ConversionTest, ParenRefWithExpression) {
  constexpr char kAstJsonString[] = R"(
    {
      "type": "Paren",
      "expression": {
        "type": "Call",
        "callee": {"type": "Identifier", "name": "a"}
      }
    }
  )";

  constexpr char kExpectedIr[] = R"(
module {
  %0 = "sir.identifier"() <{name = "a"}> : () -> !sir.any
  %1 = "sir.call"(%0) : (!sir.any) -> !sir.any
  %2 = "sir.paren_ref"(%1) : (!sir.any) -> !sir.any
}
  )";

  TestIrConversion<SParen, SirParenRefOp, SirDialect, AstToSir>({
      .ast_json_string = kAstJsonString,
      .ast_to_ir_visit = AstToSir::VisitParenRef,
      .ir_to_ast_visit = &SirToAst::VisitParenRef,
      .expected_ir_dump = kExpectedIr,
  });
}

// Alternatives of different kinds: the Identifier alternative is held by an
// lvalue op.
TEST(ConversionTest, SpecifierWithIdentifier) {
  constexpr char kAstJsonString[] = R"(
    {
      "type": "Specifier",
      "local": {"type": "Identifier", "name": "a"}
    }
  )";

  constexpr char kExpectedIr[] = R"(
module {
  %0 = "sir.identifier_ref"() <{name = "a"}> : () -> !sir.any
  "sir.specifier"(%0) : (!sir.any) -> ()
}
  )";

  TestIrConversion<SSpecifier, SirSpecifierOp, SirDialect, AstToSir>({
      .ast_json_string = kAstJsonString,
      .ast_to_ir_visit = AstToSir::VisitSpecifier,
      .ir_to_ast_visit = &SirToAst::VisitSpecifier,
      .expected_ir_dump = kExpectedIr,
  });
}

// Alternatives of different kinds: the Call alternative is held by an rvalue
// op.
TEST(ConversionTest, SpecifierWithCall) {
  constexpr char kAstJsonString[] = R"(
    {
      "type": "Specifier",
      "local": {
        "type": "Call",
        "callee": {"type": "Identifier", "name": "a"}
      }
    }
  )";

  constexpr char kExpectedIr[] = R"(
module {
  %0 = "sir.identifier"() <{name = "a"}> : () -> !sir.any
  %1 = "sir.call"(%0) : (!sir.any) -> !sir.any
  "sir.specifier"(%1) : (!sir.any) -> ()
}
  )";

  TestIrConversion<SSpecifier, SirSpecifierOp, SirDialect, AstToSir>({
      .ast_json_string = kAstJsonString,
      .ast_to_ir_visit = AstToSir::VisitSpecifier,
      .ir_to_ast_visit = &SirToAst::VisitSpecifier,
      .expected_ir_dump = kExpectedIr,
  });
}

// An unset optional `local`. `extra` is only in the AST, so the IR -> AST
// conversion leaves it unset.
TEST(ConversionTest, SpecifierWithoutLocal) {
  constexpr char kAstJsonString[] = R"(
    {
      "type": "Specifier"
    }
  )";

  constexpr char kExpectedIr[] = R"(
module {
  "sir.specifier"() : () -> ()
}
  )";

  TestIrConversion<SSpecifier, SirSpecifierOp, SirDialect, AstToSir>({
      .ast_json_string = kAstJsonString,
      .ast_to_ir_visit = AstToSir::VisitSpecifier,
      .ir_to_ast_visit = &SirToAst::VisitSpecifier,
      .expected_ir_dump = kExpectedIr,
  });
}

// The generated verifier requires exactly one of the literal attribute and the
// operand to be set.
TEST(VerifierTest, MemberRequiresExactlyOneProperty) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<SirDialect>();
  // Silence the expected verification errors.
  mlir::ScopedDiagnosticHandler handler(
      &context, [](mlir::Diagnostic&) { return mlir::success(); });

  EXPECT_TRUE(mlir::parseSourceString<mlir::ModuleOp>(R"(
    %0 = "sir.identifier"() <{name = "a"}> : () -> !sir.any
    %1 = "sir.member"(%0) <{literal_property = #sir<identifier "b">}> : (!sir.any) -> !sir.any
  )",
                                                      &context));

  EXPECT_FALSE(mlir::parseSourceString<mlir::ModuleOp>(R"(
    %0 = "sir.identifier"() <{name = "a"}> : () -> !sir.any
    %1 = "sir.member"(%0, %0) <{literal_property = #sir<identifier "b">}> : (!sir.any, !sir.any) -> !sir.any
  )",
                                                       &context));

  EXPECT_FALSE(mlir::parseSourceString<mlir::ModuleOp>(R"(
    %0 = "sir.identifier"() <{name = "a"}> : () -> !sir.any
    %1 = "sir.member"(%0) : (!sir.any) -> !sir.any
  )",
                                                       &context));
}

// The generated verifier requires at most one of the regions of an optional
// split field, and exactly one of the operands/regions of a required one.
TEST(VerifierTest, LoopSplitFields) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<SirDialect>();
  // Silence the expected verification errors.
  mlir::ScopedDiagnosticHandler handler(
      &context, [](mlir::Diagnostic&) { return mlir::success(); });

  // Both `left_declaration` and `left_pattern` are set.
  EXPECT_FALSE(mlir::parseSourceString<mlir::ModuleOp>(R"(
    %0 = "sir.identifier_ref"() <{name = "a"}> : () -> !sir.any
    "sir.loop"(%0) ({
    }, {
    }, {
      "sir.declaration"() <{id = #sir<identifier "a">}> : () -> ()
    }, {
      %1 = "sir.identifier"() <{name = "b"}> : () -> !sir.any
      "sir.expression_statement"(%1) : (!sir.any) -> ()
    }) : (!sir.any) -> ()
  )",
                                                       &context));

  // Neither `left_declaration` nor `left_pattern` is set.
  EXPECT_FALSE(mlir::parseSourceString<mlir::ModuleOp>(R"(
    "sir.loop"() ({
    }, {
    }, {
    }, {
      %1 = "sir.identifier"() <{name = "b"}> : () -> !sir.any
      "sir.expression_statement"(%1) : (!sir.any) -> ()
    }) : () -> ()
  )",
                                                       &context));

  // Both `init_declaration` and `init_expression` are set.
  EXPECT_FALSE(mlir::parseSourceString<mlir::ModuleOp>(R"(
    %0 = "sir.identifier_ref"() <{name = "a"}> : () -> !sir.any
    "sir.loop"(%0) ({
      "sir.declaration"() <{id = #sir<identifier "i">}> : () -> ()
    }, {
      %1 = "sir.identifier"() <{name = "i"}> : () -> !sir.any
      "sir.expr_region_end"(%1) : (!sir.any) -> ()
    }, {
    }, {
      %1 = "sir.identifier"() <{name = "b"}> : () -> !sir.any
      "sir.expression_statement"(%1) : (!sir.any) -> ()
    }) : (!sir.any) -> ()
  )",
                                                       &context));
}

}  // namespace
}  // namespace maldoca
