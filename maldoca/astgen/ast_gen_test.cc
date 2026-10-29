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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "maldoca/astgen/ast_def.h"
#include "maldoca/astgen/ast_def.pb.h"
#include "maldoca/astgen/ast_from_json_printer.h"
#include "maldoca/astgen/ast_header_printer.h"
#include "maldoca/astgen/symbol.h"
#include "maldoca/astgen/ts_interface_printer.h"
#include "maldoca/astgen/type.h"
#include "maldoca/base/filesystem.h"
#include "maldoca/base/testing/protocol-buffer-matchers.h"
#include "maldoca/base/testing/status_matchers.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"

namespace maldoca {
namespace {

using ::maldoca::testing::EqualsProto;

struct PrintFieldDefTestCase {
  const char* field_def;
  const char* ts_interface_field;
  const char* cc_member_variable;
};

void TestPrintFieldDef(const PrintFieldDefTestCase& test_case) {
  FieldDefPb field_def_pb;
  ABSL_ASSERT_OK(ParseTextProto(test_case.field_def, "test_case.field_def",
                                &field_def_pb));

  MALDOCA_ASSERT_OK_AND_ASSIGN(
      auto field_def, FieldDef::FromFieldDefPb(field_def_pb, "UsedLanguage"));

  std::string ts_interface_field;
  {
    google::protobuf::io::StringOutputStream os(&ts_interface_field);
    TsInterfacePrinter printer(&os);
    printer.PrintFieldDef(field_def);
  }
  EXPECT_EQ(ts_interface_field, test_case.ts_interface_field);

  std::string cc_member_variable;
  {
    google::protobuf::io::StringOutputStream os(&cc_member_variable);
    AstHeaderPrinter printer(&os);
    printer.PrintMemberVariable(field_def, "UsedLanguage");
  }
  EXPECT_EQ(cc_member_variable, test_case.cc_member_variable);
}

TEST(PrintFieldDef, BuiltinType) {
  TestPrintFieldDef(PrintFieldDefTestCase{
      .field_def = R"pb(
        name: "field"
        type { bool {} }
      )pb",
      .ts_interface_field = "field: boolean\n",
      .cc_member_variable = "bool field_;\n",
  });
}

TEST(PrintFieldDef, TypeMaybeNull) {
  TestPrintFieldDef(PrintFieldDefTestCase{
      .field_def = R"pb(
        name: "field"
        type { bool {} }
        optionalness: OPTIONALNESS_MAYBE_NULL
      )pb",
      .ts_interface_field = "field: boolean | null\n",
      .cc_member_variable = "std::optional<bool> field_;\n",
  });
}

TEST(PrintFieldDef, TypeMaybeUndefined) {
  TestPrintFieldDef(PrintFieldDefTestCase{
      .field_def = R"pb(
        name: "field"
        type { bool {} }
        optionalness: OPTIONALNESS_MAYBE_UNDEFINED
      )pb",
      .ts_interface_field = "field?: boolean\n",
      .cc_member_variable = "std::optional<bool> field_;\n",
  });
}

TEST(PrintFieldDef, PrintMultiLineTitle) {
  static const char kExpectedOutput[] = R"(
// =============================================================================
// Title Line 1
// Title Line 2
// Title Line 3
// =============================================================================
  )";

  std::string output;
  {
    google::protobuf::io::StringOutputStream os(&output);
    CcPrinterBase printer(&os);
    printer.PrintTitle("Title Line 1\nTitle Line 2\nTitle Line 3");
  }
  EXPECT_EQ(absl::StripAsciiWhitespace(output),
            absl::StripAsciiWhitespace(kExpectedOutput));
}

TEST(PrintFieldDef, PrintMultiLineTitleWithEmptyLine) {
  static const char kExpectedOutput[] = R"(
// =============================================================================
// Title Line 1
//
// Title Line 3
// =============================================================================
  )";

  std::string output;
  {
    google::protobuf::io::StringOutputStream os(&output);
    CcPrinterBase printer(&os);
    printer.PrintTitle("Title Line 1\n\nTitle Line 3");
  }
  EXPECT_EQ(absl::StripAsciiWhitespace(output),
            absl::StripAsciiWhitespace(kExpectedOutput));
}

// The nodes shared by the ir_mapping validation tests below. `%s` is replaced
// with the fields (and other properties) of the "Holder" node.
constexpr char kIrMappingAstDef[] = R"pb(
  lang_name: "t"
  nodes { name: "Expression" kinds: FIELD_KIND_RVAL }
  nodes { name: "Pattern" kinds: FIELD_KIND_LVAL }
  nodes { name: "Statement" kinds: FIELD_KIND_STMT }
  nodes {
    name: "Identifier"
    type: "Identifier"
    parents: "Expression"
    parents: "Pattern"
    should_generate_ir_attr: true
  }
  nodes { name: "Declaration" type: "Declaration" parents: "Statement" }
  nodes { name: "Holder" type: "Holder" kinds: FIELD_KIND_STMT %s }
)pb";

// Loads kIrMappingAstDef. The "Holder" node has `holder_fields`, and
// `should_generate_ir_op` iff `has_ir_op`.
absl::StatusOr<AstDef> LoadIrMappingAstDef(absl::string_view holder_fields,
                                           bool has_ir_op = true) {
  std::string text = kIrMappingAstDef;
  text.replace(text.find("%s"), 2,
               absl::StrCat(has_ir_op ? "should_generate_ir_op: true " : "",
                            holder_fields));
  AstDefPb pb;
  ABSL_RETURN_IF_ERROR(ParseTextProto(text, "ast_def", &pb));
  return AstDef::FromProto(pb);
}

// The only field of the "Holder" node.
const FieldDef& HolderField(const AstDef& ast) {
  return ast.nodes().at("Holder")->fields().front();
}

// `ForStatement.init`-like: a declaration (a statement) or an expression, in
// separate IR operands/regions.
constexpr char kSplitField[] = R"pb(
  fields {
    name: "init"
    alternatives {
      type { class: "Declaration" }
      ir_name: "init_decl"
    }
    alternatives {
      type { class: "Expression" }
      ir_name: "init_expr"
    } %s
  }
)pb";

std::string SplitField(absl::string_view extra) {
  std::string field = kSplitField;
  field.replace(field.find("%s"), 2, extra);
  return field;
}

TEST(IrMappingValidation, ValidIrNames) {
  // A statement alternative is held in a region; another one in an operand.
  MALDOCA_ASSERT_OK_AND_ASSIGN(AstDef ast, LoadIrMappingAstDef(SplitField("")));
  EXPECT_THAT(HolderField(ast).ir_splits(),
              ::testing::ElementsAre(EqualsProto(R"pb(
                                       types: "Declaration"
                                       region: "init_decl"
                                       kind: FIELD_KIND_STMT
                                     )pb"),
                                     EqualsProto(R"pb(
                                       types: "Expression"
                                       operand: "init_expr"
                                       kind: FIELD_KIND_RVAL
                                     )pb")));
  // The field's type is the variant of the alternatives' types.
  EXPECT_TRUE(HolderField(ast).type().IsA<VariantType>());

  // With `enclose_in_region`, every alternative is held in a region.
  MALDOCA_ASSERT_OK_AND_ASSIGN(
      AstDef enclosed_ast,
      LoadIrMappingAstDef(SplitField("enclose_in_region: true")));
  EXPECT_THAT(HolderField(enclosed_ast).ir_splits(),
              ::testing::ElementsAre(EqualsProto(R"pb(
                                       types: "Declaration"
                                       region: "init_decl"
                                       kind: FIELD_KIND_STMT
                                     )pb"),
                                     EqualsProto(R"pb(
                                       types: "Expression"
                                       region: "init_expr"
                                       kind: FIELD_KIND_RVAL
                                     )pb")));
}

TEST(IrMappingValidation, AlternativesSharingAnIrNameAreGrouped) {
  MALDOCA_ASSERT_OK_AND_ASSIGN(AstDef ast, LoadIrMappingAstDef(R"pb(
                                 fields {
                                   name: "init"
                                   alternatives {
                                     type { class: "Declaration" }
                                     ir_name: "stmt"
                                   }
                                   alternatives {
                                     type { class: "Statement" }
                                     ir_name: "stmt"
                                   }
                                   alternatives {
                                     type { class: "Expression" }
                                     ir_name: "expr"
                                   }
                                   enclose_in_region: true
                                 })pb"));
  EXPECT_THAT(HolderField(ast).ir_splits(),
              ::testing::ElementsAre(EqualsProto(R"pb(
                                       types: "Declaration"
                                       types: "Statement"
                                       region: "stmt"
                                       kind: FIELD_KIND_STMT
                                     )pb"),
                                     EqualsProto(R"pb(
                                       types: "Expression"
                                       region: "expr"
                                       kind: FIELD_KIND_RVAL
                                     )pb")));
}

TEST(IrMappingValidation, ExplicitKindPicksOneOfTheSupportedKinds) {
  // `Identifier` is both an expression and a pattern.
  MALDOCA_ASSERT_OK_AND_ASSIGN(AstDef ast, LoadIrMappingAstDef(R"pb(
                                 fields {
                                   name: "left"
                                   alternatives {
                                     type { class: "Declaration" }
                                     ir_name: "decl"
                                   }
                                   alternatives {
                                     type { class: "Identifier" }
                                     ir_name: "id"
                                     kind: FIELD_KIND_LVAL
                                   }
                                 })pb"));
  EXPECT_THAT(HolderField(ast).ir_splits(),
              ::testing::ElementsAre(EqualsProto(R"pb(
                                       types: "Declaration"
                                       region: "decl"
                                       kind: FIELD_KIND_STMT
                                     )pb"),
                                     EqualsProto(R"pb(
                                       types: "Identifier"
                                       operand: "id"
                                       kind: FIELD_KIND_LVAL
                                     )pb")));
}

TEST(IrMappingValidation, TypeAndAlternativesAreExclusive) {
  EXPECT_THAT(LoadIrMappingAstDef(SplitField("type { class: \"Expression\" }")),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("Exactly one")));
}

TEST(IrMappingValidation, AlternativesRequireAtLeastTwo) {
  EXPECT_THAT(LoadIrMappingAstDef(R"pb(
                fields {
                  name: "init"
                  alternatives {
                    type { class: "Expression" }
                    ir_name: "a"
                  }
                })pb"),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("at least two")));
}

TEST(IrMappingValidation, FieldKindNotAllowedWithAlternatives) {
  EXPECT_THAT(
      LoadIrMappingAstDef(SplitField("kind: FIELD_KIND_RVAL")),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("kind can't be set")));
}

TEST(IrMappingValidation, IrNamesMustBeOnAllAlternatives) {
  EXPECT_THAT(LoadIrMappingAstDef(R"pb(
                fields {
                  name: "init"
                  alternatives {
                    type { class: "Declaration" }
                    ir_name: "a"
                  }
                  alternatives { type { class: "Expression" } }
                })pb"),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("all alternatives")));
}

TEST(IrMappingValidation, IrNamesRequireAnIrOp) {
  EXPECT_THAT(
      LoadIrMappingAstDef(SplitField(""), /*has_ir_op=*/false),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("should_generate_ir_op")));
}

TEST(IrMappingValidation, AlternativesNotAllowedOnAstOnlyField) {
  EXPECT_THAT(LoadIrMappingAstDef(
                  SplitField("generation_target: GENERATION_TARGET_AST_ONLY")),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("AST_ONLY")));
}

TEST(IrMappingValidation, AlternativesSharingAnIrNameMustHaveTheSameKind) {
  EXPECT_THAT(LoadIrMappingAstDef(R"pb(
                fields {
                  name: "init"
                  alternatives {
                    type { class: "Declaration" }
                    ir_name: "a"
                  }
                  alternatives {
                    type { class: "Expression" }
                    ir_name: "a"
                  }
                })pb"),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("same kind")));
}

TEST(IrMappingValidation, ExplicitKindMustBeSupported) {
  EXPECT_THAT(
      LoadIrMappingAstDef(R"pb(
        fields {
          name: "init"
          alternatives {
            type { class: "Declaration" }
            ir_name: "a"
            kind: FIELD_KIND_RVAL
          }
          alternatives {
            type { class: "Expression" }
            ir_name: "b"
          }
        })pb"),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("doesn't support kind")));
}

TEST(IrMappingValidation, AmbiguousKindMustBeSet) {
  // `Identifier` is both an expression and a pattern.
  EXPECT_THAT(LoadIrMappingAstDef(R"pb(
                fields {
                  name: "left"
                  alternatives {
                    type { class: "Declaration" }
                    ir_name: "a"
                  }
                  alternatives {
                    type { class: "Identifier" }
                    ir_name: "b"
                  }
                })pb"),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("set its kind")));
}

// `ExportSpecifier.local`-like: an lvalue identifier or an rvalue expression,
// in the same IR operand.
constexpr char kMixedKindField[] = R"pb(
  fields {
    name: "local"
    alternatives {
      type { class: "Identifier" }
      kind: %s
    }
    alternatives { type { class: "Expression" } } %s
  }
)pb";

std::string MixedKindField(absl::string_view identifier_kind,
                           absl::string_view extra) {
  std::string field = kMixedKindField;
  field.replace(field.find("%s"), 2, identifier_kind);
  field.replace(field.find("%s"), 2, extra);
  return field;
}

TEST(IrMappingValidation, ValidAlternativesOfDifferentKinds) {
  MALDOCA_ASSERT_OK_AND_ASSIGN(
      AstDef ast, LoadIrMappingAstDef(MixedKindField("FIELD_KIND_LVAL", "")));
  const FieldDef& field = HolderField(ast);
  EXPECT_TRUE(field.ir_splits().empty());
  EXPECT_EQ(field.kind(), FIELD_KIND_RVAL);
  EXPECT_TRUE(field.has_member_ir_kinds());
  EXPECT_EQ(field.member_ir_kind("Identifier"), FIELD_KIND_LVAL);
  EXPECT_EQ(field.member_ir_kind("Expression"), std::nullopt);
}

TEST(IrMappingValidation, AlternativesOfDifferentKindsDoNotRequireAnIrOp) {
  ABSL_EXPECT_OK(LoadIrMappingAstDef(MixedKindField("FIELD_KIND_LVAL", ""),
                                     /*has_ir_op=*/false));
}

TEST(IrMappingValidation, AlternativesWithoutIrNamesMustHaveDifferentKinds) {
  EXPECT_THAT(LoadIrMappingAstDef(MixedKindField("FIELD_KIND_RVAL", "")),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("use type")));
}

TEST(IrMappingValidation, StatementAlternativeRequiresIrNames) {
  EXPECT_THAT(
      LoadIrMappingAstDef(R"pb(
        fields {
          name: "local"
          alternatives { type { class: "Declaration" } }
          alternatives { type { class: "Expression" } }
        })pb"),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("requires ir_names")));
}

TEST(IrMappingValidation,
     AlternativesOfDifferentKindsCannotBeCombinedWithAKindOverride) {
  EXPECT_THAT(
      LoadIrMappingAstDef(MixedKindField(
          "FIELD_KIND_LVAL", "ir_mapping { kind: FIELD_KIND_RVAL }")),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("cannot be combined")));
}

constexpr char kKeyField[] = R"pb(
  fields {
    name: "key"
    type { class: "Expression" }
    kind: FIELD_KIND_RVAL
    ir_mapping {
      operand: "computed_key"
      literal_attr: "literal_key"
      literal_attr_types: "Identifier"
      literal_unless: "%s"
    }
  }
)pb";

TEST(IrMappingValidation, ValidLiteralUnless) {
  std::string key_field = kKeyField;
  key_field.replace(key_field.find("%s"), 2, "computed");
  ABSL_EXPECT_OK(
      LoadIrMappingAstDef(absl::StrCat(key_field, R"pb(fields {
                                                         name: "computed"
                                                         type { bool {} }
                                                       })pb")));
}

TEST(IrMappingValidation, LiteralUnlessMustNameABoolSibling) {
  std::string key_field = kKeyField;
  key_field.replace(key_field.find("%s"), 2, "computed");
  EXPECT_THAT(
      LoadIrMappingAstDef(absl::StrCat(key_field, R"pb(fields {
                                                         name: "computed"
                                                         type { string {} }
                                                       })pb")),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("required bool")));

  key_field = kKeyField;
  key_field.replace(key_field.find("%s"), 2, "missing");
  EXPECT_THAT(LoadIrMappingAstDef(key_field),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("sibling field")));
}

TEST(IrMappingValidation, LiteralAttrRequiresOperand) {
  EXPECT_THAT(LoadIrMappingAstDef(R"pb(
                fields {
                  name: "key"
                  type { class: "Expression" }
                  kind: FIELD_KIND_RVAL
                  ir_mapping {
                    literal_attr: "literal_key"
                    literal_attr_types: "Identifier"
                  }
                })pb"),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("requires operand")));
}

}  // namespace
}  // namespace maldoca
