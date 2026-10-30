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
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "maldoca/astgen/ast_def.h"
#include "maldoca/astgen/ast_def.pb.h"
#include "maldoca/astgen/ast_from_json_printer.h"
#include "maldoca/astgen/ast_header_printer.h"
#include "maldoca/astgen/symbol.h"
#include "maldoca/astgen/ts_interface_printer.h"
#include "maldoca/astgen/type.h"
#include "maldoca/base/filesystem.h"
#include "maldoca/base/testing/status_matchers.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"

namespace maldoca {
namespace {

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

// The nodes shared by the alternatives validation tests below. `%s` is
// replaced with the fields (and other properties) of the "Holder" node.
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
  nodes {
    name: "Literal"
    type: "Literal"
    parents: "Expression"
    should_generate_ir_attr: true
  }
  nodes { name: "ArrayPattern" type: "ArrayPattern" parents: "Pattern" }
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

// Describes each of the ir_splits() of the "Holder" node's only field as
// "<region|operand> <name>: <kind> <types>".
std::vector<std::string> HolderSplits(const AstDef& ast) {
  std::vector<std::string> splits;
  for (const FieldDef::IrSplit& split : HolderField(ast).ir_splits()) {
    splits.push_back(absl::StrCat(split.is_region ? "region " : "operand ",
                                  split.name, ": ", FieldKind_Name(split.kind),
                                  " ", absl::StrJoin(split.types, " | ")));
  }
  return splits;
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
  EXPECT_THAT(
      HolderSplits(ast),
      ::testing::ElementsAre("region init_decl: FIELD_KIND_STMT Declaration",
                             "operand init_expr: FIELD_KIND_RVAL Expression"));
  // The field's type is the variant of the alternatives' types.
  EXPECT_TRUE(HolderField(ast).type().IsA<VariantType>());

  // With `enclose_in_region`, every alternative is held in a region.
  MALDOCA_ASSERT_OK_AND_ASSIGN(
      AstDef enclosed_ast,
      LoadIrMappingAstDef(SplitField("enclose_in_region: true")));
  EXPECT_THAT(
      HolderSplits(enclosed_ast),
      ::testing::ElementsAre("region init_decl: FIELD_KIND_STMT Declaration",
                             "region init_expr: FIELD_KIND_RVAL Expression"));
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
  EXPECT_THAT(HolderSplits(ast),
              ::testing::ElementsAre(
                  "region stmt: FIELD_KIND_STMT Declaration | Statement",
                  "region expr: FIELD_KIND_RVAL Expression"));
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
  EXPECT_THAT(HolderSplits(ast),
              ::testing::ElementsAre("region decl: FIELD_KIND_STMT Declaration",
                                     "operand id: FIELD_KIND_LVAL Identifier"));
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

TEST(IrMappingValidation, VariantAlternativeEqualsSeparateAlternatives) {
  MALDOCA_ASSERT_OK_AND_ASSIGN(AstDef ast, LoadIrMappingAstDef(R"pb(
                                 fields {
                                   name: "init"
                                   alternatives {
                                     type {
                                       variant {
                                         types { class: "Declaration" }
                                         types { class: "Statement" }
                                       }
                                     }
                                     ir_name: "stmt"
                                   }
                                   alternatives {
                                     type { class: "Expression" }
                                     ir_name: "expr"
                                   }
                                   enclose_in_region: true
                                 })pb"));
  EXPECT_THAT(HolderSplits(ast),
              ::testing::ElementsAre(
                  "region stmt: FIELD_KIND_STMT Declaration | Statement",
                  "region expr: FIELD_KIND_RVAL Expression"));
}

TEST(IrMappingValidation, AlternativeTypeMustBeAClassOrVariantOfClasses) {
  EXPECT_THAT(
      LoadIrMappingAstDef(R"pb(
        fields {
          name: "init"
          alternatives {
            type { string {} }
            ir_name: "a"
          }
          alternatives {
            type { class: "Expression" }
            ir_name: "b"
          }
        })pb"),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("variant of classes")));
}

// --- Field-level ir_name ---

TEST(IrMappingValidation, IrNameRenamesTheOperand) {
  MALDOCA_ASSERT_OK_AND_ASSIGN(AstDef ast, LoadIrMappingAstDef(R"pb(
                                 fields {
                                   name: "value"
                                   type { class: "Expression" }
                                   kind: FIELD_KIND_RVAL
                                   ir_name: "operand"
                                 })pb"));
  EXPECT_EQ(HolderField(ast).ir_name().ToCcVarName(), "operand");
  EXPECT_EQ(HolderField(ast).name().ToCcVarName(), "value");
}

TEST(IrMappingValidation, IrNameRenamesTheRegion) {
  MALDOCA_ASSERT_OK_AND_ASSIGN(AstDef ast, LoadIrMappingAstDef(R"pb(
                                 fields {
                                   name: "value"
                                   type { class: "Expression" }
                                   kind: FIELD_KIND_RVAL
                                   enclose_in_region: true
                                   ir_name: "region"
                                 })pb"));
  EXPECT_EQ(HolderField(ast).ir_name().ToCcVarName(), "region");
}

TEST(IrMappingValidation, IrNameDefaultsToTheFieldName) {
  MALDOCA_ASSERT_OK_AND_ASSIGN(AstDef ast, LoadIrMappingAstDef(R"pb(
                                 fields {
                                   name: "value"
                                   type { class: "Expression" }
                                   kind: FIELD_KIND_RVAL
                                 })pb"));
  EXPECT_EQ(HolderField(ast).ir_name().ToCcVarName(), "value");
  EXPECT_EQ(HolderField(ast).literal_attr(), nullptr);
  EXPECT_EQ(HolderField(ast).lval_op_override(), nullptr);
}

TEST(IrMappingValidation, IrNameCannotBeCombinedWithSplitAlternatives) {
  EXPECT_THAT(LoadIrMappingAstDef(SplitField("ir_name: \"x\"")),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("except with")));
}

TEST(IrMappingValidation, IrNameRequiresAnIrField) {
  EXPECT_THAT(LoadIrMappingAstDef(R"pb(
                fields {
                  name: "value"
                  type { class: "Expression" }
                  kind: FIELD_KIND_RVAL
                  generation_target: GENERATION_TARGET_AST_ONLY
                  ir_name: "x"
                })pb"),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("in the IR")));
}

// --- when/unless ---

// `ObjectMember.key`-like: a literal attribute unless `computed`, otherwise an
// operand. `%s` is replaced with the alternatives.
constexpr char kKeyField[] = R"pb(
  fields {
    name: "key"
    type { class: "Expression" } %s
  }
  fields {
    name: "computed"
    type { bool {} }
  }
)pb";

std::string KeyField(absl::string_view alternatives) {
  std::string field = kKeyField;
  field.replace(field.find("%s"), 2, alternatives);
  return field;
}

constexpr char kLiteralKeyAlternative[] = R"pb(
  alternatives {
    type {
      variant {
        types { class: "Identifier" }
        types { class: "Literal" }
      }
    }
    ir_name: "literal_key"
    kind: FIELD_KIND_ATTR
    unless: "computed"
  }
)pb";

constexpr char kComputedKeyAlternative[] = R"pb(
  alternatives { ir_name: "computed_key" when: "computed" }
)pb";

TEST(IrMappingValidation, WhenUnlessResolveToALiteralAttr) {
  MALDOCA_ASSERT_OK_AND_ASSIGN(
      AstDef ast, LoadIrMappingAstDef(KeyField(absl::StrCat(
                      kLiteralKeyAlternative, kComputedKeyAlternative))));
  const FieldDef& key = HolderField(ast);
  EXPECT_TRUE(key.type().IsA<ClassType>());
  EXPECT_EQ(key.kind(), FIELD_KIND_RVAL);
  EXPECT_TRUE(key.ir_splits().empty());
  EXPECT_EQ(key.ir_name().ToCcVarName(), "computed_key");
  EXPECT_EQ(key.lval_op_override(), nullptr);
  ASSERT_NE(key.literal_attr(), nullptr);
  EXPECT_EQ(key.literal_attr()->name, "literal_key");
  EXPECT_THAT(key.literal_attr()->types,
              ::testing::ElementsAre("Identifier", "Literal"));
  EXPECT_EQ(key.literal_attr()->discriminator, "computed");
  EXPECT_FALSE(key.derived_in_ir());
  EXPECT_TRUE(ast.nodes().at("Holder")->fields()[1].derived_in_ir());
}

TEST(IrMappingValidation, UnlessAlternativesSharingAnIrNameAreMerged) {
  MALDOCA_ASSERT_OK_AND_ASSIGN(AstDef ast, LoadIrMappingAstDef(KeyField(R"pb(
                                 alternatives {
                                   type { class: "Identifier" }
                                   ir_name: "literal_key"
                                   kind: FIELD_KIND_ATTR
                                   unless: "computed"
                                 }
                                 alternatives {
                                   type { class: "Literal" }
                                   ir_name: "literal_key"
                                   kind: FIELD_KIND_ATTR
                                   unless: "computed"
                                 }
                                 alternatives {
                                   ir_name: "computed_key"
                                   kind: FIELD_KIND_RVAL
                                   when: "computed"
                                 })pb")));
  ASSERT_NE(HolderField(ast).literal_attr(), nullptr);
  EXPECT_THAT(HolderField(ast).literal_attr()->types,
              ::testing::ElementsAre("Identifier", "Literal"));
}

TEST(IrMappingValidation, WhenAndUnlessAreExclusive) {
  EXPECT_THAT(
      LoadIrMappingAstDef(KeyField(absl::StrCat(kLiteralKeyAlternative, R"pb(
        alternatives {
          ir_name: "computed_key"
          when: "computed"
          unless: "computed"
        })pb"))),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("both when and unless")));
}

TEST(IrMappingValidation, WhenUnlessMustNameTheSameSibling) {
  EXPECT_THAT(
      LoadIrMappingAstDef(KeyField(absl::StrCat(kLiteralKeyAlternative, R"pb(
        alternatives { ir_name: "computed_key" when: "other" })pb"))),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("same sibling")));
}

TEST(IrMappingValidation, WhenUnlessMustNameARequiredBoolSibling) {
  std::string alternatives =
      absl::StrCat(kLiteralKeyAlternative, kComputedKeyAlternative);
  EXPECT_THAT(LoadIrMappingAstDef(absl::StrCat(
                  R"pb(fields { name: "key"
                                type { class: "Expression" })pb",
                  alternatives,
                  R"pb(}
                       fields { name: "computed" type { string {} } })pb")),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("required bool")));
  EXPECT_THAT(LoadIrMappingAstDef(
                  absl::StrCat(R"pb(fields { name: "key"
                                             type { class: "Expression" })pb",
                               alternatives, "}")),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("sibling field")));
}

TEST(IrMappingValidation, ExactlyOneWhenAlternative) {
  EXPECT_THAT(LoadIrMappingAstDef(KeyField(R"pb(
                alternatives {
                  type { class: "Identifier" }
                  ir_name: "literal_key"
                  kind: FIELD_KIND_ATTR
                  unless: "computed"
                }
                alternatives {
                  type { class: "Literal" }
                  ir_name: "literal_key"
                  kind: FIELD_KIND_ATTR
                  unless: "computed"
                })pb")),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("Exactly one")));
}

TEST(IrMappingValidation, WhenAlternativeCannotHaveAType) {
  EXPECT_THAT(
      LoadIrMappingAstDef(KeyField(absl::StrCat(kLiteralKeyAlternative, R"pb(
        alternatives {
          type { class: "Expression" }
          ir_name: "computed_key"
          when: "computed"
        })pb"))),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("whole type")));
}

TEST(IrMappingValidation, UnlessAlternativesMustBeAttributes) {
  EXPECT_THAT(
      LoadIrMappingAstDef(KeyField(absl::StrCat(R"pb(
                                                  alternatives {
                                                    type { class: "Literal" }
                                                    ir_name: "literal_key"
                                                    unless: "computed"
                                                  })pb",
                                                kComputedKeyAlternative))),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("FIELD_KIND_ATTR")));
}

TEST(IrMappingValidation, AttrKindIsOnlyForUnlessAlternatives) {
  EXPECT_THAT(LoadIrMappingAstDef(R"pb(
                fields {
                  name: "init"
                  alternatives {
                    type { class: "Identifier" }
                    ir_name: "a"
                    kind: FIELD_KIND_ATTR
                  }
                  alternatives {
                    type { class: "Declaration" }
                    ir_name: "b"
                  }
                })pb"),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("only supported")));
}

TEST(IrMappingValidation, UnlessAlternativesMustHaveIrAttrs) {
  EXPECT_THAT(
      LoadIrMappingAstDef(KeyField(absl::StrCat(R"pb(
                                                  alternatives {
                                                    type { class: "Pattern" }
                                                    ir_name: "literal_key"
                                                    kind: FIELD_KIND_ATTR
                                                    unless: "computed"
                                                  })pb",
                                                kComputedKeyAlternative))),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("should_generate_ir_attr")));
}

TEST(IrMappingValidation, UnlessAlternativesMustBeTypesOfTheField) {
  EXPECT_THAT(LoadIrMappingAstDef(absl::StrCat(
                  R"pb(
                    fields { name: "key"
                             type { class: "Pattern" }
                  )pb",
                  kLiteralKeyAlternative, kComputedKeyAlternative, R"pb(
                    }
                    fields {
                                           name: "computed"
                                           type { bool {} }
                                         })pb")),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("isn't a type of")));
}

TEST(IrMappingValidation, WhenUnlessRequireARequiredField) {
  EXPECT_THAT(LoadIrMappingAstDef(absl::StrCat(
                  R"pb(
                    fields { name: "key" optionalness: OPTIONALNESS_MAYBE_NULL
                             type { class: "Expression" }
                  )pb",
                  kLiteralKeyAlternative, kComputedKeyAlternative, R"pb(
                    }
                    fields {
                                           name: "computed"
                                           type { bool {} }
                                         })pb")),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("required field")));
}

TEST(IrMappingValidation, AlternativeWithoutTypeRequiresAFieldType) {
  EXPECT_THAT(LoadIrMappingAstDef(absl::StrCat(
                  R"pb(fields { name: "key")pb", kLiteralKeyAlternative,
                  kComputedKeyAlternative,
                  R"pb(}
                       fields { name: "computed" type { bool {} } })pb")),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("must have a type")));
}

TEST(IrMappingValidation, SelectorsMustBeOnAllAlternatives) {
  EXPECT_THAT(
      LoadIrMappingAstDef(KeyField(absl::StrCat(kLiteralKeyAlternative, R"pb(
        alternatives {
          type { class: "Expression" }
          ir_name: "computed_key"
        })pb"))),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("must have a selector")));
}

// --- op_kind ---

// The "Holder" node with rvalue and lvalue ops.
constexpr char kValueNodeKinds[] =
    "kinds: FIELD_KIND_RVAL kinds: FIELD_KIND_LVAL ";

TEST(IrMappingValidation, OpKindOnAVariantField) {
  // `ObjectProperty.value`-like.
  MALDOCA_ASSERT_OK_AND_ASSIGN(
      AstDef ast, LoadIrMappingAstDef(absl::StrCat(kValueNodeKinds, R"pb(
        fields {
          name: "value"
          alternatives {
            type { class: "Expression" }
            op_kind: FIELD_KIND_RVAL
          }
          alternatives {
            type { class: "Pattern" }
            op_kind: FIELD_KIND_LVAL
          }
        })pb")));
  const FieldDef& value = HolderField(ast);
  // The field's type is the variant of the alternatives' types.
  EXPECT_TRUE(value.type().IsA<VariantType>());
  EXPECT_EQ(value.kind(), FIELD_KIND_RVAL);
  ASSERT_NE(value.lval_op_override(), nullptr);
  EXPECT_EQ(value.lval_op_override()->kind, FIELD_KIND_LVAL);
  EXPECT_EQ(value.lval_op_override()->type, std::nullopt);
  EXPECT_EQ(value.literal_attr(), nullptr);
}

TEST(IrMappingValidation, OpKindOnAClassField) {
  // `ParenthesizedExpression.expression`-like, with a renamed operand.
  MALDOCA_ASSERT_OK_AND_ASSIGN(
      AstDef ast, LoadIrMappingAstDef(absl::StrCat(kValueNodeKinds, R"pb(
        fields {
          name: "expression"
          type { class: "Expression" }
          alternatives { op_kind: FIELD_KIND_RVAL }
          alternatives {
            type { class: "Pattern" }
            op_kind: FIELD_KIND_LVAL
          }
          ir_name: "operand"
        })pb")));
  const FieldDef& expression = HolderField(ast);
  EXPECT_TRUE(expression.type().IsA<ClassType>());
  EXPECT_EQ(expression.kind(), FIELD_KIND_RVAL);
  EXPECT_EQ(expression.ir_name().ToCcVarName(), "operand");
  ASSERT_NE(expression.lval_op_override(), nullptr);
  EXPECT_EQ(expression.lval_op_override()->kind, FIELD_KIND_LVAL);
  EXPECT_EQ(expression.lval_op_override()->type, "Pattern");
}

TEST(IrMappingValidation, OpKindWithoutChangesNeedsNoOverride) {
  // Both ops hold the Identifier, as an rvalue and as an lvalue.
  MALDOCA_ASSERT_OK_AND_ASSIGN(
      AstDef ast, LoadIrMappingAstDef(absl::StrCat(kValueNodeKinds, R"pb(
        fields {
          name: "id"
          type { class: "Identifier" }
          alternatives { kind: FIELD_KIND_RVAL op_kind: FIELD_KIND_RVAL }
          alternatives { kind: FIELD_KIND_RVAL op_kind: FIELD_KIND_LVAL }
        })pb")));
  EXPECT_EQ(HolderField(ast).kind(), FIELD_KIND_RVAL);
  EXPECT_EQ(HolderField(ast).lval_op_override(), nullptr);
}

TEST(IrMappingValidation, OpKindRequiresOneAlternativePerKind) {
  EXPECT_THAT(LoadIrMappingAstDef(absl::StrCat(kValueNodeKinds, R"pb(
                fields {
                  name: "value"
                  alternatives {
                    type { class: "Expression" }
                    op_kind: FIELD_KIND_RVAL
                  }
                  alternatives {
                    type { class: "Pattern" }
                    op_kind: FIELD_KIND_RVAL
                  }
                })pb")),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("Exactly one")));
  EXPECT_THAT(LoadIrMappingAstDef(absl::StrCat(kValueNodeKinds, R"pb(
                fields {
                  name: "value"
                  type { class: "Expression" }
                  alternatives { op_kind: FIELD_KIND_RVAL }
                  alternatives {
                    type { class: "Identifier" }
                    kind: FIELD_KIND_RVAL
                    op_kind: FIELD_KIND_RVAL
                  }
                })pb")),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("Exactly one")));
}

TEST(IrMappingValidation, OpKindMustBeSupportedByTheNode) {
  // "Holder" only has a statement op.
  EXPECT_THAT(LoadIrMappingAstDef(R"pb(
                fields {
                  name: "value"
                  alternatives {
                    type { class: "Expression" }
                    op_kind: FIELD_KIND_RVAL
                  }
                  alternatives {
                    type { class: "Pattern" }
                    op_kind: FIELD_KIND_LVAL
                  }
                })pb"),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("that kind")));
}

TEST(IrMappingValidation, OpKindAlternativesCannotHaveIrNames) {
  EXPECT_THAT(LoadIrMappingAstDef(absl::StrCat(kValueNodeKinds, R"pb(
                fields {
                  name: "value"
                  alternatives {
                    type { class: "Expression" }
                    ir_name: "a"
                    op_kind: FIELD_KIND_RVAL
                  }
                  alternatives {
                    type { class: "Pattern" }
                    ir_name: "b"
                    op_kind: FIELD_KIND_LVAL
                  }
                })pb")),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("an ir_name")));
}

TEST(IrMappingValidation, OpKindCannotBeMixedWithWhenUnless) {
  EXPECT_THAT(LoadIrMappingAstDef(
                  KeyField(absl::StrCat(kLiteralKeyAlternative, R"pb(
                    alternatives {
                      ir_name: "computed_key"
                      op_kind: FIELD_KIND_RVAL
                    })pb"))),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("can't mix")));
}

TEST(IrMappingValidation, OpKindAlternativeMustHoldTheMembersOfItsKind) {
  // The rvalue op holds the whole `Expression` member, not just `Identifier`.
  EXPECT_THAT(LoadIrMappingAstDef(absl::StrCat(kValueNodeKinds, R"pb(
                fields {
                  name: "value"
                  type {
                    variant {
                      types { class: "Expression" }
                      types { class: "Pattern" }
                    }
                  }
                  alternatives {
                    type { class: "Identifier" }
                    kind: FIELD_KIND_RVAL
                    op_kind: FIELD_KIND_RVAL
                  }
                  alternatives {
                    type { class: "Pattern" }
                    op_kind: FIELD_KIND_LVAL
                  }
                })pb")),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("must hold")));
}

TEST(IrMappingValidation, OpKindTypeMustOverlapTheFieldType) {
  EXPECT_THAT(
      LoadIrMappingAstDef(absl::StrCat(kValueNodeKinds, R"pb(
        fields {
          name: "value"
          type { class: "Identifier" }
          alternatives { kind: FIELD_KIND_RVAL op_kind: FIELD_KIND_RVAL }
          alternatives {
            type { class: "ArrayPattern" }
            op_kind: FIELD_KIND_LVAL
          }
        })pb")),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("no node in")));
}

TEST(IrMappingValidation, NarrowingOpKindRequiresARequiredField) {
  EXPECT_THAT(LoadIrMappingAstDef(absl::StrCat(kValueNodeKinds, R"pb(
                fields {
                  name: "value"
                  optionalness: OPTIONALNESS_MAYBE_NULL
                  alternatives {
                    type { class: "Expression" }
                    op_kind: FIELD_KIND_RVAL
                  }
                  alternatives {
                    type { class: "Pattern" }
                    op_kind: FIELD_KIND_LVAL
                  }
                })pb")),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("required field")));
}

}  // namespace
}  // namespace maldoca
