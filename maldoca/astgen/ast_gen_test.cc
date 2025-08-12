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

#include "maldoca/astgen/ast_gen.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/ascii.h"
#include "maldoca/astgen/ast_def.h"
#include "maldoca/astgen/ast_def.pb.h"
#include "maldoca/astgen/symbol.h"
#include "maldoca/astgen/type.h"
#include "maldoca/base/filesystem.h"
#include "maldoca/base/testing/status_matchers.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"

namespace maldoca {
namespace {

struct PrintFieldDefTestCase {
  const char *field_def;
  const char *ts_interface_field;
  const char *cc_member_variable;
};

void TestPrintFieldDef(const PrintFieldDefTestCase &test_case) {
  FieldDefPb field_def_pb;
  MALDOCA_ASSERT_OK(ParseTextProto(test_case.field_def, "test_case.field_def",
                                   &field_def_pb));

  MALDOCA_ASSERT_OK_AND_ASSIGN(auto field_def,
                       FieldDef::FromFieldDefPb(field_def_pb, "UsedLanguage"));

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

// =============================================================================
// AstFromJsonPrinter::PrintBuiltinFromJson()
// =============================================================================

TEST(AstFromJsonPrinterTest, TestPrintAssignBuiltinFromJson) {
  static const char kExpectedOutput[] = R"(
my_lhs = my_rhs.get<std::string>();
  )";

  std::string output;
  {
    google::protobuf::io::StringOutputStream os(&output);
    AstFromJsonPrinter printer(&os);
    printer.PrintBuiltinFromJson(
        AstFromJsonPrinter::Action::kAssign,
        AstFromJsonPrinter::CheckJsonType::kNo,
        BuiltinType(BuiltinTypeKind::kString, "UsedLanguage"), Symbol("my_lhs"),
        Symbol("my_rhs"));
  }

  EXPECT_EQ(absl::StripAsciiWhitespace(output),
            absl::StripAsciiWhitespace(kExpectedOutput));
}

TEST(AstFromJsonPrinterTest, TestPrintDefBuiltinFromJson) {
  static const char kExpectedOutput[] = R"(
auto my_lhs = my_rhs.get<std::string>();
  )";

  std::string output;
  {
    google::protobuf::io::StringOutputStream os(&output);
    AstFromJsonPrinter printer(&os);
    printer.PrintBuiltinFromJson(
        AstFromJsonPrinter::Action::kDef,
        AstFromJsonPrinter::CheckJsonType::kNo,
        BuiltinType(BuiltinTypeKind::kString, "UsedLanguage"), Symbol("my_lhs"),
        Symbol("my_rhs"));
  }

  EXPECT_EQ(absl::StripAsciiWhitespace(output),
            absl::StripAsciiWhitespace(kExpectedOutput));
}

TEST(AstFromJsonPrinterTest, TestPrintReturnBuiltinFromJson) {
  static const char kExpectedOutput[] = R"(
return my_rhs.get<std::string>();
  )";

  std::string output;
  {
    google::protobuf::io::StringOutputStream os(&output);
    AstFromJsonPrinter printer(&os);
    printer.PrintBuiltinFromJson(
        AstFromJsonPrinter::Action::kReturn,
        AstFromJsonPrinter::CheckJsonType::kNo,
        BuiltinType(BuiltinTypeKind::kString, "UsedLanguage"), Symbol("my_lhs"),
        Symbol("my_rhs"));
  }

  EXPECT_EQ(absl::StripAsciiWhitespace(output),
            absl::StripAsciiWhitespace(kExpectedOutput));
}

TEST(AstFromJsonPrinterTest, TestPrintAssignClassFromJson) {
  static const char kExpectedOutput[] = R"(
MALDOCA_ASSIGN_OR_RETURN(my_lhs, UsedLanguageClassType::FromJson(my_rhs));
  )";

  std::string output;
  {
    google::protobuf::io::StringOutputStream os(&output);
    AstFromJsonPrinter printer(&os);
    printer.PrintClassFromJson(AstFromJsonPrinter::Action::kAssign,
                               ClassType(Symbol("ClassType"), "UsedLanguage"),
                               Symbol("my_lhs"), Symbol("my_rhs"),
                               "UsedLanguage");
  }

  EXPECT_EQ(absl::StripAsciiWhitespace(output),
            absl::StripAsciiWhitespace(kExpectedOutput));
}

TEST(AstFromJsonPrinterTest, TestPrintDefClassFromJson) {
  static const char kExpectedOutput[] = R"(
MALDOCA_ASSIGN_OR_RETURN(auto my_lhs, UsedLanguageClassType::FromJson(my_rhs));
  )";

  std::string output;
  {
    google::protobuf::io::StringOutputStream os(&output);
    AstFromJsonPrinter printer(&os);
    printer.PrintClassFromJson(AstFromJsonPrinter::Action::kDef,
                               ClassType(Symbol("ClassType"), "UsedLanguage"),
                               Symbol("my_lhs"), Symbol("my_rhs"),
                               "UsedLanguage");
  }

  EXPECT_EQ(absl::StripAsciiWhitespace(output),
            absl::StripAsciiWhitespace(kExpectedOutput));
}

TEST(AstFromJsonPrinterTest, TestPrintReturnClassFromJson) {
  static const char kExpectedOutput[] = R"(
return UsedLanguageClassType::FromJson(my_rhs);
  )";

  std::string output;
  {
    google::protobuf::io::StringOutputStream os(&output);
    AstFromJsonPrinter printer(&os);
    printer.PrintClassFromJson(AstFromJsonPrinter::Action::kReturn,
                               ClassType(Symbol("ClassType"), "UsedLanguage"),
                               Symbol("my_lhs"), Symbol("my_rhs"),
                               "UsedLanguage");
  }

  EXPECT_EQ(absl::StripAsciiWhitespace(output),
            absl::StripAsciiWhitespace(kExpectedOutput));
}

// =============================================================================
// AstFromJsonPrinter::PrintVariantFromJson()
// =============================================================================

TEST(AstFromJsonPrinterTest, TestPrintAssignVariantFromJson) {
  static const char kExpectedOutput[] = R"(
if (my_rhs.is_string()) {
  my_lhs = my_rhs.get<std::string>();
} else if (IsClassType(my_rhs)) {
  MALDOCA_ASSIGN_OR_RETURN(my_lhs, UsedLanguageClassType::FromJson(my_rhs));
} else {
  auto result = absl::InvalidArgumentError("my_rhs has invalid type.");
  result.SetPayload("json", absl::Cord{json.dump()});
  result.SetPayload("json_element", absl::Cord{my_rhs.dump()});
  return result;
}
  )";

  std::vector<std::unique_ptr<ScalarType>> types;
  types.push_back(
      absl::make_unique<BuiltinType>(BuiltinTypeKind::kString, "UsedLanguage"));
  types.push_back(
      absl::make_unique<ClassType>(Symbol("ClassType"), "UsedLanguage"));
  auto variant_type = VariantType(std::move(types), "UsedLanguage");

  std::string output;
  {
    google::protobuf::io::StringOutputStream os(&output);
    AstFromJsonPrinter printer(&os);
    printer.PrintVariantFromJson(AstFromJsonPrinter::Action::kAssign,
                                 variant_type, Symbol("my_lhs"),
                                 Symbol("my_rhs"), "UsedLanguage");
  }

  EXPECT_EQ(absl::StripAsciiWhitespace(output),
            absl::StripAsciiWhitespace(kExpectedOutput));
}

TEST(AstFromJsonPrinterTest, TestPrintDefVariantFromJson) {
  static const char kExpectedOutput[] = R"(
std::variant<std::string, std::unique_ptr<UsedLanguageClassType>> my_lhs;
if (my_rhs.is_string()) {
  my_lhs = my_rhs.get<std::string>();
} else if (IsClassType(my_rhs)) {
  MALDOCA_ASSIGN_OR_RETURN(my_lhs, UsedLanguageClassType::FromJson(my_rhs));
} else {
  auto result = absl::InvalidArgumentError("my_rhs has invalid type.");
  result.SetPayload("json", absl::Cord{json.dump()});
  result.SetPayload("json_element", absl::Cord{my_rhs.dump()});
  return result;
}
  )";

  std::vector<std::unique_ptr<ScalarType>> types;
  types.push_back(
      absl::make_unique<BuiltinType>(BuiltinTypeKind::kString, "UsedLanguage"));
  types.push_back(
      absl::make_unique<ClassType>(Symbol("ClassType"), "UsedLanguage"));
  auto variant_type = VariantType(std::move(types), "UsedLanguage");

  std::string output;
  {
    google::protobuf::io::StringOutputStream os(&output);
    AstFromJsonPrinter printer(&os);
    printer.PrintVariantFromJson(AstFromJsonPrinter::Action::kDef, variant_type,
                                 Symbol("my_lhs"), Symbol("my_rhs"),
                                 "UsedLanguage");
  }

  EXPECT_EQ(absl::StripAsciiWhitespace(output),
            absl::StripAsciiWhitespace(kExpectedOutput));
}

TEST(AstFromJsonPrinterTest, TestPrintReturnVariantFromJson) {
  static const char kExpectedOutput[] = R"(
if (my_rhs.is_string()) {
  return my_rhs.get<std::string>();
} else if (IsClassType(my_rhs)) {
  return UsedLanguageClassType::FromJson(my_rhs);
} else {
  auto result = absl::InvalidArgumentError("my_rhs has invalid type.");
  result.SetPayload("json", absl::Cord{json.dump()});
  result.SetPayload("json_element", absl::Cord{my_rhs.dump()});
  return result;
}
  )";

  std::vector<std::unique_ptr<ScalarType>> types;
  types.push_back(
      absl::make_unique<BuiltinType>(BuiltinTypeKind::kString, "UsedLanguage"));
  types.push_back(
      absl::make_unique<ClassType>(Symbol("ClassType"), "UsedLanguage"));
  auto variant_type = VariantType(std::move(types), "UsedLanguage");

  std::string output;
  {
    google::protobuf::io::StringOutputStream os(&output);
    AstFromJsonPrinter printer(&os);
    printer.PrintVariantFromJson(AstFromJsonPrinter::Action::kReturn,
                                 variant_type, Symbol("my_lhs"),
                                 Symbol("my_rhs"), "UsedLanguage");
  }

  EXPECT_EQ(absl::StripAsciiWhitespace(output),
            absl::StripAsciiWhitespace(kExpectedOutput));
}

// =============================================================================
// AstFromJsonPrinter::PrintListFromJson()
// =============================================================================

TEST(AstFromJsonPrinterTest, TestPrintAssignListFromJson) {
  static const char kExpectedOutput[] = R"(
if (!my_rhs.is_array()) {
  return absl::InvalidArgumentError("my_rhs expected to be array.");
}

std::vector<std::string> my_lhs_value;
for (const nlohmann::json& my_rhs_element : my_rhs) {
  if (my_rhs_element.is_null()) {
    return absl::InvalidArgumentError("my_rhs_element is null.");
  }
  if (!my_rhs_element.is_string()) {
    return absl::InvalidArgumentError("Expecting my_rhs_element.is_string().");
  }
  auto my_lhs_element = my_rhs_element.get<std::string>();
  my_lhs_value.push_back(std::move(my_lhs_element));
}
my_lhs = std::move(my_lhs_value);
  )";

  auto element_type =
      absl::make_unique<BuiltinType>(BuiltinTypeKind::kString, "UsedLanguage");
  auto list_type =
      ListType(std::move(element_type), MaybeNull::kNo, "UsedLanguage");

  std::string output;
  {
    google::protobuf::io::StringOutputStream os(&output);
    AstFromJsonPrinter printer(&os);
    printer.PrintListFromJson(AstFromJsonPrinter::Action::kAssign, list_type,
                              Symbol("my_lhs"), Symbol("my_rhs"),
                              "UsedLanguage");
  }

  EXPECT_EQ(absl::StripAsciiWhitespace(output),
            absl::StripAsciiWhitespace(kExpectedOutput));
}

TEST(AstFromJsonPrinterTest, TestPrintDefListFromJson) {
  static const char kExpectedOutput[] = R"(
if (!my_rhs.is_array()) {
  return absl::InvalidArgumentError("my_rhs expected to be array.");
}

std::vector<std::string> my_lhs;
for (const nlohmann::json& my_rhs_element : my_rhs) {
  if (my_rhs_element.is_null()) {
    return absl::InvalidArgumentError("my_rhs_element is null.");
  }
  if (!my_rhs_element.is_string()) {
    return absl::InvalidArgumentError("Expecting my_rhs_element.is_string().");
  }
  auto my_lhs_element = my_rhs_element.get<std::string>();
  my_lhs.push_back(std::move(my_lhs_element));
}
  )";

  auto element_type =
      absl::make_unique<BuiltinType>(BuiltinTypeKind::kString, "UsedLanguage");
  auto list_type =
      ListType(std::move(element_type), MaybeNull::kNo, "UsedLanguage");

  std::string output;
  {
    google::protobuf::io::StringOutputStream os(&output);
    AstFromJsonPrinter printer(&os);
    printer.PrintListFromJson(AstFromJsonPrinter::Action::kDef, list_type,
                              Symbol("my_lhs"), Symbol("my_rhs"),
                              "UsedLanguage");
  }

  EXPECT_EQ(absl::StripAsciiWhitespace(output),
            absl::StripAsciiWhitespace(kExpectedOutput));
}

TEST(AstFromJsonPrinterTest, TestPrintReturnListFromJson) {
  static const char kExpectedOutput[] = R"(
if (!my_rhs.is_array()) {
  return absl::InvalidArgumentError("my_rhs expected to be array.");
}

std::vector<std::string> my_lhs;
for (const nlohmann::json& my_rhs_element : my_rhs) {
  if (my_rhs_element.is_null()) {
    return absl::InvalidArgumentError("my_rhs_element is null.");
  }
  if (!my_rhs_element.is_string()) {
    return absl::InvalidArgumentError("Expecting my_rhs_element.is_string().");
  }
  auto my_lhs_element = my_rhs_element.get<std::string>();
  my_lhs.push_back(std::move(my_lhs_element));
}
return my_lhs;
  )";

  auto element_type =
      absl::make_unique<BuiltinType>(BuiltinTypeKind::kString, "UsedLanguage");
  auto list_type =
      ListType(std::move(element_type), MaybeNull::kNo, "UsedLanguage");

  std::string output;
  {
    google::protobuf::io::StringOutputStream os(&output);
    AstFromJsonPrinter printer(&os);
    printer.PrintListFromJson(AstFromJsonPrinter::Action::kReturn, list_type,
                              Symbol("my_lhs"), Symbol("my_rhs"),
                              "UsedLanguage");
  }

  EXPECT_EQ(absl::StripAsciiWhitespace(output),
            absl::StripAsciiWhitespace(kExpectedOutput));
}

TEST(AstFromJsonPrinterTest, TestPrintDefListOfOptionalElementsFromJson) {
  static const char kExpectedOutput[] = R"(
if (!my_rhs.is_array()) {
  return absl::InvalidArgumentError("my_rhs expected to be array.");
}

std::vector<std::optional<std::string>> my_lhs;
for (const nlohmann::json& my_rhs_element : my_rhs) {
  std::optional<std::string> my_lhs_element;
  if (!my_rhs_element.is_null()) {
    if (!my_rhs_element.is_string()) {
      return absl::InvalidArgumentError("Expecting my_rhs_element.is_string().");
    }
    my_lhs_element = my_rhs_element.get<std::string>();
  }
  my_lhs.push_back(std::move(my_lhs_element));
}
  )";

  auto element_type =
      absl::make_unique<BuiltinType>(BuiltinTypeKind::kString, "UsedLanguage");
  auto list_type =
      ListType(std::move(element_type), MaybeNull::kYes, "UsedLanguage");

  std::string output;
  {
    google::protobuf::io::StringOutputStream os(&output);
    AstFromJsonPrinter printer(&os);
    printer.PrintListFromJson(AstFromJsonPrinter::Action::kDef, list_type,
                              Symbol("my_lhs"), Symbol("my_rhs"),
                              "UsedLanguage");
  }

  EXPECT_EQ(absl::StripAsciiWhitespace(output),
            absl::StripAsciiWhitespace(kExpectedOutput));
}

}  // namespace
}  // namespace maldoca
