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

#include "absl/memory/memory.h"
#include "absl/strings/ascii.h"
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

}  // namespace
}  // namespace maldoca
