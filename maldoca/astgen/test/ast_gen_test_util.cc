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

#include "maldoca/astgen/test/ast_gen_test_util.h"

#include <iostream>
#include <string>

#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "maldoca/astgen/ast_def.h"
#include "maldoca/astgen/ast_def.pb.h"
#include "maldoca/astgen/ast_gen.h"
#include "maldoca/base/filesystem.h"
#include "maldoca/base/get_runfiles_dir.h"
#include "maldoca/base/status_macros.h"
#include "maldoca/base/testing/status_matchers.h"

namespace maldoca {

absl::StatusOr<AstDef> AstGenTest::LoadAstDef() const {
  auto ast_def_path =
      GetDataDependencyFilepath(GetParam().ast_def_path);
  AstDefPb ast_def_pb;
  MALDOCA_RETURN_IF_ERROR(ParseTextProtoFile(ast_def_path, &ast_def_pb));
  MALDOCA_ASSIGN_OR_RETURN(AstDef ast_def, AstDef::FromProto(ast_def_pb));
  return ast_def;
}

TEST_P(AstGenTest, PrintTsInterfaceTest) {
  MALDOCA_ASSERT_OK_AND_ASSIGN(AstDef ast_def, LoadAstDef());

  std::string ts_interface = PrintTsInterface(ast_def);

  if (GetParam().ts_interface_path.has_value()) {
    auto ts_interface_path =
        GetDataDependencyFilepath(*GetParam().ts_interface_path);
    MALDOCA_ASSERT_OK_AND_ASSIGN(std::string expected_ts_interface,
                         GetFileContents(ts_interface_path));

    LOG(INFO) << "ts_interface_path: " << ts_interface_path;
    EXPECT_EQ(absl::StripAsciiWhitespace(ts_interface),
              absl::StripAsciiWhitespace(expected_ts_interface));
  }
}

TEST_P(AstGenTest, AstHdrTest) {
  MALDOCA_ASSERT_OK_AND_ASSIGN(AstDef ast_def, LoadAstDef());
  std::string ast_hdr =
      PrintAstHeader(ast_def, GetParam().cc_namespace, GetParam().ast_path);

  std::cout << ast_hdr << std::endl;

  if (GetParam().expected_ast_header_path.has_value()) {
    auto expected_ast_h_path =
        GetDataDependencyFilepath(*GetParam().expected_ast_header_path);
    MALDOCA_ASSERT_OK_AND_ASSIGN(std::string expected_ast_hdr,
                         GetFileContents(expected_ast_h_path));

    LOG(INFO) << " expected_ast_h_path: " << expected_ast_h_path;
    EXPECT_EQ(absl::StripAsciiWhitespace(ast_hdr),
              absl::StripAsciiWhitespace(expected_ast_hdr));
  }
}

TEST_P(AstGenTest, AstSrcTest) {
  MALDOCA_ASSERT_OK_AND_ASSIGN(AstDef ast_def, LoadAstDef());
  std::string ast_src =
      PrintAstSource(ast_def, GetParam().cc_namespace, GetParam().ast_path);

  std::cout << ast_src << std::endl;

  if (GetParam().expected_ast_source_path.has_value()) {
    auto expected_ast_src_path =
        GetDataDependencyFilepath(*GetParam().expected_ast_source_path);
    MALDOCA_ASSERT_OK_AND_ASSIGN(std::string expected_ast_src,
                         GetFileContents(expected_ast_src_path));

    LOG(INFO) << " expected_ast_src_path: " << expected_ast_src_path;
    EXPECT_EQ(absl::StripAsciiWhitespace(ast_src),
              absl::StripAsciiWhitespace(expected_ast_src));
  }
}

TEST_P(AstGenTest, AstToJsonTest) {
  MALDOCA_ASSERT_OK_AND_ASSIGN(AstDef ast_def, LoadAstDef());
  std::string ast_to_json =
      PrintAstToJson(ast_def, GetParam().cc_namespace, GetParam().ast_path);

  std::cout << ast_to_json << std::endl;

  if (GetParam().expected_ast_to_json_path.has_value()) {
    auto expected_ast_to_json_path = GetDataDependencyFilepath(
        *GetParam().expected_ast_to_json_path);
    MALDOCA_ASSERT_OK_AND_ASSIGN(std::string expected_ast_to_json,
                         GetFileContents(expected_ast_to_json_path));

    LOG(INFO) << " expected_ast_to_json_path: " << expected_ast_to_json_path;
    EXPECT_EQ(absl::StripAsciiWhitespace(ast_to_json),
              absl::StripAsciiWhitespace(expected_ast_to_json));
  }
}

TEST_P(AstGenTest, AstFromJsonTest) {
  MALDOCA_ASSERT_OK_AND_ASSIGN(AstDef ast_def, LoadAstDef());
  std::string ast_from_json =
      PrintAstFromJson(ast_def, GetParam().cc_namespace, GetParam().ast_path);

  std::cout << ast_from_json << std::endl;

  if (GetParam().expected_ast_from_json_path.has_value()) {
    auto expected_ast_from_json_path = GetDataDependencyFilepath(
        *GetParam().expected_ast_from_json_path);
    MALDOCA_ASSERT_OK_AND_ASSIGN(std::string expected_ast_from_json,
                         GetFileContents(expected_ast_from_json_path));

    LOG(INFO) << " expected_ast_from_json_path: "
              << expected_ast_from_json_path;
    EXPECT_EQ(absl::StripAsciiWhitespace(ast_from_json),
              absl::StripAsciiWhitespace(expected_ast_from_json));
  }
}

TEST_P(AstGenTest, IrTableGenTest) {
  MALDOCA_ASSERT_OK_AND_ASSIGN(AstDef ast_def, LoadAstDef());
  std::string ir_tablegen = PrintIrTableGen(ast_def, GetParam().ir_path);

  // So that we can copy from the output.
  std::cout << "Output:" << std::endl;
  std::cout << ir_tablegen << std::endl;

  if (GetParam().expected_ir_tablegen_path.has_value()) {
    auto expected_ir_tablegen_path = GetDataDependencyFilepath(
        *GetParam().expected_ir_tablegen_path);
    MALDOCA_ASSERT_OK_AND_ASSIGN(std::string expected_ir_tablegen,
                         GetFileContents(expected_ir_tablegen_path));

    LOG(INFO) << " expected_ir_tablegen_path: " << expected_ir_tablegen_path;
    EXPECT_EQ(absl::StripAsciiWhitespace(ir_tablegen),
              absl::StripAsciiWhitespace(expected_ir_tablegen));
  }
}

TEST_P(AstGenTest, AstToIrTest) {
  MALDOCA_ASSERT_OK_AND_ASSIGN(AstDef ast_def, LoadAstDef());
  std::string ast_to_ir_source =
      PrintAstToIrSource(ast_def, GetParam().cc_namespace, GetParam().ast_path,
                         GetParam().ir_path);

  std::cout << "Output:" << std::endl;
  std::cout << ast_to_ir_source << std::endl;

  if (GetParam().expected_ast_to_ir_source_path.has_value()) {
    auto cc_ast_to_ir_source_path = GetDataDependencyFilepath(
        *GetParam().expected_ast_to_ir_source_path);
    MALDOCA_ASSERT_OK_AND_ASSIGN(std::string expected_ast_to_ir_source,
                         GetFileContents(cc_ast_to_ir_source_path));

    LOG(INFO) << " cc_ast_to_ir_source_path: " << cc_ast_to_ir_source_path;
    EXPECT_EQ(absl::StripAsciiWhitespace(ast_to_ir_source),
              absl::StripAsciiWhitespace(expected_ast_to_ir_source));
  }
}

TEST_P(AstGenTest, IrToAstTest) {
  MALDOCA_ASSERT_OK_AND_ASSIGN(AstDef ast_def, LoadAstDef());
  std::string ir_to_ast_source =
      PrintIrToAstSource(ast_def, GetParam().cc_namespace, GetParam().ast_path,
                         GetParam().ir_path);

  std::cout << "Output:" << std::endl;
  std::cout << ir_to_ast_source << std::endl;

  if (GetParam().expected_ir_to_ast_source_path.has_value()) {
    auto cc_ir_to_ast_source_path = GetDataDependencyFilepath(
        *GetParam().expected_ir_to_ast_source_path);
    MALDOCA_ASSERT_OK_AND_ASSIGN(std::string expected_ir_to_ast_source,
                         GetFileContents(cc_ir_to_ast_source_path));

    LOG(INFO) << " cc_ast_to_ir_source_path: " << cc_ir_to_ast_source_path;
    EXPECT_EQ(absl::StripAsciiWhitespace(ir_to_ast_source),
              absl::StripAsciiWhitespace(expected_ir_to_ast_source));
  }
}

}  // namespace maldoca
