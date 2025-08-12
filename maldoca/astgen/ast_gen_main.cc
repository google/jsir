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

/*
  bazel build //maldoca/astgen:ast_gen_main

  ./bazel-bin/maldoca/astgen/ast_gen_main \
     --ast_def_path="maldoca/js/ast/ast_def.textproto" \
     --cc_namespace="maldoca" \
     --ast_path="maldoca/js/ast" \
     --ir_path="maldoca/js/ir"
 */

#include <iostream>
#include <string>
#include "absl/flags/flag.h"
#include "absl/strings/str_cat.h"
#include "maldoca/astgen/ast_def.h"
#include "maldoca/astgen/ast_def.pb.h"
#include "maldoca/astgen/ast_gen.h"
#include "maldoca/base/filesystem.h"
#include "maldoca/base/path.h"
#include "maldoca/base/status_macros.h"

ABSL_FLAG(std::string, ast_def_path, "",
          "The path to the ast_def.textproto file.");

ABSL_FLAG(std::string, cc_namespace, "",
          "The C++ namespace for the AST classes in C++.");

ABSL_FLAG(std::string, ast_path, "", "The directory for the AST code in C++.");

ABSL_FLAG(std::string, ir_path, "",
          "The directory for the IR code in TableGen and C++.");

namespace maldoca {
namespace {

absl::Status AstGenMain() {
  auto ast_def_path = absl::GetFlag(FLAGS_ast_def_path);
  auto cc_namespace = absl::GetFlag(FLAGS_cc_namespace);
  auto ast_path = absl::GetFlag(FLAGS_ast_path);
  auto ir_path = absl::GetFlag(FLAGS_ir_path);

  AstDefPb ast_def_pb;
  MALDOCA_RETURN_IF_ERROR(ParseTextProtoFile(ast_def_path, &ast_def_pb));
  MALDOCA_ASSIGN_OR_RETURN(AstDef ast_def, AstDef::FromProto(ast_def_pb));

  std::string ast_hdr = PrintAstHeader(ast_def, cc_namespace, ast_path);
  auto ast_hdr_path = JoinPath(ast_path, "ast.generated.h");
  std::cout << "Writing ast_hdr to " << ast_hdr_path << "\n";
  MALDOCA_RETURN_IF_ERROR(SetFileContents(ast_hdr_path, ast_hdr));

  std::string ast_src = PrintAstSource(ast_def, cc_namespace, ast_path);
  auto ast_src_path = JoinPath(ast_path, "ast.generated.cc");
  std::cout << "Writing ast_src to " << ast_src_path << "\n";
  MALDOCA_RETURN_IF_ERROR(SetFileContents(ast_src_path, ast_src));

  std::string ast_to_json = PrintAstToJson(ast_def, cc_namespace, ast_path);
  auto ast_to_json_path = JoinPath(ast_path, "ast_to_json.generated.cc");
  std::cout << "Writing ast_to_json to " << ast_to_json_path << "\n";
  MALDOCA_RETURN_IF_ERROR(
      SetFileContents(ast_to_json_path, ast_to_json));

  std::string ast_from_json = PrintAstFromJson(ast_def, cc_namespace, ast_path);
  auto ast_from_json_path =
      JoinPath(ast_path, "ast_from_json.generated.cc");
  std::cout << "Writing ast_from_json to " << ast_from_json_path << "\n";
  MALDOCA_RETURN_IF_ERROR(
      SetFileContents(ast_from_json_path, ast_from_json));

  if (!ir_path.empty()) {
    std::string ir_tablegen = PrintIrTableGen(ast_def, ir_path);
    auto ir_tablegen_path = JoinPath(
        ir_path, absl::StrCat(ast_def.lang_name(), "ir_ops.generated.td"));
    std::cout << "Writing ir_tablegen to " << ir_tablegen_path << "\n";
    MALDOCA_RETURN_IF_ERROR(
        SetFileContents(ir_tablegen_path, ir_tablegen));

    std::string ast_to_ir =
        PrintAstToIrSource(ast_def, cc_namespace, ast_path, ir_path);
    auto ast_to_ir_path = JoinPath(
        ir_path, "conversion",
        absl::StrCat("ast_to_", ast_def.lang_name(), "ir.generated.cc"));
    std::cout << "Writing ast_to_ir to " << ast_to_ir_path << "\n";
    MALDOCA_RETURN_IF_ERROR(
        SetFileContents(ast_to_ir_path, ast_to_ir));

    std::string ir_to_ast =
        PrintIrToAstSource(ast_def, cc_namespace, ast_path, ir_path);
    auto ir_to_ast_path = JoinPath(
        ir_path, "conversion",
        absl::StrCat(ast_def.lang_name(), "ir_to_ast.generated.cc"));
    std::cout << "Writing ir_to_ast to " << ir_to_ast_path << "\n";
    MALDOCA_RETURN_IF_ERROR(
        SetFileContents(ir_to_ast_path, ir_to_ast));
  }

  return absl::OkStatus();
}

}  // namespace
}  // namespace maldoca

int main(int argc, char* argv[]) {

  auto status = maldoca::AstGenMain();
  if (!status.ok()) {
    std::cerr << "Error: " << status.ToString() << std::endl;
    return 1;
  }

  return 0;
}
