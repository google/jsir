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

#ifndef MALDOCA_ASTGEN_TEST_AST_GEN_TEST_UTIL_H_
#define MALDOCA_ASTGEN_TEST_AST_GEN_TEST_UTIL_H_

#include <optional>
#include <string>

#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "maldoca/astgen/ast_def.h"

namespace maldoca {

// Specifies a test case for ASTGen.
struct AstGenTestParam {
  // Path to the "ast.textproto" AST specification file.
  // Should start with "google3/".
  // Example:
  // "maldoca/astgen/test/lambda/ast_def.textproto"
  std::string ast_def_path;

  // Path to the expected generated TypeScript interface file.
  // Should start with "google3/".
  // Example:
  // "maldoca/astgen/test/lambda/ast_ts_interface.generated"
  std::optional<std::string> ts_interface_path;

  // The C++ namespace for the AST classes in C++.
  // Example:
  // "maldoca::astgen"
  std::string cc_namespace;

  // The directory for the AST code in C++.
  // Inside the directory, there are the following files:
  // - "ast.generated.h"
  // - "ast.generated.cc"
  // - "ast_to_json.generated.cc"
  // - "ast_from_json.generated.cc"
  std::string ast_path;

  // The directory for the IR code in TableGen and C++.
  // Inside the directory, there are the following files:
  // - "<lang>ir_ops.generated.td"
  // - "conversion/ast_to_<lang>hir.generated.cc"
  // - "conversion/<lang>hir_to_ast.generated.cc"
  std::string ir_path;

  // Path to the expected "ast.generated.h" C++ header file.
  // Should start with "google3/".
  // Example:
  // "maldoca/astgen/test/lambda/ast.generated.h"
  std::optional<std::string> expected_ast_header_path;

  // Path to the expected "ast.generated.cc" C++ source file.
  // Should start with "google3/".
  // Example:
  // "maldoca/astgen/test/lambda/ast.generated.cc"
  std::optional<std::string> expected_ast_source_path;

  // Path to the expected "ast_to_json.generated.cc" C++ source file.
  // Should start with "google3/".
  // Example:
  // "maldoca/astgen/test/lambda/ast_to_json.generated.cc"
  std::optional<std::string> expected_ast_to_json_path;

  // Path to the expected "ast_from_json.generated.cc" C++ source file.
  // Should start with "google3/".
  // Example:
  // "maldoca/astgen/test/lambda/ast_from_json.generated.cc"
  std::optional<std::string> expected_ast_from_json_path;

  // Path to the expected "<lang_name>ir_ops.generated.td" TableGen source file.
  // Should start with "google3/".
  // Example:
  // "maldoca/astgen/test/lambda/lambdair_ops.generated.td"
  std::optional<std::string> expected_ir_tablegen_path;

  // Path to the expected "ast_to_<lang_name>ir.generated.cc" C++ source file.
  // Should start with "google3/".
  // Example:
  // "maldoca/astgen/test/lambda/ast_to_lambdair.generated.cc"
  std::optional<std::string> expected_ast_to_ir_source_path;

  // Path to the expected "<lang_name>ir_to_ast.generated.cc" C++ source file.
  // Should start with "google3/".
  // Example:
  // "maldoca/astgen/test/lambda/lambdair_to_ast.generated.cc"
  std::optional<std::string> expected_ir_to_ast_source_path;
};

class AstGenTest : public ::testing::TestWithParam<AstGenTestParam> {
 protected:
  absl::StatusOr<AstDef> LoadAstDef() const;
};

}  // namespace maldoca

#endif  // MALDOCA_ASTGEN_TEST_AST_GEN_TEST_UTIL_H_
