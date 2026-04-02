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
#include "maldoca/astgen/test/ast_gen_test_util.h"

namespace maldoca {
namespace {

INSTANTIATE_TEST_SUITE_P(
    Lambda, AstGenTest,
    ::testing::Values(AstGenTestParam{
        .ast_def_path =
            "maldoca/astgen/test/enum/ast_def.textproto",
        .ts_interface_path = "maldoca/astgen/test/"
                                     "enum/ast_ts_interface.generated",
        .cc_namespace = "maldoca",
        .ast_path = "maldoca/astgen/test/enum",
        .ir_path = "maldoca/astgen/test/enum",
        .expected_ast_header_path =
            "maldoca/astgen/test/enum/ast.generated.h",
        .expected_ast_source_path =
            "maldoca/astgen/test/enum/ast.generated.cc",
        .expected_ast_to_json_path =
            "maldoca/astgen/test/"
            "enum/ast_to_json.generated.cc",
        .expected_ast_from_json_path =
            "maldoca/astgen/test/"
            "enum/ast_from_json.generated.cc",
        .expected_ir_tablegen_path =
            "maldoca/astgen/test/enum/eir_ops.generated.td",
        .expected_ast_to_ir_source_path =
            "maldoca/astgen/test/enum/conversion/"
            "ast_to_eir.generated.cc",
        .expected_ir_to_ast_source_path =
            "maldoca/astgen/test/enum/conversion/"
            "eir_to_ast.generated.cc",
    }));

}  // namespace
}  // namespace maldoca
