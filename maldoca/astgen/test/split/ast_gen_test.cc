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

#include "gtest/gtest.h"
#include "maldoca/astgen/test/ast_gen_test_util.h"

namespace maldoca {
namespace {

INSTANTIATE_TEST_SUITE_P(
    Split, AstGenTest,
    ::testing::Values(AstGenTestParam{
        .ast_def_path = "maldoca/astgen/test/split/ast_def.textproto",
        .cc_namespace = "maldoca",
        .ast_path = "maldoca/astgen/test/split",
        .ir_path = "maldoca/astgen/test/split",
        .expected_ast_header_path = "maldoca/astgen/test/split/ast.generated.h",
        .expected_ast_visitor_header_path =
            "maldoca/astgen/test/split/ast_visitor.generated.h",
        .expected_ast_walker_header_path =
            "maldoca/astgen/test/split/ast_walker.generated.h",
        .expected_ast_source_path =
            "maldoca/astgen/test/split/ast.generated.cc",
        .expected_ast_to_json_path =
            "maldoca/astgen/test/split/ast_to_json.generated.cc",
        .expected_ast_from_json_path =
            "maldoca/astgen/test/split/ast_from_json.generated.cc",
        .expected_ir_tablegen_path =
            "maldoca/astgen/test/split/sir_ops.generated.td",
        .expected_ir_attr_tablegen_path =
            "maldoca/astgen/test/split/sir_attrs.generated.td",
        .expected_ast_to_ir_source_path =
            "maldoca/astgen/test/split/conversion/ast_to_sir.generated.cc",
        .expected_ir_to_ast_source_path =
            "maldoca/astgen/test/split/conversion/sir_to_ast.generated.cc",
    }));

}  // namespace
}  // namespace maldoca
