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
    MultipleInheritance, AstGenTest,
    ::testing::Values(AstGenTestParam{
        .ast_def_path = "maldoca/astgen/test/"
                                "multiple_inheritance/ast_def.textproto",
        .ts_interface_path =
            "maldoca/astgen/test/multiple_inheritance/"
            "ast_ts_interface.generated",
        .cc_namespace = "maldoca",
        .ast_path = "maldoca/astgen/test/multiple_inheritance",
        .expected_ast_header_path =
            "maldoca/astgen/test/"
            "multiple_inheritance/ast.generated.h",
        .expected_ast_source_path =
            "maldoca/astgen/test/"
            "multiple_inheritance/ast.generated.cc",
        .expected_ast_to_json_path =
            "maldoca/astgen/test/"
            "multiple_inheritance/ast_to_json.generated.cc",
        .expected_ast_from_json_path =
            "maldoca/astgen/test/"
            "multiple_inheritance/ast_from_json.generated.cc",
    }));

}  // namespace
}  // namespace maldoca
