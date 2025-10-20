// Copyright 2025 Google LLC
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

#include "maldoca/js/driver/conversion.h"

#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/DebugStringHelper.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "nlohmann/json.hpp"
#include "maldoca/base/filesystem.h"
#include "maldoca/base/get_runfiles_dir.h"
#include "maldoca/base/path.h"
#include "maldoca/base/status_macros.h"
#include "maldoca/base/testing/status_matchers.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/babel/babel.pb.h"
#include "maldoca/js/driver/driver.h"
#include "maldoca/js/ir/conversion/utils.h"
#include "maldoca/js/quickjs_babel/quickjs_babel.h"
#include "maldoca/base/testing/protocol-buffer-matchers.h"

namespace maldoca {
namespace {

using testing::EqualsProto;

struct TestCase {
  std::string source;

  // Expected parser result
  nlohmann::ordered_json parsed_ast_json;
  BabelScopes scopes;
  BabelAstString parsed_babel_ast_string;

  // Expected AST
  nlohmann::ordered_json serialized_ast_json;
  std::unique_ptr<JsFile> ast;

  std::unique_ptr<mlir::MLIRContext> mlir_context;

  // Expected HIR
  // Note: `hir_repr` is not constructed by parsing the golden file. Instead, it
  // is constructed by converting the `ast` above. This is because the golden
  // file does not contain loc information.
  JsHirRepr hir_repr;
  std::string hir_dump;

  BabelAstString lifted_babel_ast_string;
};

std::string CompactJsonString(absl::string_view json_str) {
  auto json = nlohmann::ordered_json::parse(json_str);
  return json.dump();
}

void CheckAst(const JsAstRepr &repr, const TestCase &test_case) {
  std::stringstream ss;
  repr.ast->Serialize(ss);
  std::string serialized_ast_json_str = ss.str();

  EXPECT_EQ(serialized_ast_json_str, test_case.serialized_ast_json.dump());
  EXPECT_THAT(repr.scopes, EqualsProto(test_case.scopes));
}

absl::StatusOr<TestCase> GetTestCase() {
  static const char kGoogle3Dir[] = "maldoca/js/driver";
  std::string dir = GetDataDependencyFilepath(kGoogle3Dir);

  auto load_content = [&](absl::string_view file_name) {
    return GetFileContents(JoinPath(dir, file_name));
  };

  MALDOCA_ASSIGN_OR_RETURN(std::string source,
                           load_content("test_source.js.test"));

  MALDOCA_ASSIGN_OR_RETURN(std::string parsed_ast_json_str,
                           load_content("test_parsed_ast.json"));
  auto parsed_ast_json = nlohmann::ordered_json::parse(parsed_ast_json_str);

  BabelScopes scopes;
  MALDOCA_RETURN_IF_ERROR(ParseTextProtoFile(
      GetDataDependencyFilepath("maldoca/js/driver/test_babel_scopes.txtpb"),
      &scopes));

  BabelAstString babel_ast_string;
  babel_ast_string.set_value(CompactJsonString(parsed_ast_json_str));
  babel_ast_string.set_string_literals_base64_encoded(false);
  *babel_ast_string.mutable_scopes() = scopes;

  MALDOCA_ASSIGN_OR_RETURN(std::string serialized_ast_json_str,
                           load_content("test_serialized_ast.json"));
  auto serialized_ast_json =
      nlohmann::ordered_json::parse(serialized_ast_json_str);

  MALDOCA_ASSIGN_OR_RETURN(auto ast, JsFile::FromJson(serialized_ast_json));

  auto mlir_context = std::make_unique<mlir::MLIRContext>();
  LoadNecessaryDialects(*mlir_context);

  MALDOCA_ASSIGN_OR_RETURN(
      JsHirRepr hir_repr,
      ToJsHirRepr::FromJsAstRepr(*ast, scopes, *mlir_context));
  MALDOCA_ASSIGN_OR_RETURN(auto hir_str, load_content("test_hir.mlir.test"));

  BabelAstString lifted_babel_ast_string;
  lifted_babel_ast_string.set_value(CompactJsonString(serialized_ast_json_str));
  lifted_babel_ast_string.set_string_literals_base64_encoded(false);
  *lifted_babel_ast_string.mutable_scopes() = scopes;

  return TestCase{
      .source = source,

      .parsed_ast_json = parsed_ast_json,
      .scopes = scopes,
      .parsed_babel_ast_string = babel_ast_string,

      .serialized_ast_json = serialized_ast_json,
      .ast = std::move(ast),

      .mlir_context = std::move(mlir_context),

      .hir_repr = std::move(hir_repr),
      .hir_dump = hir_str,

      .lifted_babel_ast_string = std::move(lifted_babel_ast_string),
  };
}

// =============================================================================
// Lowering conversions
// =============================================================================

TEST(ConversionTest, SourceToAstString) {
  MALDOCA_ASSERT_OK_AND_ASSIGN(TestCase test_case, GetTestCase());
  QuickJsBabel babel;

  BabelParseRequest parse_request;
  parse_request.set_compute_scopes(true);

  MALDOCA_ASSERT_OK_AND_ASSIGN(
      JsAstStringRepr repr,
      ToJsAstStringRepr::FromJsSourceRepr(test_case.source, parse_request,
                                          absl::InfiniteDuration(), babel));

  EXPECT_THAT(repr.ast_string, EqualsProto(test_case.parsed_babel_ast_string));
}

TEST(ConversionTest, AstStringToAst) {
  MALDOCA_ASSERT_OK_AND_ASSIGN(TestCase test_case, GetTestCase());

  MALDOCA_ASSERT_OK_AND_ASSIGN(
      JsAstRepr repr,
      ToJsAstRepr::FromJsAstStringRepr(test_case.parsed_babel_ast_string,
                                       /*recursion_depth_limit=*/std::nullopt));

  CheckAst(repr, test_case);
}

TEST(ConversionTest, AstToHir) {
  MALDOCA_ASSERT_OK_AND_ASSIGN(TestCase test_case, GetTestCase());

  mlir::MLIRContext mlir_context;
  LoadNecessaryDialects(mlir_context);

  MALDOCA_ASSERT_OK_AND_ASSIGN(
      JsHirRepr repr, ToJsHirRepr::FromJsAstRepr(
                          *test_case.ast, test_case.scopes, mlir_context));

  EXPECT_EQ(mlir::debugString(*repr.op), test_case.hir_dump);
  EXPECT_THAT(repr.scopes, EqualsProto(test_case.scopes));
}

TEST(ConversionTest, SourceToAst) {
  MALDOCA_ASSERT_OK_AND_ASSIGN(TestCase test_case, GetTestCase());
  QuickJsBabel babel;

  BabelParseRequest parse_request;
  parse_request.set_compute_scopes(true);

  MALDOCA_ASSERT_OK_AND_ASSIGN(
      JsAstRepr repr,
      ToJsAstRepr::FromJsSourceRepr(
          test_case.source, parse_request, absl::InfiniteDuration(),
          /*recursion_depth_limit=*/std::nullopt, babel));

  CheckAst(repr, test_case);
}

TEST(ConversionTest, SourceToHir) {
  MALDOCA_ASSERT_OK_AND_ASSIGN(TestCase test_case, GetTestCase());
  QuickJsBabel babel;

  BabelParseRequest parse_request;
  parse_request.set_compute_scopes(true);

  mlir::MLIRContext mlir_context;
  LoadNecessaryDialects(mlir_context);

  MALDOCA_ASSERT_OK_AND_ASSIGN(
      JsHirRepr repr,
      ToJsHirRepr::FromJsSourceRepr(
          test_case.source, parse_request, absl::InfiniteDuration(),
          /*recursion_depth_limit=*/std::nullopt, babel, mlir_context));

  EXPECT_EQ(mlir::debugString(*repr.op), test_case.hir_dump);
  EXPECT_THAT(repr.scopes, EqualsProto(test_case.scopes));
}

TEST(ConversionTest, AstStringToHir) {
  MALDOCA_ASSERT_OK_AND_ASSIGN(TestCase test_case, GetTestCase());

  mlir::MLIRContext mlir_context;
  LoadNecessaryDialects(mlir_context);

  MALDOCA_ASSERT_OK_AND_ASSIGN(
      JsHirRepr repr,
      ToJsHirRepr::FromJsAstStringRepr(test_case.parsed_babel_ast_string,
                                       /*recursion_depth_limit=*/std::nullopt,
                                       mlir_context));

  EXPECT_EQ(mlir::debugString(*repr.op), test_case.hir_dump);
  EXPECT_THAT(repr.scopes, EqualsProto(test_case.scopes));
}

// =============================================================================
// Lifting conversions
// =============================================================================

TEST(ConversionTest, HirToAst) {
  MALDOCA_ASSERT_OK_AND_ASSIGN(TestCase test_case, GetTestCase());

  MALDOCA_ASSERT_OK_AND_ASSIGN(JsAstRepr repr,
                               ToJsAstRepr::FromJsHirRepr(test_case.hir_repr));

  CheckAst(repr, test_case);
}

TEST(ConversionTest, AstToAstString) {
  MALDOCA_ASSERT_OK_AND_ASSIGN(TestCase test_case, GetTestCase());

  MALDOCA_ASSERT_OK_AND_ASSIGN(
      JsAstStringRepr repr,
      ToJsAstStringRepr::FromJsAstRepr(*test_case.ast, test_case.scopes));

  EXPECT_THAT(repr.ast_string, EqualsProto(test_case.lifted_babel_ast_string));
}

TEST(ConversionTest, AstStringToSource) {
  MALDOCA_ASSERT_OK_AND_ASSIGN(TestCase test_case, GetTestCase());

  QuickJsBabel babel;

  BabelGenerateOptions generate_options;

  MALDOCA_ASSERT_OK_AND_ASSIGN(
      JsSourceRepr repr,
      ToJsSourceRepr::FromJsAstStringRepr(test_case.lifted_babel_ast_string,
                                          generate_options,
                                          absl::InfiniteDuration(), babel));

  EXPECT_EQ(repr.source, test_case.source);
}

TEST(ConversionTest, HirToAstString) {
  MALDOCA_ASSERT_OK_AND_ASSIGN(TestCase test_case, GetTestCase());

  MALDOCA_ASSERT_OK_AND_ASSIGN(
      JsAstStringRepr repr,
      ToJsAstStringRepr::FromJsHirRepr(test_case.hir_repr));

  EXPECT_THAT(repr.ast_string, EqualsProto(test_case.lifted_babel_ast_string));
}

TEST(ConversionTest, HirToSource) {
  MALDOCA_ASSERT_OK_AND_ASSIGN(TestCase test_case, GetTestCase());

  QuickJsBabel babel;

  MALDOCA_ASSERT_OK_AND_ASSIGN(
      JsSourceRepr repr,
      ToJsSourceRepr::FromJsHirRepr(test_case.hir_repr, BabelGenerateOptions(),
                                    absl::InfiniteDuration(), babel));
  EXPECT_EQ(repr.source, test_case.source);
}

TEST(ConversionTest, AstToSource) {
  MALDOCA_ASSERT_OK_AND_ASSIGN(TestCase test_case, GetTestCase());

  QuickJsBabel babel;

  MALDOCA_ASSERT_OK_AND_ASSIGN(
      JsSourceRepr repr,
      ToJsSourceRepr::FromJsAstRepr(*test_case.ast, BabelGenerateOptions(),
                                    absl::InfiniteDuration(), babel));
  EXPECT_EQ(repr.source, test_case.source);
}

}  // namespace
}  // namespace maldoca
