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

#include "maldoca/js/ir/analyses/dynamic_constant_propagation/symbol_mutation_info.h"

#include <optional>
#include <string>

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/time/time.h"
#include "maldoca/base/testing/status_matchers.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/driver/conversion.h"
#include "maldoca/js/driver/driver.h"
#include "maldoca/js/driver/driver.pb.h"
#include "maldoca/js/ir/conversion/utils.h"
#include "maldoca/js/ir/ir.h"
#include "maldoca/js/quickjs_babel/quickjs_babel.h"

namespace maldoca {
namespace {

using ::testing::ElementsAreArray;
using ::testing::UnorderedElementsAreArray;

struct LvalueRootSymbolsTestCase {
  std::string source;
  LvalueRootSymbols root_symbols;
};

class GetLvalueRootSymbolsTest
    : public ::testing::TestWithParam<LvalueRootSymbolsTestCase> {};

TEST_P(GetLvalueRootSymbolsTest, GetLvalueRootSymbols) {
  LvalueRootSymbolsTestCase test_case = GetParam();
  QuickJsBabel babel;

  BabelParseRequest request;
  request.set_compute_scopes(true);

  mlir::MLIRContext mlir_context;
  LoadNecessaryDialects(mlir_context);

  MALDOCA_ASSERT_OK_AND_ASSIGN(
      JsHirRepr hir_repr,
      ToJsHirRepr::FromJsSourceRepr(
          test_case.source, request, absl::InfiniteDuration(),
          /*recursion_depth_limit=*/std::nullopt, babel, mlir_context));

  mlir::Value left = nullptr;
  hir_repr.op->walk([&](JsirAssignmentExpressionOp op) {
    ASSERT_EQ(left, nullptr) << "Found multiple declarations or assignments.";
    left = op.getLeft();
  });
  hir_repr.op->walk([&](JsirVariableDeclaratorOp op) {
    ASSERT_EQ(left, nullptr) << "Found multiple declarations or assignments.";
    left = op.getId();
  });

  LvalueRootSymbols root_symbols = GetLvalueRootSymbols(hir_repr.scopes, left);

  EXPECT_THAT(root_symbols.assignment_symbols,
              ElementsAreArray(test_case.root_symbols.assignment_symbols));
  EXPECT_THAT(root_symbols.mutation_symbols,
              ElementsAreArray(test_case.root_symbols.mutation_symbols));
}

INSTANTIATE_TEST_SUITE_P(
    GetLvalueRootSymbolsTest, GetLvalueRootSymbolsTest,
    ::testing::ValuesIn({
        LvalueRootSymbolsTestCase{
            .source = R"js(
              a = b;
            )js",
            .root_symbols =
                LvalueRootSymbols{
                    .assignment_symbols = {JsSymbolId{"a", std::nullopt}},
                    .mutation_symbols = {},
                },
        },
        LvalueRootSymbolsTestCase{
            .source = R"js(
              a.b = c;
            )js",
            .root_symbols =
                LvalueRootSymbols{
                    .assignment_symbols = {},
                    .mutation_symbols = {JsSymbolId{"a", std::nullopt}},
                },
        },
        LvalueRootSymbolsTestCase{
            .source = R"js(
              a.b.c = d;
            )js",
            .root_symbols =
                LvalueRootSymbols{
                    .assignment_symbols = {},
                    .mutation_symbols = {JsSymbolId{"a", std::nullopt}},
                },
        },
        LvalueRootSymbolsTestCase{
            .source = R"js(
              let {key: a} = {key: 0};
            )js",
            .root_symbols =
                LvalueRootSymbols{
                    .assignment_symbols = {JsSymbolId{"a", 0}},
                    .mutation_symbols = {},
                },
        },
    }));

struct GetSymbolMutationInfosTestCase {
  std::string source;
  absl::flat_hash_map<JsSymbolId, SymbolMutationInfo> infos;
};

class GetSymbolMutationInfosTest
    : public ::testing::TestWithParam<GetSymbolMutationInfosTestCase> {};

TEST_P(GetSymbolMutationInfosTest, GetSymbolMutationInfos) {
  GetSymbolMutationInfosTestCase test_case = GetParam();
  QuickJsBabel babel;

  BabelParseRequest request;
  request.set_compute_scopes(true);

  mlir::MLIRContext mlir_context;
  LoadNecessaryDialects(mlir_context);

  MALDOCA_ASSERT_OK_AND_ASSIGN(
      JsHirRepr hir_repr,
      ToJsHirRepr::FromJsSourceRepr(
          test_case.source, request, absl::InfiniteDuration(),
          /*recursion_depth_limit=*/std::nullopt, babel, mlir_context));

  absl::flat_hash_map<JsSymbolId, SymbolMutationInfo> infos =
      GetSymbolMutationInfos(hir_repr.scopes, *hir_repr.op);

  EXPECT_THAT(infos, UnorderedElementsAreArray(test_case.infos));
}

INSTANTIATE_TEST_SUITE_P(GetSymbolMutationInfosTest, GetSymbolMutationInfosTest,
                         ::testing::ValuesIn({
                             GetSymbolMutationInfosTestCase{
                                 .source = R"js(
              a = b;
            )js",
                                 .infos =
                                     {
                                         {
                                             JsSymbolId{"a", std::nullopt},
                                             SymbolMutationInfo{
                                                 .num_assignments = 1,
                                                 .num_mutations = 0,
                                             },
                                         },
                                     }},
                             GetSymbolMutationInfosTestCase{
                                 .source = R"js(
              a.b = c;
            )js",
                                 .infos =
                                     {
                                         {
                                             JsSymbolId{"a", std::nullopt},
                                             SymbolMutationInfo{
                                                 .num_assignments = 0,
                                                 .num_mutations = 1,
                                             },
                                         },
                                     },
                             },
                             GetSymbolMutationInfosTestCase{
                                 .source = R"js(
              a.b.c = d;
            )js",
                                 .infos =
                                     {
                                         {
                                             JsSymbolId{"a", std::nullopt},
                                             SymbolMutationInfo{
                                                 .num_assignments = 0,
                                                 .num_mutations = 1,
                                             },
                                         },
                                     },
                             },
                             GetSymbolMutationInfosTestCase{
                                 .source = R"js(
              let {key: a} = {key: 0};
            )js",
                                 .infos =
                                     {
                                         {
                                             JsSymbolId{"a", 0},
                                             SymbolMutationInfo{
                                                 .num_assignments = 1,
                                                 .num_mutations = 0,
                                             },
                                         },
                                     },
                             },
                         }));

}  // namespace
}  // namespace maldoca
