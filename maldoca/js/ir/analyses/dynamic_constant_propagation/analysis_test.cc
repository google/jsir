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

#include "maldoca/js/ir/analyses/dynamic_constant_propagation/analysis.h"

#include <iostream>
#include <optional>

#include "llvm/Support/Casting.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/time/time.h"
#include "maldoca/base/testing/status_matchers.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/ast/ast_util.h"
#include "maldoca/js/ast/transforms/extract_prelude/pass.h"
#include "maldoca/js/babel/babel.h"
#include "maldoca/js/driver/conversion.h"
#include "maldoca/js/driver/driver.h"
#include "maldoca/js/ir/analyses/constant_propagation/analysis.h"
#include "maldoca/js/ir/analyses/dataflow_analysis.h"
#include "maldoca/js/ir/analyses/dynamic_constant_propagation/attrs.h"
#include "maldoca/js/ir/analyses/dynamic_constant_propagation/dynamic_prelude.h"
#include "maldoca/js/ir/conversion/utils.h"
#include "maldoca/js/ir/ir.h"
#include "maldoca/js/quickjs_babel/quickjs_babel.h"

namespace maldoca {
namespace {

static constexpr char kPrelude[] =
    R"js(function prelude() {
  return 0;
}
)js";

static constexpr char kSource[] = R"js(
  var x = prelude();
)js";

static constexpr char kCombined[] = R"js(
  // exec:begin
  function prelude() {
    return 0;
  }
  // exec:end

  var x = prelude();
)js";

void RunTest(const JsirAnalysisConfig::DynamicConstantPropagation &config,
             const JsFile &ast, const BabelScopes &scopes, Babel &babel) {
  MALDOCA_ASSERT_OK_AND_ASSIGN(DynamicPrelude prelude,
                       DynamicPrelude::Create(config, babel));

  mlir::MLIRContext mlir_context;
  LoadNecessaryDialects(mlir_context);
  mlir_context.loadDialect<JsirBuiltinDialect>();

  MALDOCA_ASSERT_OK_AND_ASSIGN(
      JsHirRepr repr, ToJsHirRepr::FromJsAstRepr(ast, scopes, mlir_context));

  // Get the const bindings.
  absl::flat_hash_map<JsSymbolId, mlir::Attribute> const_bindings =
      GetConstBindings(repr.scopes, *repr.op);

  // Run the dataflow analysis.
  mlir::DataFlowSolver solver;
  JsirDynamicConstantPropagationAnalysis *analysis =
      solver.load<JsirDynamicConstantPropagationAnalysis>(
          &repr.scopes, &prelude, const_bindings);

  mlir::LogicalResult mlir_result = solver.initializeAndRun(*repr.op);
  ASSERT_TRUE(mlir::succeeded(mlir_result));

  std::cout << "Analysis result:" << std::endl;
  std::cout << analysis->PrintOp(*repr.op) << std::endl;

  // Now we will manually check that the state attached to the call expression
  // is 0, the inlined function call result.
  JsirCallExpressionOp call_op = nullptr;
  repr.op->walk([&](JsirCallExpressionOp op) {
    CHECK(call_op == nullptr) << "Multiple call expression ops.";
    call_op = op;
  });

  JsirStateRef<JsirConstantPropagationValue> state_ref =
      analysis->GetStateAt(call_op);

  JsirConstantPropagationValue cp_value = state_ref.value();
  ASSERT_FALSE(cp_value.IsUninitialized());
  ASSERT_FALSE(cp_value.IsUnknown());
  mlir::Attribute cp_value_attr = **cp_value;

  auto cp_value_float_attr = llvm::dyn_cast<mlir::FloatAttr>(cp_value_attr);
  ASSERT_NE(cp_value_float_attr, nullptr);
  EXPECT_EQ(cp_value_float_attr.getValue().convertToDouble(), 0.0);
}

TEST(JsirDynamicConstantPropagationAnalysisTest, Basic) {
  QuickJsBabel babel;

  // Convert the source to AST.
  BabelParseRequest parse_request;
  parse_request.set_compute_scopes(true);

  MALDOCA_ASSERT_OK_AND_ASSIGN(JsAstRepr ast_repr,
                       ToJsAstRepr::FromJsSourceRepr(kSource, parse_request,
                                                     absl::InfiniteDuration(),
                                                     std::nullopt, babel));

  // Create the dynamic prelude.
  JsirAnalysisConfig::DynamicConstantPropagation prelude_config;
  prelude_config.set_prelude_source(kPrelude);

  // TODO(tzx): Think about this.
  // prelude_config.set_extracted_from_scope_uid(0);

  RunTest(prelude_config, *ast_repr.ast, ast_repr.scopes, babel);
}

// Similar to the above, but this time the input source contains the prelude,
// and we use `ExtractPrelude` to extract it.
TEST(JsirDynamicConstantPropagationAnalysisTest, CombinedSource) {
  QuickJsBabel babel;

  // Convert the **combined** source to AST.
  BabelParseRequest parse_request;
  parse_request.set_compute_scopes(true);

  MALDOCA_ASSERT_OK_AND_ASSIGN(JsAstRepr ast_repr,
                       ToJsAstRepr::FromJsSourceRepr(kCombined, parse_request,
                                                     absl::InfiniteDuration(),
                                                     std::nullopt, babel));

  // Extract the prelude from the AST.
  JsirAnalysisConfig::DynamicConstantPropagation prelude_config =
      ExtractPreludeByAnnotations(kCombined, *ast_repr.ast);

  EXPECT_EQ(
      PrettyPrintSourceFromSourceString(babel, prelude_config.prelude_source(),
                                        absl::InfiniteDuration()),
      PrettyPrintSourceFromSourceString(babel, kPrelude,
                                        absl::InfiniteDuration()));
  EXPECT_EQ(prelude_config.extracted_from_scope_uid(), 0);

  RunTest(prelude_config, *ast_repr.ast, ast_repr.scopes, babel);
}

// Babel increments its internal scope uid counter after use, causing the global
// scope uid in a subsequent AST to be non-zero. This test verifies that the
// dynamic constant propagation analysis can handle this case.
TEST(JsirDynamicConstantPropagationAnalysisTest, BabelReuse) {
  QuickJsBabel babel;

  BabelParseRequest parse_request;
  parse_request.set_compute_scopes(true);

  // Use the Babel instance so that its internal scope uid counter is
  // incremented.
  MALDOCA_ASSERT_OK(
      babel.Parse(kSource, parse_request, absl::InfiniteDuration()));

  // Convert the **combined** source to AST.
  MALDOCA_ASSERT_OK_AND_ASSIGN(JsAstRepr ast_repr,
                       ToJsAstRepr::FromJsSourceRepr(kCombined, parse_request,
                                                     absl::InfiniteDuration(),
                                                     std::nullopt, babel));

  // Extract the prelude from the AST.
  JsirAnalysisConfig::DynamicConstantPropagation prelude_config =
      ExtractPreludeByAnnotations(kCombined, *ast_repr.ast);

  EXPECT_EQ(
      PrettyPrintSourceFromSourceString(babel, prelude_config.prelude_source(),
                                        absl::InfiniteDuration()),
      PrettyPrintSourceFromSourceString(babel, kPrelude,
                                        absl::InfiniteDuration()));

  std::optional global_scope_uid = ast_repr.ast->program()->scope_uid();
  ASSERT_TRUE(global_scope_uid.has_value());
  EXPECT_EQ(prelude_config.extracted_from_scope_uid(), *global_scope_uid);

  RunTest(prelude_config, *ast_repr.ast, ast_repr.scopes, babel);
}

}  // namespace
}  // namespace maldoca
