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

#include "maldoca/js/ir/analyses/analysis.h"

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "llvm/Support/raw_ostream.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "maldoca/base/ret_check.h"
#include "maldoca/base/status_macros.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/babel/babel.h"
#include "maldoca/js/babel/babel.pb.h"
#include "maldoca/js/ir/analyses/constant_propagation/analysis.h"
#include "maldoca/js/ir/analyses/dataflow_analysis.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {

template <typename AnalysisT, typename... Args>
static absl::StatusOr<JsirAnalysisResult::DataFlow> RunJsirDataFlowAnalysis(
    mlir::Operation *op, Args &&...args) {
  static_assert(std::is_base_of_v<JsirDataFlowAnalysisPrinter, AnalysisT>,
                "The analysis must inherit JsirDataFlowAnalysisPrinter.");

  mlir::DataFlowSolver solver;

  JsirDataFlowAnalysisPrinter *analysis =
      solver.load<AnalysisT>(std::forward<Args>(args)...);

  mlir::LogicalResult mlir_result = solver.initializeAndRun(op);
  MALDOCA_RET_CHECK(mlir::succeeded(mlir_result));

  JsirAnalysisResult::DataFlow result;
  result.set_output(analysis->PrintOp(op));
  return result;
}

absl::StatusOr<JsirAnalysisResult> RunJsirAnalysis(
    JsirFileOp op, const BabelScopes &scopes, const JsirAnalysisConfig &config,
    Babel *absl_nullable babel) {
  switch (config.kind_case()) {
    case JsirAnalysisConfig::KIND_NOT_SET: {
      return absl::InvalidArgumentError("JsAnalysisConfig kind not set");
    }

    case JsirAnalysisConfig::kConstantPropagation: {
      MALDOCA_ASSIGN_OR_RETURN(
          JsirAnalysisResult::DataFlow detailed_result,
          RunJsirDataFlowAnalysis<JsirConstantPropagationAnalysis>(op,
                                                                   &scopes));

      JsirAnalysisResult result;
      result.mutable_constant_propagation()->Swap(&detailed_result);
      return result;
    }
  }
}

}  // namespace maldoca
