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

#include "maldoca/js/ir/transforms/transform.h"

#include <memory>
#include <utility>
#include <vector>

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "maldoca/base/ret_check.h"
#include "maldoca/base/status_macros.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/babel/babel.h"
#include "maldoca/js/driver/driver.pb.h"
#include "maldoca/js/ir/conversion/utils.h"
#include "maldoca/js/ir/ir.h"
#include "maldoca/js/ir/transforms/constant_propagation/pass.h"
#include "maldoca/js/ir/transforms/move_named_functions/pass.h"
#include "maldoca/js/ir/transforms/normalize_object_properties/pass.h"
#include "maldoca/js/ir/transforms/peel_parentheses/pass.h"
#include "maldoca/js/ir/transforms/remove_directives/pass.h"
#include "maldoca/js/ir/transforms/split_declaration_statements/pass.h"
#include "maldoca/js/ir/transforms/split_sequence_expressions/pass.h"

namespace maldoca {

absl::StatusOr<std::unique_ptr<mlir::Pass>> CreateJsirTransformPass(
    const BabelScopes *absl_nullable scopes, const JsirTransformConfig &config,
    Babel *absl_nullable babel,
    JsAnalysisOutputs *absl_nullable analysis_outputs) {
  switch (config.kind_case()) {
    case JsirTransformConfig::KIND_NOT_SET:
      LOG(FATAL) << "No transform config set";

    case JsirTransformConfig::kConstantPropagation: {
      if (scopes == nullptr) {
        return absl::InvalidArgumentError(
            "scopes is required for dynamic constant propagation");
      }
      return std::make_unique<JsirConstantPropagationPass>(scopes);
    }

    case JsirTransformConfig::kMoveNamedFunctions:
      return std::make_unique<MoveNamedFunctionsPass>();

    case JsirTransformConfig::kNormalizeObjectProperties:
      return std::make_unique<NormalizeObjectPropertiesPass>();

    case JsirTransformConfig::kPeelParentheses:
      return std::make_unique<PeelParenthesesPass>();

    case JsirTransformConfig::kSplitSequenceExpressions:
      return std::make_unique<SplitSequenceExpressionsPass>();

    case JsirTransformConfig::kSplitDeclarationStatements:
      return std::make_unique<SplitDeclarationStatementsPass>();

    case JsirTransformConfig::kRemoveDirectives:
      return std::make_unique<RemoveDirectivesPass>();
  }
}

absl::Status TransformJsir(JsirFileOp jsir_file, const BabelScopes &scopes,
                           const JsirTransformConfig &config,
                           Babel *absl_nullable babel,
                           JsAnalysisOutputs *absl_nullable analysis_outputs) {
  std::vector<JsirTransformConfig> configs = {std::move(config)};
  return TransformJsir(jsir_file, scopes, std::move(configs), babel,
                       analysis_outputs);
}

absl::Status TransformJsir(JsirFileOp jsir_file, const BabelScopes &scopes,
                           std::vector<JsirTransformConfig> configs,
                           Babel *absl_nullable babel,
                           JsAnalysisOutputs *absl_nullable analysis_outputs) {
  mlir::PassManager pass_manager{jsir_file.getContext()};

  // TODO(b/204592400): Fix the IR design so that verification passes.
  pass_manager.enableVerifier(false);

  for (auto &&config : configs) {
    MALDOCA_ASSIGN_OR_RETURN(std::unique_ptr<mlir::Pass> pass,
                             CreateJsirTransformPass(&scopes, std::move(config),
                                                     babel, analysis_outputs));
    pass_manager.addPass(std::move(pass));
  }

  mlir::LogicalResult result = pass_manager.run(jsir_file);

  MALDOCA_RET_CHECK(mlir::succeeded(result));
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<JsFile>> TransformJsAst(
    const JsFile &ast, const BabelScopes &scopes,
    std::vector<JsirTransformConfig> configs, Babel *absl_nullable babel) {
  mlir::MLIRContext mlir_context;
  LoadNecessaryDialects(mlir_context);

  MALDOCA_ASSIGN_OR_RETURN(auto jshir_file, AstToJshirFile(ast, mlir_context));

  JsAnalysisOutputs analysis_outputs;
  MALDOCA_RETURN_IF_ERROR(TransformJsir(*jshir_file, scopes, std::move(configs),
                                        babel, &analysis_outputs));

  return JshirFileToAst(*jshir_file);
}

}  // namespace maldoca
