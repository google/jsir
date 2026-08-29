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

// NOTE:
// This code is adapted from mlir/lib/Transforms/SCCP.cpp

#include "maldoca/js/ir/transforms/constant_propagation/pass.h"

#include <cassert>
#include <vector>

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/FoldUtils.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/ir/analyses/constant_propagation/analysis.h"
#include "maldoca/js/ir/analyses/dataflow_analysis.h"
#include "maldoca/js/ir/analyses/dynamic_constant_propagation/attrs.h"
#include "maldoca/js/ir/ir.h"
#include <optional>
#include <string>
#include <utility>

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "maldoca/js/babel/babel.h"
#include "maldoca/js/babel/scope.h"
#include "maldoca/js/ir/analyses/dynamic_constant_propagation/analysis.h"
#include "maldoca/js/ir/analyses/dynamic_constant_propagation/dynamic_prelude.h"

namespace maldoca {

// Replaces all uses of the given value with a constant if the corresponding
// lattice represents a constant. Returns success if the value was replaced,
// failure otherwise.
static mlir::LogicalResult ReplaceUsesWithConstant(
    JsirDialect *jsir_dialect, JsirConstantPropagationAnalysis &analysis,
    mlir::OpBuilder &builder, mlir::OperationFolder &folder,
    mlir::Value value) {
  // If the value is not used, then there is no need to create a substitute
  // constant op.
  if (value.getUses().empty()) {
    return mlir::success();
  }

  JsirStateRef<JsirConstantPropagationValue> state_ref =
      analysis.GetStateAt(value);
  if (state_ref.value().IsUninitialized() || state_ref.value().IsUnknown()) {
    return mlir::failure();
  }
  mlir::Attribute constant_attr = **state_ref.value();
  mlir::Value constant_value =
      folder.getOrCreateConstant(builder.getInsertionBlock(), jsir_dialect,
                                 constant_attr, value.getType());

  if (constant_value == nullptr) {
    return mlir::failure();
  }
  value.replaceAllUsesWith(constant_value);
  return mlir::success();
}

mlir::LogicalResult PerformConstantPropagation(mlir::Operation *op,
                                               const BabelScopes &scopes) {
  mlir::DataFlowSolver solver;

  auto *analysis = solver.load<JsirConstantPropagationAnalysis>(&scopes);

  mlir::LogicalResult result = solver.initializeAndRun(op);
  if (mlir::failed(result)) {
    return result;
  }

  return PerformConstantPropagation(op, *analysis);
}

mlir::ChangeResult TransformInlineCall(
    mlir::Operation *op, JsirConstantPropagationAnalysis &analysis,
    mlir::OpBuilder &builder) {

  // obj = {
  //   key: (a, b) => a(b)
  //                  ~~~~ inline_call_expr
  //                  ~    inline_call_expr.callee
  // }
  //
  // obj.key(A, B)
  // ~~~~~~~~~~~~~ op == call_expr_op
  // ~~~~~~~       call_expr_op.callee == member_expr == inline_func_expr

  auto call_expr_op = llvm::dyn_cast<JsirCallExpressionOp>(op);
  if (call_expr_op == nullptr) {
    return mlir::ChangeResult::NoChange;
  }

  auto member_expr = llvm::dyn_cast_if_present<JsirMemberExpressionOp>(
      call_expr_op.getCallee().getDefiningOp());
  if (member_expr == nullptr) {
    return mlir::ChangeResult::NoChange;
  }

  JsirStateRef<JsirConstantPropagationValue> state_ref =
      analysis.GetStateAt(member_expr);
  if (state_ref.value().IsUninitialized() || state_ref.value().IsUnknown()) {
    return mlir::ChangeResult::NoChange;
  }
  mlir::Attribute constant_attr = **state_ref.value();

  auto inline_func_expr =
      llvm::dyn_cast<JsirInlineExpressionFunctionAttr>(constant_attr);
  if (inline_func_expr == nullptr) {
    return mlir::ChangeResult::NoChange;
  }

  auto inline_call_expr =
      llvm::dyn_cast<JsirInlineExpressionCallAttr>(inline_func_expr.getBody());
  if (inline_call_expr == nullptr) {
    return mlir::ChangeResult::NoChange;
  }

  auto FindValue = [&](mlir::Attribute attr) -> mlir::Value {
    auto symbol_id_attr = llvm::dyn_cast_if_present<JsirSymbolIdAttr>(attr);
    if (symbol_id_attr == nullptr) {
      return nullptr;
    }

    JsSymbolId symbol_id{symbol_id_attr.getName().str(),
                         symbol_id_attr.getBindingUid()};

    for (auto [idx, param] : llvm::enumerate(inline_func_expr.getParams())) {
      JsSymbolId param_symbol_id(param.getName().str(), param.getBindingUid());
      if (symbol_id == param_symbol_id) {
        if (idx >= call_expr_op.getArguments().size()) {
          return nullptr;
        }
        return call_expr_op.getArguments()[idx];
      }
    }

    return nullptr;
  };

  mlir::Value callee_value = FindValue(inline_call_expr.getCallee());
  if (callee_value == nullptr) {
    return mlir::ChangeResult::NoChange;
  }

  std::vector<mlir::Value> param_values;
  for (mlir::Attribute arg : inline_call_expr.getArguments()) {
    mlir::Value param_value = FindValue(arg);
    if (param_value == nullptr) {
      return mlir::ChangeResult::NoChange;
    }
    param_values.push_back(param_value);
  }

  call_expr_op->replaceAllUsesWith(JsirCallExpressionOp::create(
      builder, call_expr_op.getLoc(), callee_value, param_values));

  return mlir::ChangeResult::Change;
}

mlir::LogicalResult PerformConstantPropagation(
    mlir::Operation *op, JsirConstantPropagationAnalysis &analysis) {
  mlir::MLIRContext *context = op->getContext();
  auto *jsir_dialect = context->getLoadedDialect<JsirDialect>();

  mlir::DominanceInfo dominance_info{op};

  // Initialize the worklist.
  //
  // In general, we want to traverse the program in "reverse" order, so that we
  // can discover dead code earlier and skip some transforms.
  //
  // Imagine that we have code like this with the knowledge that `%arg` holds
  // the constant 100:
  //
  // ```
  // ^BB(%arg):
  //   %r0 = %arg + %arg
  //   %r1 = %r0 + %r0
  // ```
  //
  // The constant propagation analysis would first calculate the values for both
  // %r0 and %r1: %r0 = 200, %r1 = 400.
  //
  // Now, when performing the transformation, if we traverse the program in
  // order, then the following transformations happen in order:
  // +---------------------+  +--------------------+  +----------------+
  // |                     |  |    %c200 = 200     |  |    %c200 = 200 |
  // |                     |  |    ~~~~~~~~~~~     |  |                |
  // |                     |  |                    |  |    %c400 = 400 |
  // |                     |  |                    |  |    ~~~~~~~~~~~ |
  // |                     |  |    ...             |  |    ...         |
  // | ^BB(%arg):          |->| ^BB(%arg):         |->| ^BB(%arg):     |
  // |   %r0 = %arg + %arg |  |    %r0 = %c200     |  |    %r0 = %c200 |
  // |                     |  |          ~~~~~     |  |                |
  // |   %r1 = %r0 + %r0   |  |    %r1 = %r0 + %r0 |  |    %r1 = %c400 |
  // |                     |  |                    |  |          ~~~~~ |
  // +---------------------+  +--------------------+  +----------------+
  //
  // However, if we traverse the program in reverse order, then by replacing
  // `%r0 + %r0` with `%c400`, we have eliminated all uses of `%r0`. At this
  // point, we can safely ignore the transformation on `%arg + %arg`.
  // +---------------------+  +------------------------+
  // |                     |  |    %c400 = 400         |
  // |                     |  |    ~~~~~~~~~~~         |
  // |                     |  |    ...                 |
  // | ^BB(%arg):          |->| ^BB(%arg):             |
  // |   %r0 = %arg + %arg |  |    %r0 = %arg + %arg   |
  // |   %r1 = %r0 + %r0   |  |    %r1 = %c400         |
  // |                     |  |          ~~~~~         |
  // +---------------------+  +------------------------+
  llvm::SmallVector<mlir::Block *> worklist;
  auto add_to_worklist = [&](mlir::MutableArrayRef<mlir::Region> regions) {
    for (mlir::Region &region : regions) {
      switch (region.getBlocks().size()) {
        case 0:
          break;
        case 1:
          worklist.push_back(&region.front());
          break;
        default: {
          auto &dom_tree = dominance_info.getDomTree(&region);
          auto *dom_root_node = dom_tree.getRootNode();
          for (auto *node : llvm::post_order(dom_root_node)) {
            worklist.push_back(node->getBlock());
          }
        }
      }
    }
  };

  // An operation folder used to create and unique constants.
  mlir::OperationFolder folder(context);
  mlir::OpBuilder builder(context);

  add_to_worklist(op->getRegions());
  while (!worklist.empty()) {
    mlir::Block *block = worklist.pop_back_val();

    for (mlir::Operation &op :
         llvm::make_early_inc_range(llvm::reverse(*block))) {
      builder.setInsertionPoint(&op);

      // Replace any result with constants.
      bool replaced_all = true;
      for (mlir::Value res : op.getResults()) {
        replaced_all &= mlir::succeeded(ReplaceUsesWithConstant(
            jsir_dialect, analysis, builder, folder, res));
      }

      // If all of the results of the operation were replaced, try to erase
      // the operation completely.
      if (replaced_all && mlir::wouldOpBeTriviallyDead(&op)) {
        assert(op.use_empty() && "expected all uses to be replaced");
        op.erase();
        continue;
      }

      if (TransformInlineCall(&op, analysis, builder) ==
          mlir::ChangeResult::Change) {
        continue;
      }

      // Add any the regions of this operation to the worklist.
      // Note that if previous transforms have caused the op to be dead, there
      // is no need to traverse its regions, and this logic is skipped.
      add_to_worklist(op.getRegions());
    }

    // Replace any block arguments with constants.
    builder.setInsertionPointToStart(block);
    for (mlir::BlockArgument arg : block->getArguments())
      (void)ReplaceUsesWithConstant(jsir_dialect, analysis, builder, folder,
                                    arg);
  }

  return mlir::success();
}


void JsirConstantPropagationPass::getDependentDialects(
    mlir::DialectRegistry &registry) const {
  Base::getDependentDialects(registry);
  if (babel_ != nullptr) {
    registry.insert<JsirBuiltinDialect>();
  }
}

void JsirConstantPropagationPass::runOnOperation() {
  if (babel_ != nullptr) {
    JsirAnalysisResult::DynamicConstantPropagation detailed_analysis_result;
    mlir::LogicalResult result = PerformDynamicConstantPropagation(
        getOperation(), scopes_, dynamic_config_, *babel_,
        &detailed_analysis_result);
    if (mlir::failed(result)) {
      signalPassFailure();
    }
    if (js_analysis_outputs_ != nullptr) {
      JsAnalysisOutput *js_analysis_output = js_analysis_outputs_->add_outputs();
      JsirAnalysisResult *jsir_analysis_output =
          js_analysis_output->mutable_jsir_analysis();
      *jsir_analysis_output->mutable_dynamic_constant_propagation() =
          std::move(detailed_analysis_result);
    }
    return;
  }

  if (mlir::failed(PerformConstantPropagation(getOperation(), scopes_))) {
    signalPassFailure();
  }
}

mlir::LogicalResult PerformDynamicConstantPropagation(
    mlir::Operation *op, const BabelScopes &scopes,
    const JsirAnalysisConfig::DynamicConstantPropagation &config, Babel &babel,
    JsirAnalysisResult::DynamicConstantPropagation
        *absl_nullable analysis_result) {
  absl::StatusOr<DynamicPrelude> dynamic_prelude =
      DynamicPrelude::Create(config, babel);
  if (!dynamic_prelude.ok()) {
    return mlir::emitError(op->getLoc()) << dynamic_prelude.status().message();
  }

  return PerformDynamicConstantPropagation(op, scopes, &*dynamic_prelude,
                                           analysis_result);
}

mlir::LogicalResult PerformDynamicConstantPropagation(
    mlir::Operation *op, const BabelScopes &scopes,
    DynamicPrelude *dynamic_prelude,
    JsirAnalysisResult::DynamicConstantPropagation
        *absl_nullable analysis_result) {
  using ComputedConstant =
      JsirAnalysisResult::DynamicConstantPropagation::ComputedConstant;

  mlir::DataFlowSolver solver;
  absl::flat_hash_map<JsSymbolId, mlir::Attribute> const_bindings =
      GetConstBindings(scopes, op);

  auto *analysis = solver.load<JsirDynamicConstantPropagationAnalysis>(
      &scopes, dynamic_prelude, const_bindings);

  mlir::LogicalResult result = solver.initializeAndRun(op);
  if (mlir::failed(result)) {
    return result;
  }

  std::vector<ComputedConstant> computed_constants;

  op->walk([&](mlir::Operation *walk_op) {
    ComputedConstant computed_constant;

    if (llvm::isa_and_nonnull<JsirLiteralOpInterface>(walk_op)) {
      return;
    }
    if (walk_op->getNumResults() != 1) {
      return;
    }

    JsirStateRef<JsirConstantPropagationValue> cp_state_ref =
        analysis->GetStateAt(walk_op->getResult(0));
    if (cp_state_ref == nullptr) {
      return;
    }

    const JsirConstantPropagationValue &cp_value = cp_state_ref.value();
    if (cp_value.IsUninitialized() || cp_value.IsUnknown()) {
      return;
    }

    mlir::Attribute cp_attr = **cp_value;
    llvm::TypeSwitch<mlir::Attribute, void>(cp_attr)
        .Case([&](mlir::BoolAttr value) {
          computed_constant.set_bool_value(value.getValue());
        })
        .Case([&](mlir::FloatAttr value) {
          computed_constant.set_number_value(value.getValueAsDouble());
        })
        .Case([&](mlir::StringAttr value) {
          computed_constant.set_string_value(value.getValue());
        })
        .Case([&](JsirBigIntLiteralAttr value) {
          computed_constant.set_big_int_value(value.getValue().str());
        })
        .Default([&](mlir::Attribute value) {});
    if (computed_constant.value_kind_case() ==
        ComputedConstant::VALUE_KIND_NOT_SET) {
      return;
    }

    auto trivia = llvm::dyn_cast_if_present<JsirTriviaAttr>(walk_op->getLoc());
    if (trivia == nullptr) {
      return;
    }
    if (!trivia.getLoc().getStartIndex().has_value()) {
      return;
    }
    if (!trivia.getLoc().getEndIndex().has_value()) {
      return;
    }

    computed_constant.set_start_offset(*trivia.getLoc().getStartIndex());
    computed_constant.set_end_offset(*trivia.getLoc().getEndIndex());

    computed_constants.push_back(std::move(computed_constant));
  });

  if (analysis_result != nullptr) {
    std::string const_bindings_str;
    llvm::raw_string_ostream os(const_bindings_str);
    PrintBindings(os, const_bindings);
    analysis_result->set_bindings(std::move(const_bindings_str));

    analysis_result->mutable_data_flow()->set_output(analysis->PrintOp(op));

    analysis_result->mutable_computed_constants()->Assign(
        computed_constants.begin(), computed_constants.end());
  }

  return PerformDynamicConstantPropagation(op, *analysis);
}

mlir::LogicalResult PerformDynamicConstantPropagation(
    mlir::Operation *op, JsirDynamicConstantPropagationAnalysis &analysis) {
  return PerformConstantPropagation(op, analysis);
}

}  // namespace maldoca
