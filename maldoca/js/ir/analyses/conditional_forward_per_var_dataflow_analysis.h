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

#ifndef MALDOCA_JS_IR_ANALYSES_CONDITIONAL_FORWARD_PER_VAR_DATAFLOW_ANALYSIS_H_
#define MALDOCA_JS_IR_ANALYSES_CONDITIONAL_FORWARD_PER_VAR_DATAFLOW_ANALYSIS_H_

#include <cassert>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "absl/strings/string_view.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/babel/babel.pb.h"
#include "maldoca/js/ir/analyses/conditional_forward_dataflow_analysis.h"
#include "maldoca/js/ir/analyses/dataflow_analysis.h"
#include "maldoca/js/ir/analyses/per_var_state.h"
#include "maldoca/js/ir/analyses/scope.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {

template <typename ValueT>
class JsirConditionalForwardPerVarDataFlowAnalysis
    : public JsirConditionalForwardDataFlowAnalysis<ValueT,
                                                    JsirPerVarState<ValueT>> {
 public:
  using StateT = JsirPerVarState<ValueT>;
  using Base = JsirConditionalForwardDataFlowAnalysis<ValueT, StateT>;

  explicit JsirConditionalForwardPerVarDataFlowAnalysis(
      mlir::DataFlowSolver &solver, const BabelScopes *scopes)
      : Base(solver), scopes_(*scopes) {}

  template <typename OpT>
  using OperandStates =
      OpT::template GenericAdaptor<llvm::ArrayRef<const ValueT *>>;

  using Base::BoundaryInitialValue;
  using Base::GetStateAtEndOf;
  using Base::GetStateAtEntryOf;

 protected:
  void InitializeBoundaryBlock(
      mlir::Block *block, JsirStateRef<StateT> boundary_state,
      llvm::MutableArrayRef<JsirStateRef<ValueT>> arg_states) override {
    // Most of the time, the state at the entry of a region equals to the state
    // before the op enclosing the region.
    //
    // For example, in the code below, states at entries of both true_branch and
    // false_branch should equal to the state before if_statement.
    //
    // ```
    // <state before if_statement>
    // jshir.if_statement (%cond) {
    //   <state at entry of true_branch>
    // }, {
    //   <state at entry of false_branch>
    // }
    // ```
    //
    // In cases like above, we should not initialize the entry state of the
    // region.
    //
    // Only in top-level ops, we initialize with boundary values.
    if (Base::IsEntryBlock(block)) {
      boundary_state.Write(StateT(BoundaryInitialValue()));
      for (auto arg_state : arg_states) {
        arg_state.Write(BoundaryInitialValue());
      }
    }
  }

  void WriteDenseAfterState(mlir::Operation *op, llvm::StringRef name,
                            const ValueT &value, const StateT *before,
                            JsirStateRef<StateT> after) {
    JsSymbolId target_symbol{std::string(name), FindSymbol(scopes_, op, name)};

    after.Join(*before);
    after.Write([&](StateT *after) {
      mlir::ChangeResult changed = mlir::ChangeResult::NoChange;

      for (const auto &[symbol, value] : *before) {
        // Copy all symbols
        if (symbol == target_symbol) {
          continue;
        }
        changed |= after->Set(symbol, value);
      }

      changed |= after->Set(target_symbol, value);

      return changed;
    });
  }

  void VisitIdentifier(JsirIdentifierOp op,
                       OperandStates<JsirIdentifierOp> operands,
                       const StateT *before, JsirStateRef<ValueT> result) {
    absl::string_view name = op.getName();
    JsSymbolId symbol{std::string(name), FindSymbol(scopes_, op, name)};
    ValueT value = before->Get(symbol);
    result.Join(value);
  }

  void VisitJsirPrivateName(JsirPrivateNameOp op,
                            OperandStates<JsirPrivateNameOp> operands,
                            const StateT *before,
                            llvm::MutableArrayRef<JsirStateRef<ValueT>> results,
                            JsirStateRef<StateT> after) {
    absl::string_view name = op.getId().getName().strref();
    JsSymbolId symbol{std::string(name), FindSymbol(scopes_, op, name)};
    ValueT value = before->Get(symbol);

    assert(results.size() == 1);
    auto &result = results[0];
    result.Join(value);
  }

  void VisitVariableDeclaration(
      JsirVariableDeclarationOp op,
      OperandStates<JsirVariableDeclarationOp> operands, const StateT *before,
      llvm::MutableArrayRef<JsirStateRef<ValueT>> results,
      JsirStateRef<StateT> after) {
    if (op.getDeclarations().empty()) {
      after.Join(*before);
      return;
    }

    // TODO(tzx): `declarations_exit` might not be the last block.
    mlir::Block &declarations_entry = op.getDeclarations().front();
    mlir::Block &declarations_exit = op.getDeclarations().back();

    // Propagate into the region.
    JsirStateRef<StateT> entry_state = GetStateAtEntryOf(&declarations_entry);
    entry_state.Join(*before);

    // Propagate from the region.
    JsirStateRef<StateT> exit_state = GetStateAtEndOf(&declarations_exit);
    exit_state.AddDependent(Base::getProgramPointAfter(op));
    after.Join(exit_state.value());
  }

  void VisitVariableDeclarator(
      JsirVariableDeclaratorOp op,
      OperandStates<JsirVariableDeclaratorOp> operands, const StateT *before,
      llvm::MutableArrayRef<JsirStateRef<ValueT>> results,
      JsirStateRef<StateT> after) {
    auto id = llvm::dyn_cast<JsirIdentifierRefOp>(op.getId().getDefiningOp());
    if (id == nullptr) {
      return after.Join(*before);
    }

    const ValueT *init = operands.getInit();
    if (init == nullptr) {
      return after.Join(*before);
    }

    WriteDenseAfterState(op, id.getName(), *init, before, after);
  }

  const BabelScopes &scopes_;
};

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_ANALYSES_CONDITIONAL_FORWARD_PER_VAR_DATAFLOW_ANALYSIS_H_
