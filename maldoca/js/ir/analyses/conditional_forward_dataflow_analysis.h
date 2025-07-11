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

#ifndef MALDOCA_JS_IR_ANALYSES_COND_FORWARD_DATAFLOW_ANALYSIS_H_
#define MALDOCA_JS_IR_ANALYSES_COND_FORWARD_DATAFLOW_ANALYSIS_H_

#include <cstddef>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BlockSupport.h"
#include "mlir/IR/Operation.h"
#include "maldoca/js/ir/analyses/dataflow_analysis.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {

// =============================================================================
// JsirConditionalForwardDataFlowAnalysis
// =============================================================================
// A forward dataflow analysis API that attaches lattices to operations.
//
// Added IsExecutable states for each operation and block, so that some blocks
// will not be visited if it is under a false conditional branch.
// This enables built-in support for dead code analysis.
template <typename ValueT, typename StateT>
class JsirConditionalForwardDataFlowAnalysis
    : public JsirForwardDataFlowAnalysis<ValueT, StateT> {
 public:
  using Base = JsirForwardDataFlowAnalysis<ValueT, StateT>;
  using DenseBase = JsirDenseForwardDataFlowAnalysis<StateT>;

  explicit JsirConditionalForwardDataFlowAnalysis(mlir::DataFlowSolver &solver)
      : JsirForwardDataFlowAnalysis<ValueT, StateT>(solver) {}

  // Gets the information about whether a block is executable.
  JsirStateRef<JsirExecutable> GetIsExecutable(mlir::Block *block);

  // Inherit the same transfer function from base class.
  virtual void VisitOp(mlir::Operation *op,
                       llvm::ArrayRef<const ValueT *> operands,
                       const StateT *before,
                       llvm::MutableArrayRef<JsirStateRef<ValueT>> results,
                       JsirStateRef<StateT> after) = 0;

  virtual bool IsCfgEdgeExecutable(JsirGeneralCfgEdge *edge) { return true; }

  void PrintAtBlockEntry(mlir::Block &block, size_t num_indents,
                         llvm::raw_ostream &os) override {
    os.indent(num_indents + 2);
    os << "// ";
    auto executable_ref = GetIsExecutable(&block);
    executable_ref.value().print(os);
    os << "\n";

    Base::PrintAtBlockEntry(block, num_indents, os);
  }

 private:
  // We override the three methods from base classes to add IsExecutable info.
  void VisitOp(mlir::Operation *op) override;
  void VisitBlock(mlir::Block *block) override;
  void InitializeBlockDependencies(mlir::Block *block) override;
  using Base::VisitCfgEdge;
};

template <typename ValueT, typename StateT>
JsirStateRef<JsirExecutable>
JsirConditionalForwardDataFlowAnalysis<ValueT, StateT>::GetIsExecutable(
    mlir::Block *block) {
  return Base::template GetStateImpl<JsirExecutable>(
      Base::getProgramPointBefore(block));
}

template <typename ValueT, typename StateT>
void JsirConditionalForwardDataFlowAnalysis<ValueT, StateT>::VisitOp(
    mlir::Operation *op) {
  JsirStateRef<StateT> before_state_ref = DenseBase::GetStateBefore(op);
  const StateT *before = &before_state_ref.value();

  JsirStateRef after_state_ref = DenseBase::GetStateAfter(op);

  auto [operands, result_state_refs] = Base::GetValueStateRefs(op);

  for (JsirGeneralCfgEdge *edge : this->op_to_cfg_edges_[op]) {
    if (!IsCfgEdgeExecutable(edge) ||
        !*GetIsExecutable(edge->getPred()->getBlock()).value()) {
      continue;
    }

    JsirStateRef<JsirExecutable> succ_executable_ref =
        GetIsExecutable(edge->getSucc()->getBlock());
    succ_executable_ref.Write(JsirExecutable{true});

    VisitCfgEdge(edge);
  }

  // Don't call the user-defined `VisitOp` if this is an op with a fixed
  // standard visitor.
  // TODO(b/425421947): Create MLIR trait rather than having a list of ops here.
  if (llvm::isa<JshirBlockStatementOp>(op) ||
      llvm::isa<JshirBreakStatementOp>(op) ||
      llvm::isa<JshirConditionalExpressionOp>(op) ||
      llvm::isa<JshirContinueStatementOp>(op) ||
      llvm::isa<JshirDoWhileStatementOp>(op) ||
      llvm::isa<JshirForStatementOp>(op) ||
      llvm::isa<JshirLabeledStatementOp>(op) ||
      llvm::isa<JshirLogicalExpressionOp>(op) ||
      llvm::isa<JshirIfStatementOp>(op) || llvm::isa<JshirTryStatementOp>(op) ||
      llvm::isa<JshirWhileStatementOp>(op)) {
    return;
  }
  VisitOp(op, operands, before, result_state_refs, after_state_ref);
}

template <typename ValueT, typename StateT>
void JsirConditionalForwardDataFlowAnalysis<
    ValueT, StateT>::InitializeBlockDependencies(mlir::Block *block) {
  // The block depends on its incoming CFG edges.
  //
  // In particular, when an incoming CFG is marked as live, the block is
  // visited.
  for (mlir::Block *pred : block->getPredecessors()) {
    mlir::ProgramPoint *after_pred = Base::getProgramPointAfter(pred);
    JsirStateRef<StateT> after_pred_state_ref =
        Base::template GetStateImpl<StateT>(after_pred);
    after_pred_state_ref.AddDependent(Base::getProgramPointBefore(block));
  }

  // The first time the block is marked as executable, visit all ops.
  //
  // This is because some ops (e.g. constant) do not have other dependencies.
  JsirStateRef<JsirExecutable> block_executable_ref = GetIsExecutable(block);
  for (mlir::Operation &op : *block) {
    block_executable_ref.AddDependent(Base::getProgramPointAfter(&op));
  }

  if (Base::IsEntryBlock(block)) {
    // Entry blocks are always executable.
    // This also triggers all ops to be visited.
    block_executable_ref.Write(JsirExecutable{true});
  }
}

template <typename ValueT, typename StateT>
void JsirConditionalForwardDataFlowAnalysis<ValueT, StateT>::VisitBlock(
    mlir::Block *block) {
  for (auto *edge : Base::block_to_cfg_edges_[block]) {
    if (!IsCfgEdgeExecutable(edge) ||
        !*GetIsExecutable(edge->getPred()->getBlock()).value()) {
      continue;
    }

    // If this is a flip, it causes all ops in the block to be visited.
    JsirStateRef<JsirExecutable> block_executable_ref = GetIsExecutable(block);
    block_executable_ref.Write(JsirExecutable{true});

    VisitCfgEdge(edge);
  }
}

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_ANALYSES_COND_FORWARD_DATAFLOW_ANALYSIS_H_
