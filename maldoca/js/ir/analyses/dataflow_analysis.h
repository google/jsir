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

#ifndef MALDOCA_JS_IR_ANALYSES_DATAFLOW_ANALYSIS_H_
#define MALDOCA_JS_IR_ANALYSES_DATAFLOW_ANALYSIS_H_

#include <cstddef>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/ir/analyses/jump_env.h"
#include "maldoca/js/ir/analyses/state.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {

// The actual data stored on each mlir::ProgramPoint.
// Users should use JsirStateRef instead of this.
namespace detail {
template <typename T>
class JsirStateElement;
}  // namespace detail

// An accessor to the state stored to each mlir::ProgramPoint.
// Writes cause dependents to be visited.
template <typename T>
class JsirStateRef;

namespace detail {

template <typename T>
class JsirStateElement : public mlir::AnalysisState {
 public:
  static_assert(std::is_base_of_v<JsirState<T>, T>,
                "Must use the CRTP type JsirState. "
                "E.g. class MyState : public JsirState<MyState> {};");

  explicit JsirStateElement(mlir::LatticeAnchor anchor)
      : AnalysisState(anchor) {}

  // Read-only. Please use JsirStateRef to modify the value.
  const T &value() const { return value_; }

  void print(llvm::raw_ostream &os) const override { value_.print(os); }

 private:
  friend class JsirStateRef<T>;
  T value_;
};

}  // namespace detail

enum class LivenessKind {
  kLiveIfTrueOrUnknown,
  kLiveIfFalseOrUnknown,
  kLiveIfNullOrUnknown,
  kLiveIfNonNullOrUnknown
};

class JsirGeneralCfgEdge
    : public mlir::GenericLatticeAnchorBase<
          JsirGeneralCfgEdge,
          std::tuple<mlir::ProgramPoint *, mlir::ProgramPoint *,
                     mlir::SmallVector<mlir::Value>,
                     mlir::SmallVector<mlir::Value>,
                     std::optional<std::tuple<mlir::Value, LivenessKind>>>> {
 public:
  using Base::Base;

  mlir::ProgramPoint *getPred() const { return std::get<0>(getValue()); }

  mlir::ProgramPoint *getSucc() const { return std::get<1>(getValue()); }

  const mlir::SmallVector<mlir::Value> &getPredValues() const {
    return std::get<2>(getValue());
  }

  const mlir::SmallVector<mlir::Value> &getSuccValues() const {
    return std::get<3>(getValue());
  }

  std::optional<std::tuple<mlir::Value, LivenessKind>> getLivenessInfo() const {
    return std::get<4>(getValue());
  }

  void print(llvm::raw_ostream &os) const override {
    os << "JsirGeneralCfgEdge";
    os << "\n  pred: ";
    getPred()->print(os);
    os << "\n  succ: ";
    getSucc()->print(os);
    os << "\n  pred values size: ";
    os << getPredValues().size();
    os << "\n  succ values size: ";
    os << getSuccValues().size();
    if (getLivenessInfo().has_value()) {
      os << "\n  liveness kind: ";
      switch (std::get<1>(getLivenessInfo().value())) {
        case LivenessKind::kLiveIfTrueOrUnknown:
          os << "LiveIfTrueOrUnknown";
          break;
        case LivenessKind::kLiveIfFalseOrUnknown:
          os << "LiveIfFalseOrUnknown";
          break;
        case LivenessKind::kLiveIfNullOrUnknown:
          os << "LiveIfNullOrUnknown";
          break;
        case LivenessKind::kLiveIfNonNullOrUnknown:
          os << "LiveIfNonNullOrUnknown";
          break;
      }
    }
  }

  mlir::Location getLoc() const override {
    return mlir::FusedLoc::get(getPred()->getBlock()->getParent()->getContext(),
                               {getPred()->getBlock()->getParent()->getLoc(),
                                getSucc()->getBlock()->getParent()->getLoc()});
  }
};

template <typename T>
class JsirStateRef {
 public:
  static_assert(std::is_base_of_v<JsirState<T>, T>,
                "Must use the CRTP type JsirState. "
                "E.g. class MyState : public JsirState<MyState> {};");

  explicit JsirStateRef()
      : element_(nullptr), solver_(nullptr), analysis_(nullptr) {}

  explicit JsirStateRef(detail::JsirStateElement<T> *element,
                        mlir::DataFlowSolver *solver,
                        mlir::DataFlowAnalysis *analysis)
      : element_(element), solver_(solver), analysis_(analysis) {}

  bool operator==(std::nullptr_t) const { return element_ == nullptr; }
  bool operator!=(std::nullptr_t) const { return element_ != nullptr; }

  detail::JsirStateElement<T> *element() { return element_; }

  const T &value() const { return element_->value(); }

  // Marks a program point as depending on this state.
  // This means that whenever this state is updated, we trigger a visit() of
  // that program point.
  void AddDependent(mlir::ProgramPoint *point);

  // Writes the state and triggers visit()s of its dependents.
  void Write(absl::FunctionRef<mlir::ChangeResult(T *)> write_fn);

  // Writes the state and triggers visit()s of its dependents.
  void Write(T &&lattice);
  void Write(const T &lattice);

  // Joins the state and triggers visit()s of its dependents.
  void Join(const T &lattice);

 private:
  // Points to the actual data attached to the program point.
  detail::JsirStateElement<T> *element_;

  // The solver that drives the worklist algorithm.
  // We need this to access the solver APIs to propagate changes.
  mlir::DataFlowSolver *solver_;

  // The analysis that this state belongs to.
  // When we schedule a new program point to be visited, we need to specify the
  // analysis, hence the need of this field.
  mlir::DataFlowAnalysis *analysis_;
};

// A lattice that represents if a piece of code is executable.
// Join(executable, non-executable) = executable
class JsirExecutable : public JsirState<JsirExecutable> {
 public:
  explicit JsirExecutable(bool executable = false) : executable_(executable) {}

  mlir::ChangeResult Join(const JsirExecutable &other) override;

  const bool &operator*() const { return executable_; }

  bool operator==(const JsirExecutable &rhs) const override {
    return executable_ == rhs.executable_;
  }

  bool operator!=(const JsirExecutable &rhs) const override {
    return !(operator==(rhs));
  }

  void print(llvm::raw_ostream &os) const override;

 private:
  bool executable_ = false;
};

template <typename T>
class JsirDenseStates {
 public:
  virtual ~JsirDenseStates() = default;

  // Gets the state attached before an op.
  virtual T GetStateBefore(mlir::Operation *op) = 0;

  // Gets the state attached after an op.
  virtual T GetStateAfter(mlir::Operation *op) = 0;

  // Gets the state attached at the entry of a block.
  virtual T GetStateAtEntryOf(mlir::Block *block) = 0;

  // Gets the state attached at the end of a block.
  virtual T GetStateAtEndOf(mlir::Block *block) = 0;
};

template <typename T>
class JsirSparseStates {
 public:
  virtual ~JsirSparseStates() = default;

  // Gets the state at an SSA value.
  virtual T GetStateAt(mlir::Value value) = 0;
};

class JsirDataFlowAnalysisPrinter {
 public:
  virtual ~JsirDataFlowAnalysisPrinter() = default;

  // Format:
  //
  // ^block_name:
  //   <AtBlockEntry>%result0 = an_op (%arg0, %arg1, ...)<AfterOp>
  //   %result1 = another_op (%arg0, %arg1, ...)<AfterOp>
  //   ...
  virtual void PrintOp(mlir::Operation *op, size_t num_indents,
                       mlir::AsmState &asm_state, llvm::raw_ostream &os) = 0;

  std::string PrintOp(mlir::Operation *op) {
    std::string output;
    llvm::raw_string_ostream os(output);
    mlir::AsmState asm_state(op);
    PrintOp(op, /*num_indents=*/0, asm_state, os);
    os.flush();
    return output;
  }
};

enum class DataflowDirection { kForward, kBackward };

// =============================================================================
// JsirDenseDataFlowAnalysis
// =============================================================================
// A dataflow analysis API that attaches lattices to operations. This analysis
// supports both forward and backward analysis.
template <typename StateT, DataflowDirection direction>
class JsirDenseDataFlowAnalysis : public mlir::DataFlowAnalysis,
                                  public JsirDataFlowAnalysisPrinter,
                                  public JsirDenseStates<JsirStateRef<StateT>> {
 public:
  explicit JsirDenseDataFlowAnalysis(mlir::DataFlowSolver &solver)
      : mlir::DataFlowAnalysis(solver), solver_(solver) {
    registerAnchorKind<JsirGeneralCfgEdge>();
  }

  // Set the initial state of an entry block for forward analysis or exit block
  // for backward analysis.
  virtual void InitializeBoundaryBlock(mlir::Block *block,
                                       JsirStateRef<StateT> boundary_state) = 0;

  // This virtual method is the transfer function for an operation. It is called
  // by its overloaded protected method. Remember, what the input and output
  // should come from is different in forward and backward analysis. In forward
  // analysis, the input should be the state before the op. In backward
  // analysis, it should be the state after the op.
  // +--------+-------------------+-------------------+
  // |        | Forward Analysis  | Backward Analysis |
  // +--------+-------------------+-------------------+
  // | Input  |       Before      |       After       |
  // +--------+-------------------+-------------------+
  // | Output |       After       |       Before      |
  // +--------+-------------------+-------------------+
  //
  // Input:
  // - The analysis state (lattice value) input for the op
  //
  // Output:
  // - The analysis state (lattice value) output for the op
  virtual void VisitOp(mlir::Operation *op, const StateT *input,
                       JsirStateRef<StateT> output) = 0;

  // Gets the state attached before an op.
  JsirStateRef<StateT> GetStateBefore(mlir::Operation *op) final;

  // Gets the state attached after an op.
  JsirStateRef<StateT> GetStateAfter(mlir::Operation *op) final;

  // Gets the state attached at the entry of a block.
  JsirStateRef<StateT> GetStateAtEntryOf(mlir::Block *block) final;

  // Gets the state attached at the end of a block.
  JsirStateRef<StateT> GetStateAtEndOf(mlir::Block *block) final;

  // Format:
  //
  // ^block_name:
  //   <AtBlockEntry>%result0 = an_op (%arg0, %arg1, ...)<AfterOp>
  //   %result1 = another_op (%arg0, %arg1, ...)<AfterOp>
  //   ...
  void PrintOp(mlir::Operation *op, size_t num_indents,
               mlir::AsmState &asm_state, llvm::raw_ostream &os) override;

  using JsirDataFlowAnalysisPrinter::PrintOp;

  void PrintRegion(mlir::Region &region, size_t num_indents,
                   mlir::AsmState &asm_state, llvm::raw_ostream &os);

  bool IsEntryBlock(mlir::Block *block);

  // When we visit the op, visit all the CFG edges associated with that op.
  absl::flat_hash_map<mlir::Operation *, std::vector<JsirGeneralCfgEdge *>>
      op_to_cfg_edges_;

  // TODO(b/425421947) Consider merging this with `op_to_cfg_edges_`.
  absl::flat_hash_map<mlir::Block *, std::vector<JsirGeneralCfgEdge *>>
      block_to_cfg_edges_;

 protected:
  JsirGeneralCfgEdge *GetCfgEdge(
      mlir::ProgramPoint *pred, mlir::ProgramPoint *succ,
      std::optional<std::tuple<mlir::Value, LivenessKind>> liveness_info,
      llvm::SmallVector<mlir::Value> pred_values,
      llvm::SmallVector<mlir::Value> succ_values);

  void MaybeEmplaceCfgEdge(mlir::ProgramPoint *from, mlir::ProgramPoint *to,
                           mlir::Operation *op,
                           std::optional<std::tuple<mlir::Value, LivenessKind>>
                               liveness_info = std::nullopt,
                           llvm::SmallVector<mlir::Value> pred_values = {},
                           llvm::SmallVector<mlir::Value> succ_values = {});

  static llvm::SmallVector<mlir::Value> GetPredValuesEmpty(mlir::Block *block) {
    return {};
  }

  void MaybeEmplaceCfgEdgesFromRegion(
      mlir::Region &from_exits_of, mlir::ProgramPoint *to, mlir::Operation *op,
      std::optional<std::tuple<mlir::Value, LivenessKind>> liveness_info =
          std::nullopt,
      absl::FunctionRef<llvm::SmallVector<mlir::Value>(mlir::Block *)>
          get_pred_values = GetPredValuesEmpty,
      llvm::SmallVector<mlir::Value> succ_values = {});

  void MaybeEmplaceCfgEdgesBetweenRegions(
      mlir::Region &from_exits_of, mlir::Region &to_entry_of,
      mlir::Operation *op,
      std::optional<std::tuple<mlir::Value, LivenessKind>> liveness_info =
          std::nullopt,
      absl::FunctionRef<llvm::SmallVector<mlir::Value>(mlir::Block *)>
          get_pred_values = GetPredValuesEmpty,
      llvm::SmallVector<mlir::Value> succ_values = {});

  static llvm::SmallVector<mlir::Value> GetExprRegionEndValues(
      mlir::Block *block) {
    auto term_op = block->getTerminator();
    if (auto expr_region_end_op =
            llvm::dyn_cast<JsirExprRegionEndOp>(term_op)) {
      return {expr_region_end_op.getArgument()};
    }
    return llvm::SmallVector<mlir::Value>{};
  }

  static llvm::SmallVector<mlir::Value> GetExprRegionEndValuesFromRegion(
      mlir::Region &region) {
    for (auto &block : region.getBlocks()) {
      if (block.hasNoSuccessors()) {
        auto end_values = GetExprRegionEndValues(&block);
        if (!end_values.empty()) {
          return end_values;
        }
      }
    }
    return llvm::SmallVector<mlir::Value>{};
  }

  static mlir::Region &GetForStatementContinueTargetRegion(
      JshirForStatementOp for_stmt) {
    if (!for_stmt.getUpdate().empty()) {
      return for_stmt.getUpdate();
    }
    if (!for_stmt.getTest().empty()) {
      return for_stmt.getTest();
    }
    return for_stmt.getBody();
  }

  // Gets the state at the program point.
  template <typename T>
  JsirStateRef<T> GetStateImpl(mlir::LatticeAnchor anchor);

  mlir::LogicalResult initialize(mlir::Operation *op) override;
  void InitializeBlock(mlir::Block *block);

  // Since our analysis algorithm is based on MLIR's dataflow analysis, we need
  // to set up the dependency information between basic blocks so that the
  // fixpoint algorithm works.
  // Different analyses may have different strategies. For instance, conditional
  // forward analysis requires to mark whether each successor basic block is
  // executable and selectively add executable basic blocks as successors. Here,
  // we provide a vanilla (unconditional) dependency initialization that
  // provides all successors as dependencies.
  // This method is called inside `InitializeBlock`.
  virtual void InitializeBlockDependencies(mlir::Block *block);
  virtual void VisitBlock(mlir::Block *block);
  virtual void VisitOp(mlir::Operation *op);

  // This method mainly serves to "join" states from blocks. i.e., this method
  // should implement the "join" operation in a dataflow analysis. It should
  // join the states from the end of the predecessor into the entry of the
  // successor for a forward analysis, or join the states from the entry of a
  // block to the end of the predecessor for a backward analysis.
  virtual void VisitCfgEdge(JsirGeneralCfgEdge *edge);

  // Callbacks for `PrintOp`. See comments of `PrintOp` for the format.
  virtual void PrintAtBlockEntry(mlir::Block &block, size_t num_indents,
                                 llvm::raw_ostream &os);
  virtual void PrintAfterOp(mlir::Operation *op, size_t num_indents,
                            mlir::AsmState &asm_state, llvm::raw_ostream &os);

  mlir::DataFlowSolver &solver_;

 private:
  // TODO(b/425421947) Could this be not a member variable?
  JumpEnv jump_env_;

  // TODO(b/425421947) It would be nice to refactor the jump environment so this
  // logic can go in the cases for each op in `initialize`.
  std::optional<JumpTargets> GetJumpTargets(mlir::Operation *op) {
    if (auto with_stmt = llvm::dyn_cast<JshirWithStatementOp>(op);
        with_stmt != nullptr) {
      return JumpTargets{
          .labeled_break_target = getProgramPointAfter(with_stmt),
          .unlabeled_break_target = std::nullopt,
          .continue_target = std::nullopt,
      };
    }
    if (auto if_stmt = llvm::dyn_cast<JshirIfStatementOp>(op);
        if_stmt != nullptr) {
      return JumpTargets{
          .labeled_break_target = getProgramPointAfter(if_stmt),
          .unlabeled_break_target = std::nullopt,
          .continue_target = std::nullopt,
      };
    }
    if (auto switch_stmt = llvm::dyn_cast<JshirSwitchStatementOp>(op);
        switch_stmt != nullptr) {
      return JumpTargets{
          .labeled_break_target = getProgramPointAfter(switch_stmt),
          .unlabeled_break_target = getProgramPointAfter(switch_stmt),
          .continue_target = std::nullopt,
      };
    }
    if (auto while_stmt = llvm::dyn_cast<JshirWhileStatementOp>(op);
        while_stmt != nullptr) {
      return JumpTargets{
          .labeled_break_target = getProgramPointAfter(while_stmt),
          .unlabeled_break_target = getProgramPointAfter(while_stmt),
          .continue_target =
              getProgramPointBefore(&while_stmt.getTest().front()),
      };
    }
    if (auto do_while_stmt = llvm::dyn_cast<JshirDoWhileStatementOp>(op);
        do_while_stmt != nullptr) {
      return JumpTargets{
          .labeled_break_target = getProgramPointAfter(do_while_stmt),
          .unlabeled_break_target = getProgramPointAfter(do_while_stmt),
          .continue_target =
              getProgramPointBefore(&do_while_stmt.getTest().front()),
      };
    }
    if (auto for_stmt = llvm::dyn_cast<JshirForStatementOp>(op);
        for_stmt != nullptr) {
      return JumpTargets{
          .labeled_break_target = getProgramPointAfter(for_stmt),
          .unlabeled_break_target = getProgramPointAfter(for_stmt),
          .continue_target = getProgramPointBefore(
              &GetForStatementContinueTargetRegion(for_stmt).front()),
      };
    }
    if (auto for_in_stmt = llvm::dyn_cast<JshirForInStatementOp>(op);
        for_in_stmt != nullptr) {
      return JumpTargets{
          .labeled_break_target = getProgramPointAfter(for_in_stmt),
          .unlabeled_break_target = getProgramPointAfter(for_in_stmt),
          .continue_target =
              getProgramPointBefore(&for_in_stmt.getBody().front()),
      };
    }
    if (auto for_of_stmt = llvm::dyn_cast<JshirForOfStatementOp>(op);
        for_of_stmt != nullptr) {
      return JumpTargets{
          .labeled_break_target = getProgramPointAfter(for_of_stmt),
          .unlabeled_break_target = getProgramPointAfter(for_of_stmt),
          .continue_target =
              getProgramPointBefore(&for_of_stmt.getBody().front()),
      };
    }
    return std::nullopt;
  }

  std::optional<decltype(jump_env_.WithJumpTargets({}))> WithJumpTargets(
      mlir::Operation *op) {
    auto maybe_jump_targets = GetJumpTargets(op);
    if (maybe_jump_targets.has_value()) {
      return jump_env_.WithJumpTargets(maybe_jump_targets.value());
    }
    return std::nullopt;
  }

  std::optional<decltype(jump_env_.WithLabel({}))> WithLabel(
      mlir::Operation *op) {
    if (auto labeled_stmt = llvm::dyn_cast<JshirLabeledStatementOp>(op);
        labeled_stmt != nullptr) {
      return jump_env_.WithLabel(labeled_stmt.getLabel().getName());
    }
    return std::nullopt;
  }

  // Override `mlir::DataFlowAnalysis::visit` and redirect to `Visit{Op,Block}`.
  mlir::LogicalResult visit(mlir::ProgramPoint *point) override;
};

template <typename StateT>
using JsirDenseForwardDataFlowAnalysis =
    JsirDenseDataFlowAnalysis<StateT, DataflowDirection::kForward>;

template <typename StateT>
using JsirDenseBackwardDataFlowAnalysis =
    JsirDenseDataFlowAnalysis<StateT, DataflowDirection::kBackward>;

// =============================================================================
// JsirDataFlowAnalysis
// =============================================================================
// A dataflow analysis API that attaches lattices to both values and operations.
// This analysis supports both forward and backward analysis.
template <typename ValueT, typename StateT, DataflowDirection direction>
class JsirDataFlowAnalysis
    : public JsirDenseDataFlowAnalysis<StateT, direction>,
      public JsirSparseStates<JsirStateRef<ValueT>> {
 public:
  using Base = JsirDenseDataFlowAnalysis<StateT, direction>;

  explicit JsirDataFlowAnalysis(mlir::DataFlowSolver &solver)
      : JsirDenseDataFlowAnalysis<StateT, direction>(solver) {}

  // The initial state on a boundary `mlir::Value`, e.g. a parameter of an entry
  // block. This is used in both backward and forward analysis, when visiting
  // the CFG edges.
  virtual ValueT BoundaryInitialValue() const = 0;

  // Sets the initial state on a boundary `mlir::Block`, i.e. the entry state of
  // an entry block for a forward analysis, or the exit state of an exit block
  // for a backward analysis.
  virtual void InitializeBoundaryBlock(
      mlir::Block *block, JsirStateRef<StateT> boundary_state,
      llvm::MutableArrayRef<JsirStateRef<ValueT>> arg_states) = 0;

  using Base::InitializeBoundaryBlock;
  void InitializeBoundaryBlock(mlir::Block *block,
                               JsirStateRef<StateT> boundary_state) override {
    std::vector<JsirStateRef<ValueT>> arg_states;
    for (mlir::Value arg : block->getArguments()) {
      arg_states.push_back(GetStateAt(arg));
    }
    return InitializeBoundaryBlock(block, boundary_state, arg_states);
  }

  // This virtual method is the transfer function for an operation. It is called
  // by its overloaded protected method. Same as its version in dense analysis,
  // what the input and output of sparse states (`ValueT`) should come from
  // is different in forward and backward analysis.
  //
  // Generally, a transfer function in a dataflow analysis can be represented in
  // a form of
  //
  //   output = gen ∪ (input - kill)
  //
  // where ∪ is the lattice join operation. Usually, for forward analysis, the
  // `gen` set comes from the `results` in a JSIR `Operation`, and `kill` set
  // comes from the `operands`. For backward analysis, it is the opposite case.
  //
  // For sparse values, we would update the values in `gen` set, and read values
  // from `kill` set. Thus, we have the following table for sparse values:
  // +--------+-------------------+-------------------+
  // |        | Forward Analysis  | Backward Analysis |
  // +--------+-------------------+-------------------+
  // | Input  |      Operands     |     Results       |
  // +--------+-------------------+-------------------+
  // | Output |      Results      |     Operands      |
  // +--------+-------------------+-------------------+
  virtual void VisitOp(
      mlir::Operation *op, llvm::ArrayRef<const ValueT *> sparse_input,
      const StateT *dense_input,
      llvm::MutableArrayRef<JsirStateRef<ValueT>> sparse_output,
      JsirStateRef<StateT> dense_output) = 0;

  // Gets the state at an SSA value.
  JsirStateRef<ValueT> GetStateAt(mlir::Value value) final;

 protected:
  using Base::GetCfgEdge;
  using Base::InitializeBlockDependencies;
  using Base::VisitBlock;
  using Base::VisitOp;
  void VisitCfgEdge(JsirGeneralCfgEdge *edge) override;

  // Helper function to get the `StateRef`s for the operands and results of an
  // op. For forward analysis, the input should be the operands and the output
  // should be the results. For backward analysis, the input should be the
  // results and the output should be the operands.
  struct ValueStateRefs {
    std::vector<const ValueT *> inputs;
    std::vector<JsirStateRef<ValueT>> outputs;
  };
  ValueStateRefs GetValueStateRefs(mlir::Operation *op);

  void PrintAfterOp(mlir::Operation *op, size_t num_indents,
                    mlir::AsmState &asm_state, llvm::raw_ostream &os) override;

 private:
  mlir::LogicalResult initialize(mlir::Operation *op) override;

  // Override the transfer function in `JsirDenseDataFlowAnalysis` and
  // redirect to the transfer function supporting sparse values in
  // `JsirDataFlowAnalysis`.
  void VisitOp(mlir::Operation *op, const StateT *input,
               JsirStateRef<StateT> output) override;

  using Base::solver_;
};

template <typename StateT, typename ValueT>
using JsirForwardDataFlowAnalysis =
    JsirDataFlowAnalysis<StateT, ValueT, DataflowDirection::kForward>;

template <typename StateT, typename ValueT>
using JsirBackwardDataFlowAnalysis =
    JsirDataFlowAnalysis<StateT, ValueT, DataflowDirection::kBackward>;

// =============================================================================
// JsirStateRef
// =============================================================================

template <typename T>
void JsirStateRef<T>::AddDependent(mlir::ProgramPoint *point) {
  element_->addDependency(point, analysis_);
}

template <typename T>
void JsirStateRef<T>::Write(
    absl::FunctionRef<mlir::ChangeResult(T *)> write_fn) {
  mlir::ChangeResult changed = write_fn(&element_->value_);
  solver_->propagateIfChanged(element_, changed);
}

template <typename T>
void JsirStateRef<T>::Write(T &&lattice) {
  if (element_->value_ == lattice) {
    return;
  }

  element_->value_ = std::move(lattice);
  solver_->propagateIfChanged(element_, mlir::ChangeResult::Change);
}

template <typename T>
void JsirStateRef<T>::Write(const T &lattice) {
  T lattice_copy = lattice;
  Write(std::move(lattice_copy));
}

template <typename T>
void JsirStateRef<T>::Join(const T &lattice) {
  mlir::ChangeResult changed = element_->value_.Join(lattice);
  solver_->propagateIfChanged(element_, changed);
}

// =============================================================================
// JsirDenseDataFlowAnalysis
// =============================================================================

template <typename StateT, DataflowDirection direction>
JsirGeneralCfgEdge *JsirDenseDataFlowAnalysis<StateT, direction>::GetCfgEdge(
    mlir::ProgramPoint *pred, mlir::ProgramPoint *succ,
    std::optional<std::tuple<mlir::Value, LivenessKind>> liveness_info,
    llvm::SmallVector<mlir::Value> pred_values,
    llvm::SmallVector<mlir::Value> succ_values) {
  return getLatticeAnchor<JsirGeneralCfgEdge>(pred, succ, pred_values,
                                              succ_values, liveness_info);
}

template <typename StateT, DataflowDirection direction>
void JsirDenseDataFlowAnalysis<StateT, direction>::MaybeEmplaceCfgEdge(
    mlir::ProgramPoint *from, mlir::ProgramPoint *to, mlir::Operation *op,
    std::optional<std::tuple<mlir::Value, LivenessKind>> liveness_info,
    llvm::SmallVector<mlir::Value> pred_values,
    llvm::SmallVector<mlir::Value> succ_values) {
  for (auto &op : *from->getBlock()) {
    if (getProgramPointBefore(&op) == from) {
      break;
    }
    if (llvm::isa<JshirBreakStatementOp>(op) ||
        llvm::isa<JshirContinueStatementOp>(op)) {
      return;
    }
  }

  op_to_cfg_edges_[op].push_back(
      GetCfgEdge(from, to, liveness_info, pred_values, succ_values));
  auto from_state = GetStateImpl<StateT>(from);
  from_state.AddDependent(getProgramPointAfter(op));
}

template <typename StateT, DataflowDirection direction>
void JsirDenseDataFlowAnalysis<StateT, direction>::
    MaybeEmplaceCfgEdgesFromRegion(
        mlir::Region &from_exits_of, mlir::ProgramPoint *to,
        mlir::Operation *op,
        std::optional<std::tuple<mlir::Value, LivenessKind>> liveness_info,
        absl::FunctionRef<llvm::SmallVector<mlir::Value>(mlir::Block *)>
            get_pred_values,
        llvm::SmallVector<mlir::Value> succ_values) {
  for (mlir::Block &block : from_exits_of) {
    if (block.getSuccessors().empty()) {
      mlir::ProgramPoint *after_block = getProgramPointAfter(&block);
      MaybeEmplaceCfgEdge(after_block, to, op, liveness_info,
                          get_pred_values(&block), succ_values);
    }
  }
}

template <typename StateT, DataflowDirection direction>
void JsirDenseDataFlowAnalysis<StateT, direction>::
    MaybeEmplaceCfgEdgesBetweenRegions(
        mlir::Region &from_exits_of, mlir::Region &to_entry_of,
        mlir::Operation *op,
        std::optional<std::tuple<mlir::Value, LivenessKind>> liveness_info,
        absl::FunctionRef<llvm::SmallVector<mlir::Value>(mlir::Block *)>
            get_pred_values,
        llvm::SmallVector<mlir::Value> succ_values) {
  CHECK(!from_exits_of.empty());
  mlir::ProgramPoint *entry = getProgramPointBefore(&to_entry_of.front());
  MaybeEmplaceCfgEdgesFromRegion(from_exits_of, entry, op, liveness_info,
                                 get_pred_values, succ_values);
}

template <typename StateT, DataflowDirection direction>
template <typename T>
JsirStateRef<T> JsirDenseDataFlowAnalysis<StateT, direction>::GetStateImpl(
    mlir::LatticeAnchor anchor) {
  auto *element =
      mlir::DataFlowAnalysis::getOrCreate<detail::JsirStateElement<T>>(anchor);
  return JsirStateRef<T>{element, &solver_, this};
}

template <typename StateT, DataflowDirection direction>
JsirStateRef<StateT>
JsirDenseDataFlowAnalysis<StateT, direction>::GetStateBefore(
    mlir::Operation *op) {
  if (auto *prev_op = op->getPrevNode()) {
    return GetStateAfter(prev_op);
  } else {
    return GetStateImpl<StateT>(getProgramPointBefore(op->getBlock()));
  }
}

template <typename StateT, DataflowDirection direction>
JsirStateRef<StateT>
JsirDenseDataFlowAnalysis<StateT, direction>::GetStateAfter(
    mlir::Operation *op) {
  return GetStateImpl<StateT>(getProgramPointAfter(op));
}

template <typename StateT, DataflowDirection direction>
JsirStateRef<StateT>
JsirDenseDataFlowAnalysis<StateT, direction>::GetStateAtEntryOf(
    mlir::Block *block) {
  return GetStateImpl<StateT>(getProgramPointBefore(block));
}

template <typename StateT, DataflowDirection direction>
JsirStateRef<StateT>
JsirDenseDataFlowAnalysis<StateT, direction>::GetStateAtEndOf(
    mlir::Block *block) {
  if (block->empty()) {
    return GetStateAtEntryOf(block);
  } else {
    return GetStateAfter(&block->back());
  }
}

template <typename StateT, DataflowDirection direction>
void JsirDenseDataFlowAnalysis<StateT, direction>::PrintOp(
    mlir::Operation *op, size_t num_indents, mlir::AsmState &asm_state,
    llvm::raw_ostream &os) {
  size_t num_results = op->getNumResults();
  size_t num_operands = op->getNumOperands();
  size_t num_attributes = op->getAttrs().size();
  size_t num_regions = op->getNumRegions();

  for (size_t i = 0; i != num_results; ++i) {
    if (i != 0) {
      os << ", ";
    }
    op->getResult(i).printAsOperand(os, asm_state);
  }

  if (num_results != 0) {
    os << " = ";
  }

  os << op->getName();

  if (num_operands != 0) {
    os << " (";
    for (size_t i = 0; i != num_operands; ++i) {
      if (i != 0) {
        os << ", ";
      }
      op->getOperand(i).printAsOperand(os, asm_state);
    }
    os << ")";
  }

  if (num_attributes != 0) {
    os << " {";
    for (size_t i = 0; i != num_attributes; ++i) {
      if (i != 0) {
        os << ", ";
      }
      op->getAttrs()[i].getValue().print(os);
    }
    os << "}";
  }

  if (num_regions != 0) {
    os << " (";
    for (size_t i = 0; i != num_regions; ++i) {
      if (i != 0) {
        os << ", ";
      }
      PrintRegion(op->getRegion(i), num_indents, asm_state, os);
    }
    os << ")";
  }

  PrintAfterOp(op, num_indents, asm_state, os);
}

template <typename StateT, DataflowDirection direction>
void JsirDenseDataFlowAnalysis<StateT, direction>::PrintRegion(
    mlir::Region &region, size_t num_indents, mlir::AsmState &asm_state,
    llvm::raw_ostream &os) {
  os << "{\n";
  {
    llvm::SaveAndRestore<size_t> num_indents_in_region{num_indents,
                                                       num_indents + 2};

    for (mlir::Block &block : region.getBlocks()) {
      os.indent(num_indents);
      block.printAsOperand(os, asm_state);
      os << ":\n";

      llvm::SaveAndRestore<size_t> num_indents_in_block{num_indents,
                                                        num_indents + 2};

      PrintAtBlockEntry(block, num_indents, os);

      for (mlir::Operation &op : block) {
        os.indent(num_indents);
        PrintOp(&op, num_indents, asm_state, os);
        os << "\n";
      }
    }
  }
  os.indent(num_indents);
  os << "}";
}

template <typename StateT, DataflowDirection direction>
bool JsirDenseDataFlowAnalysis<StateT, direction>::IsEntryBlock(
    mlir::Block *block) {
  mlir::Operation *parent_op = block->getParentOp();

  if (llvm::isa<JsirProgramOp>(parent_op) || llvm::isa<JsirFileOp>(parent_op) ||
      llvm::isa<JsirFunctionDeclarationOp>(parent_op) ||
      llvm::isa<JsirFunctionExpressionOp>(parent_op) ||
      llvm::isa<JsirObjectMethodOp>(parent_op) ||
      llvm::isa<JsirClassMethodOp>(parent_op) ||
      llvm::isa<JsirClassPrivateMethodOp>(parent_op) ||
      llvm::isa<JsirArrowFunctionExpressionOp>(parent_op)) {
    return block->isEntryBlock();
  }

  return false;
}

// Initializes states on all program points:
// - On every `mlir::Value`:
//   ValueT.
// - After every `mlir::Operation`:
//   StateT.
// - At the entry of every `mlir::Block`:
//   StateT.
// - On every `mlir::Block`:
//   JsirExecutable.
// - On every CFG edge (Block -> Block):
//   JsirExecutable.
template <typename StateT, DataflowDirection direction>
mlir::LogicalResult JsirDenseDataFlowAnalysis<StateT, direction>::initialize(
    mlir::Operation *op) {
  // Register `op`'s dependent state.
  if (op->getParentOp() != nullptr) {
    if constexpr (direction == DataflowDirection::kForward) {
      JsirStateRef<StateT> before_state_ref = GetStateBefore(op);
      before_state_ref.AddDependent(getProgramPointAfter(op));
    } else if constexpr (direction == DataflowDirection::kBackward) {
      JsirStateRef<StateT> after_state_ref = GetStateAfter(op);
      after_state_ref.AddDependent(getProgramPointAfter(op));
    }
  }

  if (auto branch = llvm::dyn_cast<mlir::cf::BranchOp>(op); branch != nullptr) {
    mlir::ProgramPoint *after_branch = getProgramPointAfter(branch);
    mlir::ProgramPoint *before_dest = getProgramPointBefore(branch.getDest());

    llvm::SmallVector<mlir::Value> succ_values = {
        branch.getDest()->getArguments().begin(),
        branch.getDest()->getArguments().end()};

    auto *edge = GetCfgEdge(after_branch, before_dest, std::nullopt,
                            branch.getDestOperands(), succ_values);
    block_to_cfg_edges_[edge->getSucc()->getBlock()].push_back(edge);
  }

  if (auto cond_branch = llvm::dyn_cast<mlir::cf::CondBranchOp>(op);
      cond_branch != nullptr) {
    mlir::ProgramPoint *after_cond_branch = getProgramPointAfter(cond_branch);
    mlir::ProgramPoint *before_true_dest =
        getProgramPointBefore(cond_branch.getTrueDest());
    mlir::ProgramPoint *before_false_dest =
        getProgramPointBefore(cond_branch.getFalseDest());

    llvm::SmallVector<mlir::Value> true_succ_values = {
        cond_branch.getTrueDest()->getArguments().begin(),
        cond_branch.getTrueDest()->getArguments().end()};
    llvm::SmallVector<mlir::Value> false_succ_values = {
        cond_branch.getFalseDest()->getArguments().begin(),
        cond_branch.getFalseDest()->getArguments().end()};

    auto *true_edge =
        GetCfgEdge(after_cond_branch, before_true_dest,
                   std::tuple{cond_branch.getCondition(),
                              LivenessKind::kLiveIfTrueOrUnknown},
                   cond_branch.getTrueDestOperands(), true_succ_values);
    block_to_cfg_edges_[true_edge->getSucc()->getBlock()].push_back(true_edge);

    auto *false_edge =
        GetCfgEdge(after_cond_branch, before_false_dest,
                   std::tuple{cond_branch.getCondition(),
                              LivenessKind::kLiveIfFalseOrUnknown},
                   cond_branch.getFalseDestOperands(), false_succ_values);
    block_to_cfg_edges_[false_edge->getSucc()->getBlock()].push_back(
        false_edge);
  }

  // Handle ops with a single region.
  if (llvm::isa<JsirVariableDeclarationOp>(op) ||
      llvm::isa<JsirObjectExpressionOp>(op) ||
      llvm::isa<JsirClassPropertyOp>(op) ||
      llvm::isa<JsirExportDefaultDeclarationOp>(op) ||
      llvm::isa<JshirWithStatementOp>(op) ||
      llvm::isa<JshirLabeledStatementOp>(op) ||
      llvm::isa<JsirObjectPatternRefOp>(op) ||
      llvm::isa<JsirClassPrivatePropertyOp>(op) ||
      llvm::isa<JsirClassBodyOp>(op) ||
      llvm::isa<JsirClassDeclarationOp>(op) /* TODO Should this be here? */ ||
      llvm::isa<JsirClassExpressionOp>(op) ||
      llvm::isa<JsirExportNamedDeclarationOp>(op)) {
    mlir::ProgramPoint *before_op = getProgramPointBefore(op);
    mlir::ProgramPoint *after_op = getProgramPointAfter(op);

    if (!op->getRegion(0).empty()) {
      mlir::ProgramPoint *before_region =
          getProgramPointBefore(&op->getRegion(0).front());

      MaybeEmplaceCfgEdge(before_op, before_region, op);
      MaybeEmplaceCfgEdgesFromRegion(op->getRegion(0), after_op, op);
    } else {
      MaybeEmplaceCfgEdge(before_op, after_op, op);
    }
  }

  // ┌─────◄
  // │     jshir.if_statement (
  // ├─────► ┌───────────────┐
  // │       │ true region   │
  // │  ┌──◄ └───────────────┘
  // └──│──► ┌───────────────┐
  //    │    │ false region  │
  //    ├──◄ └───────────────┘
  //    │  );
  //    └──►
  if (auto if_stmt = llvm::dyn_cast<JshirIfStatementOp>(op);
      if_stmt != nullptr) {
    mlir::ProgramPoint *before_if_stmt = getProgramPointBefore(if_stmt);
    mlir::ProgramPoint *after_if_stmt = getProgramPointAfter(if_stmt);
    mlir::ProgramPoint *before_consequent =
        getProgramPointBefore(&if_stmt.getConsequent().front());

    MaybeEmplaceCfgEdge(
        before_if_stmt, before_consequent, if_stmt,
        std::tuple{if_stmt.getTest(), LivenessKind::kLiveIfTrueOrUnknown});
    MaybeEmplaceCfgEdgesFromRegion(if_stmt.getConsequent(), after_if_stmt,
                                   if_stmt);

    if (!if_stmt.getAlternate().empty()) {
      mlir::ProgramPoint *before_alternate =
          getProgramPointBefore(&if_stmt.getAlternate().front());

      MaybeEmplaceCfgEdge(
          before_if_stmt, before_alternate, if_stmt,
          std::tuple{if_stmt.getTest(), LivenessKind::kLiveIfFalseOrUnknown});
      MaybeEmplaceCfgEdgesFromRegion(if_stmt.getAlternate(), after_if_stmt,
                                     if_stmt);
    } else {
      MaybeEmplaceCfgEdge(
          before_if_stmt, after_if_stmt, if_stmt,
          std::tuple{if_stmt.getTest(), LivenessKind::kLiveIfFalseOrUnknown});
    }
  }

  // ┌─────◄
  // │     jshir.block_statement (
  // └─────► ┌───────────────┐
  //         │ directives    │
  //    ┌──◄ └───────────────┘
  //    └──► ┌───────────────┐
  //         │ body region   │
  // ┌─────◄ └───────────────┘
  // │     );
  // └─────►
  if (auto block_stmt = llvm::dyn_cast<JshirBlockStatementOp>(op);
      block_stmt != nullptr) {
    mlir::ProgramPoint *before_block_stmt = getProgramPointBefore(block_stmt);
    mlir::ProgramPoint *after_block_stmt = getProgramPointAfter(block_stmt);
    mlir::ProgramPoint *before_directives =
        getProgramPointBefore(&block_stmt.getDirectives().front());

    MaybeEmplaceCfgEdge(before_block_stmt, before_directives, block_stmt);
    MaybeEmplaceCfgEdgesBetweenRegions(block_stmt.getDirectives(),
                                       block_stmt.getBody(), block_stmt);
    MaybeEmplaceCfgEdgesFromRegion(block_stmt.getBody(), after_block_stmt,
                                   block_stmt);
  }

  // ┌─────◄
  // │     jshir.while_statement (
  // ├─────► ┌───────────────┐
  // │       │ test region   │
  // │  ┌──◄ └───────────────┘
  // │  ├──► ┌───────────────┐
  // │  │    │ body region   │
  // └──│──◄ └───────────────┘
  //    │  );
  //    └──►
  if (auto while_stmt = llvm::dyn_cast<JshirWhileStatementOp>(op);
      while_stmt != nullptr) {
    mlir::ProgramPoint *before_while_stmt = getProgramPointBefore(while_stmt);
    mlir::ProgramPoint *after_while_stmt = getProgramPointAfter(while_stmt);
    mlir::ProgramPoint *before_test =
        getProgramPointBefore(&while_stmt.getTest().front());

    MaybeEmplaceCfgEdge(before_while_stmt, before_test, while_stmt);
    MaybeEmplaceCfgEdgesBetweenRegions(
        while_stmt.getTest(), while_stmt.getBody(), while_stmt,
        std::tuple{GetExprRegionEndValuesFromRegion(while_stmt.getTest())[0],
                   LivenessKind::kLiveIfTrueOrUnknown});
    MaybeEmplaceCfgEdgesBetweenRegions(while_stmt.getBody(),
                                       while_stmt.getTest(), while_stmt);
    MaybeEmplaceCfgEdgesFromRegion(
        while_stmt.getTest(), after_while_stmt, while_stmt,
        std::tuple{GetExprRegionEndValuesFromRegion(while_stmt.getTest())[0],
                   LivenessKind::kLiveIfFalseOrUnknown});
  }

  // ┌─────◄
  // │     jshir.do_while_statement (
  // ├─────► ┌───────────────┐
  // │       │ body region   │
  // │  ┌──◄ └───────────────┘
  // │  └──► ┌───────────────┐
  // │       │ test region   │
  // ├─────◄ └───────────────┘
  // │     );
  // └─────►
  if (auto do_while_stmt = llvm::dyn_cast<JshirDoWhileStatementOp>(op);
      do_while_stmt != nullptr) {
    mlir::ProgramPoint *before_do_while_stmt =
        getProgramPointBefore(do_while_stmt);
    mlir::ProgramPoint *after_do_while_stmt =
        getProgramPointAfter(do_while_stmt);
    mlir::ProgramPoint *before_body =
        getProgramPointBefore(&do_while_stmt.getBody().front());

    MaybeEmplaceCfgEdge(before_do_while_stmt, before_body, do_while_stmt);
    MaybeEmplaceCfgEdgesBetweenRegions(do_while_stmt.getBody(),
                                       do_while_stmt.getTest(), do_while_stmt);
    MaybeEmplaceCfgEdgesBetweenRegions(
        do_while_stmt.getTest(), do_while_stmt.getBody(), do_while_stmt,
        std::tuple{GetExprRegionEndValuesFromRegion(do_while_stmt.getTest())[0],
                   LivenessKind::kLiveIfTrueOrUnknown});
    MaybeEmplaceCfgEdgesFromRegion(
        do_while_stmt.getTest(), after_do_while_stmt, do_while_stmt,
        std::tuple{GetExprRegionEndValuesFromRegion(do_while_stmt.getTest())[0],
                   LivenessKind::kLiveIfFalseOrUnknown});
  }

  //    ┌─────◄
  //    │     jshir.for_statement (
  //    └─────► ┌───────────────┐
  //            │ init region   │
  // ┌────────◄ └───────────────┘
  // ├────────► ┌───────────────┐
  // │          │ test region   │
  // │  ┌─────◄ └───────────────┘
  // │  ├─────► ┌───────────────┐
  // │  │       │ body region   │
  // │  │  ┌──◄ └───────────────┘
  // │  │  └──► ┌───────────────┐
  // │  │       │ update region │
  // └──│─────◄ └───────────────┘
  //    │     );
  //    └─────►
  if (auto for_stmt = llvm::dyn_cast<JshirForStatementOp>(op);
      for_stmt != nullptr) {
    mlir::ProgramPoint *before_for_stmt = getProgramPointBefore(for_stmt);
    mlir::ProgramPoint *after_for_stmt = getProgramPointAfter(for_stmt);

    // Emplace an edge into the first non-empty region of the for-statement.
    mlir::Region &first_region =
        !for_stmt.getInit().empty()
            ? for_stmt.getInit()
            : (!for_stmt.getTest().empty() ? for_stmt.getTest()
                                           : for_stmt.getBody());
    mlir::ProgramPoint *before_first_region =
        getProgramPointBefore(&first_region.front());
    MaybeEmplaceCfgEdge(before_for_stmt, before_first_region, for_stmt);

    if (!for_stmt.getInit().empty()) {
      // Init;
      mlir::Region &successor =
          !for_stmt.getTest().empty() ? for_stmt.getTest() : for_stmt.getBody();

      MaybeEmplaceCfgEdgesBetweenRegions(for_stmt.getInit(), successor,
                                         for_stmt);
    }

    if (!for_stmt.getTest().empty()) {
      // Test
      MaybeEmplaceCfgEdgesBetweenRegions(
          for_stmt.getTest(), for_stmt.getBody(), for_stmt,
          std::tuple{GetExprRegionEndValuesFromRegion(for_stmt.getTest())[0],
                     LivenessKind::kLiveIfTrueOrUnknown});
      MaybeEmplaceCfgEdgesFromRegion(
          for_stmt.getTest(), after_for_stmt, for_stmt,
          std::tuple{GetExprRegionEndValuesFromRegion(for_stmt.getTest())[0],
                     LivenessKind::kLiveIfFalseOrUnknown});
    }

    {
      // Body
      MaybeEmplaceCfgEdgesBetweenRegions(
          for_stmt.getBody(), GetForStatementContinueTargetRegion(for_stmt),
          for_stmt);
    }

    if (!for_stmt.getUpdate().empty()) {
      // Update
      mlir::Region &successor =
          !for_stmt.getTest().empty() ? for_stmt.getTest() : for_stmt.getBody();

      MaybeEmplaceCfgEdgesBetweenRegions(for_stmt.getUpdate(), successor,
                                         for_stmt);
    }
  }

  // ┌─────◄
  // │     jshir.for_in_statement (
  // └──┬──► ┌───────────────┐
  //    │    │ body region   │
  // ┌──┴──◄ └───────────────┘
  // │     );
  // └─────►
  if (auto for_in_stmt = llvm::dyn_cast<JshirForInStatementOp>(op);
      for_in_stmt != nullptr) {
    mlir::ProgramPoint *before_for_in_stmt = getProgramPointBefore(for_in_stmt);
    mlir::ProgramPoint *after_for_in_stmt = getProgramPointAfter(for_in_stmt);

    mlir::ProgramPoint *before_body =
        getProgramPointBefore(&for_in_stmt.getBody().front());

    MaybeEmplaceCfgEdge(before_for_in_stmt, before_body, for_in_stmt);
    MaybeEmplaceCfgEdgesBetweenRegions(for_in_stmt.getBody(),
                                       for_in_stmt.getBody(), for_in_stmt);
    MaybeEmplaceCfgEdgesFromRegion(for_in_stmt.getBody(), after_for_in_stmt,
                                   for_in_stmt);
  }

  // ┌─────◄
  // │     jshir.for_of_statement (
  // └──┬──► ┌───────────────┐
  //    │    │ body region   │
  // ┌──┴──◄ └───────────────┘
  // │     );
  // └─────►
  if (auto for_of_stmt = llvm::dyn_cast<JshirForOfStatementOp>(op);
      for_of_stmt != nullptr) {
    mlir::ProgramPoint *before_for_of_stmt = getProgramPointBefore(for_of_stmt);
    mlir::ProgramPoint *after_for_of_stmt = getProgramPointAfter(for_of_stmt);

    mlir::ProgramPoint *before_body =
        getProgramPointBefore(&for_of_stmt.getBody().front());

    MaybeEmplaceCfgEdge(before_for_of_stmt, before_body, for_of_stmt);
    MaybeEmplaceCfgEdgesBetweenRegions(for_of_stmt.getBody(),
                                       for_of_stmt.getBody(), for_of_stmt);
    MaybeEmplaceCfgEdgesFromRegion(for_of_stmt.getBody(), after_for_of_stmt,
                                   for_of_stmt);
  }

  // ┌─────◄
  // │     jshir.logical_expression (
  // ├─────► ┌───────────────┐
  // │       │ right region  │
  // │  ┌──◄ └───────────────┘
  // │  │  );
  // └──┴──►
  if (auto logical_expr = llvm::dyn_cast<JshirLogicalExpressionOp>(op);
      logical_expr != nullptr) {
    LivenessKind after_logical_expr_liveness_kind;
    LivenessKind before_right_liveness_kind;
    switch (*StringToJsLogicalOperator(logical_expr.getOperator_())) {
      case JsLogicalOperator::kAnd:
        // left && right => left ? right : left
        after_logical_expr_liveness_kind = LivenessKind::kLiveIfFalseOrUnknown;
        before_right_liveness_kind = LivenessKind::kLiveIfTrueOrUnknown;
        break;
      case JsLogicalOperator::kOr:
        // left || right => left ? left : right
        after_logical_expr_liveness_kind = LivenessKind::kLiveIfTrueOrUnknown;
        before_right_liveness_kind = LivenessKind::kLiveIfFalseOrUnknown;
        break;
      case JsLogicalOperator::kNullishCoalesce:
        // left ?? right => (left == null) ? right : left
        after_logical_expr_liveness_kind =
            LivenessKind::kLiveIfNonNullOrUnknown;
        before_right_liveness_kind = LivenessKind::kLiveIfNullOrUnknown;
        break;
    }

    mlir::ProgramPoint *before_logical_expr =
        getProgramPointBefore(logical_expr);
    mlir::ProgramPoint *after_logical_expr = getProgramPointAfter(logical_expr);
    mlir::ProgramPoint *before_right =
        getProgramPointBefore(&logical_expr.getRight().front());
    mlir::Value left_value = logical_expr.getLeft();

    MaybeEmplaceCfgEdge(
        before_logical_expr, after_logical_expr, logical_expr,
        std::tuple{left_value, after_logical_expr_liveness_kind}, {left_value},
        logical_expr->getResults());
    MaybeEmplaceCfgEdge(before_logical_expr, before_right, logical_expr,
                        std::tuple{left_value, before_right_liveness_kind});
    MaybeEmplaceCfgEdgesFromRegion(
        logical_expr.getRight(), after_logical_expr, logical_expr, std::nullopt,
        GetExprRegionEndValues, logical_expr->getResults());
  }

  // ┌─────◄
  // │     jshir.conditional_expression (
  // ├─────► ┌───────────────┐
  // │       │ true region   │
  // │  ┌──◄ └───────────────┘
  // └──│──► ┌───────────────┐
  //    │    │ false region  │
  //    ├──◄ └───────────────┘
  //    │  );
  //    └──►
  if (auto conditional_expr = llvm::dyn_cast<JshirConditionalExpressionOp>(op);
      conditional_expr != nullptr) {
    mlir::ProgramPoint *before_conditional_expr =
        getProgramPointBefore(conditional_expr);
    mlir::ProgramPoint *after_conditional_expr =
        getProgramPointAfter(conditional_expr);
    mlir::ProgramPoint *before_consequent =
        getProgramPointBefore(&conditional_expr.getConsequent().front());
    mlir::ProgramPoint *before_alternate =
        getProgramPointBefore(&conditional_expr.getAlternate().front());

    MaybeEmplaceCfgEdge(before_conditional_expr, before_consequent,
                        conditional_expr,
                        std::tuple{conditional_expr.getTest(),
                                   LivenessKind::kLiveIfTrueOrUnknown});
    MaybeEmplaceCfgEdge(before_conditional_expr, before_alternate,
                        conditional_expr,
                        std::tuple{conditional_expr.getTest(),
                                   LivenessKind::kLiveIfFalseOrUnknown});
    MaybeEmplaceCfgEdgesFromRegion(conditional_expr.getConsequent(),
                                   after_conditional_expr, conditional_expr,
                                   std::nullopt, GetExprRegionEndValues,
                                   conditional_expr->getResults());
    MaybeEmplaceCfgEdgesFromRegion(conditional_expr.getAlternate(),
                                   after_conditional_expr, conditional_expr,
                                   std::nullopt, GetExprRegionEndValues,
                                   conditional_expr->getResults());
  }

  if (auto break_stmt = llvm::dyn_cast<JshirBreakStatementOp>(op);
      break_stmt != nullptr) {
    mlir::ProgramPoint *before_break_stmt = getProgramPointBefore(break_stmt);
    absl::StatusOr<mlir::ProgramPoint *> break_target;

    JsirIdentifierAttr label = break_stmt.getLabelAttr();
    if (label == nullptr) {
      break_target = jump_env_.break_target();
    } else {
      break_target = jump_env_.break_target(label.getName());
    }

    if (break_target.ok()) {
      MaybeEmplaceCfgEdge(before_break_stmt, break_target.value(), break_stmt,
                          std::nullopt);
    }
  }

  if (auto continue_stmt = llvm::dyn_cast<JshirContinueStatementOp>(op);
      continue_stmt != nullptr) {
    mlir::ProgramPoint *before_continue_stmt =
        getProgramPointBefore(continue_stmt);
    absl::StatusOr<mlir::ProgramPoint *> continue_target;

    JsirIdentifierAttr label = continue_stmt.getLabelAttr();
    if (label == nullptr) {
      continue_target = jump_env_.continue_target();
    } else {
      continue_target = jump_env_.continue_target(label.getName());
    }

    if (continue_target.ok()) {
      MaybeEmplaceCfgEdge(before_continue_stmt, continue_target.value(),
                          continue_stmt, std::nullopt);
    }
  }

  // ┌─────◄
  // │     jshir.try_statement (
  // └─────► ┌───────────────┐
  //         │ block         │
  //    ┌──◄ └───────────────┘
  //    │    ┌───────────────┐
  //    │    │ handler       │
  //    │    └───────────────┘
  //    ├──► ┌───────────────┐
  //    │    │ finalizer     │
  // ┌──┴──◄ └───────────────┘
  // │     );
  // └─────►
  if (auto try_stmt = llvm::dyn_cast<JshirTryStatementOp>(op);
      try_stmt != nullptr) {
    mlir::ProgramPoint *before_try_stmt = getProgramPointBefore(try_stmt);
    mlir::ProgramPoint *after_try_stmt = getProgramPointAfter(try_stmt);
    mlir::ProgramPoint *before_block =
        getProgramPointBefore(&try_stmt.getBlock().front());

    MaybeEmplaceCfgEdge(before_try_stmt, before_block, try_stmt, std::nullopt);
    if (!try_stmt.getFinalizer().empty()) {
      MaybeEmplaceCfgEdgesBetweenRegions(try_stmt.getBlock(),
                                         try_stmt.getFinalizer(), try_stmt);
      MaybeEmplaceCfgEdgesFromRegion(try_stmt.getFinalizer(), after_try_stmt,
                                     try_stmt, std::nullopt);
    } else {
      MaybeEmplaceCfgEdgesFromRegion(try_stmt.getBlock(), after_try_stmt,
                                     try_stmt, std::nullopt);
    }
  }

  // Get optional jump targets and label to be used during recursive
  // initialization. These variables use RAII.
  auto with_jump_targets = WithJumpTargets(op);
  auto with_label = WithLabel(op);

  // Recursively initialize.
  for (mlir::Region &region : op->getRegions()) {
    for (mlir::Block &block : region.getBlocks()) {
      InitializeBlock(&block);
    }
  }

  return mlir::success();
}

template <typename StateT, DataflowDirection direction>
void JsirDenseDataFlowAnalysis<StateT, direction>::InitializeBlock(
    mlir::Block *block) {
  // Initialize all inner ops.
  for (mlir::Operation &op : *block) {
    initialize(&op);
  }
  InitializeBlockDependencies(block);
  if constexpr (direction == DataflowDirection::kForward) {
    if (IsEntryBlock(block)) {
      JsirStateRef<StateT> block_state_ref = GetStateAtEntryOf(block);
      InitializeBoundaryBlock(block, block_state_ref);
    }

    solver_.enqueue(
        mlir::DataFlowSolver::WorkItem{getProgramPointBefore(block), this});
  } else if constexpr (direction == DataflowDirection::kBackward) {
    // The definition below is copied from https://reviews.llvm.org/D154713.
    auto is_exit_block = [](mlir::Block *b) {
      // Treat empty and terminator-less blocks as exit blocks.
      if (b->empty() ||
          !b->back().mightHaveTrait<mlir::OpTrait::IsTerminator>())
        return true;

      // There may be a weird case where a terminator may be transferring
      // control either to the parent or to another block, so exit blocks and
      // successors are not mutually exclusive.
      mlir::Operation *terminator = b->getTerminator();
      return terminator && terminator->hasTrait<mlir::OpTrait::ReturnLike>();
    };

    if (is_exit_block(block)) {
      JsirStateRef<StateT> block_state_ref = GetStateAtEndOf(block);
      InitializeBoundaryBlock(block, block_state_ref);
    }

    solver_.enqueue(
        mlir::DataFlowSolver::WorkItem{getProgramPointAfter(block), this});
  }
}

template <typename StateT, DataflowDirection direction>
void JsirDenseDataFlowAnalysis<StateT, direction>::InitializeBlockDependencies(
    mlir::Block *block) {
  if constexpr (direction == DataflowDirection::kForward) {
    // For each block, we should update its successor blocks when the state
    // at the end of the block updates. Thus, we enumerate each predecessor's
    // end state and link it to the block.
    for (mlir::Block *pred : block->getPredecessors()) {
      JsirStateRef<StateT> pred_state_ref = GetStateAtEndOf(pred);
      pred_state_ref.AddDependent(getProgramPointBefore(block));
    }
  } else if constexpr (direction == DataflowDirection::kBackward) {
    // For each block, we should update its predecessor blocks when the state
    // at the end of the block updates. Thus, we enumerate each successor's
    // end state and link it to the block.
    for (mlir::Block *succ : block->getSuccessors()) {
      JsirStateRef<StateT> succ_state_ref = GetStateAtEntryOf(succ);
      succ_state_ref.AddDependent(getProgramPointBefore(block));
    }
  }
}

template <typename StateT, DataflowDirection direction>
mlir::LogicalResult JsirDenseDataFlowAnalysis<StateT, direction>::visit(
    mlir::ProgramPoint *point) {
  if (!point->isBlockStart()) {
    VisitOp(point->getPrevOp());
  } else if (!point->isNull()) {
    VisitBlock(point->getBlock());
  }
  return mlir::success();
}

template <typename StateT, DataflowDirection direction>
void JsirDenseDataFlowAnalysis<StateT, direction>::VisitOp(
    mlir::Operation *op) {
  if constexpr (direction == DataflowDirection::kForward) {
    JsirStateRef<StateT> before_state_ref = GetStateBefore(op);
    const StateT *before = &before_state_ref.value();

    JsirStateRef after_state_ref = GetStateAfter(op);

    VisitOp(op, before, after_state_ref);
  } else if constexpr (direction == DataflowDirection::kBackward) {
    JsirStateRef<StateT> after_state_ref = GetStateAfter(op);
    const StateT *after = &after_state_ref.value();

    JsirStateRef before_state_ref = GetStateBefore(op);

    VisitOp(op, after, before_state_ref);
  }
}

template <typename StateT, DataflowDirection direction>
void JsirDenseDataFlowAnalysis<StateT, direction>::VisitBlock(
    mlir::Block *block) {
  for (auto *edge : block_to_cfg_edges_[block]) {
    VisitCfgEdge(edge);
  }
}

template <typename StateT, DataflowDirection direction>
void JsirDenseDataFlowAnalysis<StateT, direction>::VisitCfgEdge(
    JsirGeneralCfgEdge *edge) {
  JsirStateRef<StateT> pred_state_ref = GetStateImpl<StateT>(edge->getPred());
  JsirStateRef<StateT> succ_state_ref = GetStateImpl<StateT>(edge->getSucc());

  if constexpr (direction == DataflowDirection::kForward) {
    // Merge the predecessor into the successor.
    pred_state_ref.AddDependent(edge->getSucc());
    succ_state_ref.Join(pred_state_ref.value());
  } else if constexpr (direction == DataflowDirection::kBackward) {
    // Merge the successor into the predecessor.
    succ_state_ref.AddDependent(edge->getPred());
    pred_state_ref.Join(succ_state_ref.value());
  }
}

template <typename StateT, DataflowDirection direction>
void JsirDenseDataFlowAnalysis<StateT, direction>::PrintAtBlockEntry(
    mlir::Block &block, size_t num_indents, llvm::raw_ostream &os) {
  os.indent(num_indents + 2);
  os << "// ";
  GetStateAtEntryOf(&block).value().print(os);
  os << "\n";
}

template <typename StateT, DataflowDirection direction>
void JsirDenseDataFlowAnalysis<StateT, direction>::PrintAfterOp(
    mlir::Operation *op, size_t num_indents, mlir::AsmState &asm_state,
    llvm::raw_ostream &os) {
  os << "\n";
  os.indent(num_indents + 2);
  os << "// ";
  GetStateAfter(op).value().print(os);
}

// =============================================================================
// JsirDataFlowAnalysis
// =============================================================================

template <typename ValueT, typename StateT, DataflowDirection direction>
JsirStateRef<ValueT>
JsirDataFlowAnalysis<ValueT, StateT, direction>::GetStateAt(mlir::Value value) {
  return Base::template GetStateImpl<ValueT>(value);
}

template <typename ValueT, typename StateT, DataflowDirection direction>
void JsirDataFlowAnalysis<ValueT, StateT, direction>::PrintAfterOp(
    mlir::Operation *op, size_t num_indents, mlir::AsmState &asm_state,
    llvm::raw_ostream &os) {
  for (mlir::Value result : op->getResults()) {
    auto result_state_ref = GetStateAt(result);

    os << "\n";
    os.indent(num_indents + 2);
    os << "// ";
    result.printAsOperand(os, asm_state);
    os << " = ";
    result_state_ref.value().print(os);
  }

  Base::PrintAfterOp(op, num_indents, asm_state, os);
}

template <typename ValueT, typename StateT, DataflowDirection direction>
mlir::LogicalResult JsirDataFlowAnalysis<ValueT, StateT, direction>::initialize(
    mlir::Operation *op) {
  // The op depends on its input operands.
  for (mlir::Value operand : op->getOperands()) {
    JsirStateRef<ValueT> operand_state_ref = GetStateAt(operand);
    operand_state_ref.AddDependent(Base::getProgramPointAfter(op));
  }

  return Base::initialize(op);
}

template <typename ValueT, typename StateT, DataflowDirection direction>
typename JsirDataFlowAnalysis<ValueT, StateT, direction>::ValueStateRefs
JsirDataFlowAnalysis<ValueT, StateT, direction>::GetValueStateRefs(
    mlir::Operation *op) {
  if constexpr (direction == DataflowDirection::kForward) {
    std::vector<const ValueT *> operands;
    for (mlir::Value operand : op->getOperands()) {
      auto operand_state_ref = GetStateAt(operand);
      operands.push_back(&operand_state_ref.value());
    }

    std::vector<JsirStateRef<ValueT>> result_state_refs;
    for (size_t i = 0; i != op->getNumResults(); ++i) {
      mlir::Value result = op->getResult(i);
      JsirStateRef<ValueT> result_state_ref = GetStateAt(result);
      result_state_refs.push_back(std::move(result_state_ref));
    }

    return ValueStateRefs{
        .inputs = std::move(operands),
        .outputs = std::move(result_state_refs),
    };
  } else if constexpr (direction == DataflowDirection::kBackward) {
    std::vector<const ValueT *> results;
    for (size_t i = 0; i != op->getNumResults(); ++i) {
      mlir::Value result = op->getResult(i);
      auto result_state_ref = GetStateAt(result);
      results.push_back(&result_state_ref.value());
    }
    std::vector<JsirStateRef<ValueT>> operand_state_refs;
    for (mlir::Value operand : op->getOperands()) {
      JsirStateRef<ValueT> operand_state_ref = GetStateAt(operand);
      operand_state_refs.push_back(std::move(operand_state_ref));
    }

    return ValueStateRefs{
        .outputs = std::move(operand_state_refs),
        .inputs = std::move(results),
    };
  }
}

template <typename ValueT, typename StateT, DataflowDirection direction>
void JsirDataFlowAnalysis<ValueT, StateT, direction>::VisitOp(
    mlir::Operation *op, const StateT *input, JsirStateRef<StateT> output) {
  if constexpr (direction == DataflowDirection::kForward) {
    auto [operands, result_state_refs] = GetValueStateRefs(op);
    return VisitOp(op, operands, input, result_state_refs, output);
  } else if constexpr (direction == DataflowDirection::kBackward) {
    auto [results, operand_state_refs] = GetValueStateRefs(op);
    return VisitOp(op, results, input, operand_state_refs, output);
  }
}

template <typename ValueT, typename StateT, DataflowDirection direction>
void JsirDataFlowAnalysis<ValueT, StateT, direction>::VisitCfgEdge(
    JsirGeneralCfgEdge *edge) {
  // Match arguments from the predecessor to the successor.
  for (const auto [pred_value, succ_value] :
       llvm::zip(edge->getPredValues(), edge->getSuccValues())) {
    CHECK(pred_value != nullptr);

    JsirStateRef<ValueT> succ_state_ref = GetStateAt(succ_value);
    JsirStateRef<ValueT> pred_state_ref = GetStateAt(pred_value);

    if constexpr (direction == DataflowDirection::kForward) {
      succ_state_ref.Join(pred_state_ref.value());
    } else if constexpr (direction == DataflowDirection::kBackward) {
      pred_state_ref.Join(succ_state_ref.value());
    }
  }

  Base::VisitCfgEdge(edge);
}

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_ANALYSES_DATAFLOW_ANALYSIS_H_
