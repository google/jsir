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

#ifndef MALDOCA_JS_IR_ANALYSES_DATAFLOW_ANALYSIS_H_
#define MALDOCA_JS_IR_ANALYSES_DATAFLOW_ANALYSIS_H_

#include <cstddef>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
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
  const T& value() const { return value_; }

  void print(llvm::raw_ostream& os) const override { value_.print(os); }

 private:
  friend class JsirStateRef<T>;
  T value_;
};

}  // namespace detail

enum class LivenessKind {
  kLiveIfTruthyOrUnknown,
  kLiveIfFalsyOrUnknown,
  kLiveIfEqualOrUnknown,
  kLiveIfNotEqualOrUnknown
};

// TODO Can we extend an std::tuple instead of defining the DenseMapInfo below?
struct LivenessInfo {
  LivenessKind kind;
  llvm::SmallVector<llvm::PointerUnion<mlir::Value, mlir::Attribute>> values;

  bool operator==(const LivenessInfo& other) const {
    return kind == other.kind && values == other.values;
  }
};

}  // namespace maldoca

namespace llvm {

template <>
struct DenseMapInfo<maldoca::LivenessInfo> {
  using TupleInfo = llvm::DenseMapInfo<std::tuple<
      maldoca::LivenessKind,
      llvm::SmallVector<llvm::PointerUnion<mlir::Value, mlir::Attribute>>>>;
  using KindInfo = llvm::DenseMapInfo<maldoca::LivenessKind>;
  using ValueInfo =
      llvm::DenseMapInfo<llvm::PointerUnion<mlir::Value, mlir::Attribute>>;

  static maldoca::LivenessInfo getEmptyKey() {
    auto [kind, values] = TupleInfo::getEmptyKey();
    return maldoca::LivenessInfo{.kind = kind, .values = std::move(values)};
  }

  static maldoca::LivenessInfo getTombstoneKey() {
    auto [kind, values] = TupleInfo::getTombstoneKey();
    return maldoca::LivenessInfo{.kind = kind, .values = std::move(values)};
  }

  static llvm::hash_code getHashValue(maldoca::LivenessInfo info) {
    auto kind_hash = KindInfo::getHashValue(info.kind);
    for (auto value : info.values) {
      kind_hash = hash_combine(kind_hash, ValueInfo::getHashValue(value));
    }
    return kind_hash;
  }

  static bool isEqual(const maldoca::LivenessInfo& a,
                      const maldoca::LivenessInfo& b) {
    return TupleInfo::isEqual({a.kind, a.values}, {b.kind, b.values});
  }
};

}  // namespace llvm

namespace maldoca {

class JsirGeneralCfgEdge
    : public mlir::GenericLatticeAnchorBase<
          JsirGeneralCfgEdge,
          std::tuple<mlir::ProgramPoint*, mlir::ProgramPoint*,
                     mlir::SmallVector<mlir::Value>,
                     mlir::SmallVector<mlir::Value>,
                     std::optional<LivenessInfo>>> {
 public:
  using Base::Base;

  mlir::ProgramPoint* getPred() const { return std::get<0>(getValue()); }

  mlir::ProgramPoint* getSucc() const { return std::get<1>(getValue()); }

  const mlir::SmallVector<mlir::Value>& getPredValues() const {
    return std::get<2>(getValue());
  }

  const mlir::SmallVector<mlir::Value>& getSuccValues() const {
    return std::get<3>(getValue());
  }

  std::optional<LivenessInfo> getLivenessInfo() const {
    return std::get<4>(getValue());
  }

  void print(llvm::raw_ostream& os) const override {
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
      switch (getLivenessInfo().value().kind) {
        case LivenessKind::kLiveIfTruthyOrUnknown:
          os << "LiveIfTruthyOrUnknown";
          break;
        case LivenessKind::kLiveIfFalsyOrUnknown:
          os << "LiveIfFalsyOrUnknown";
          break;
        case LivenessKind::kLiveIfEqualOrUnknown:
          os << "LiveIfEqualOrUnknown";
          break;
        case LivenessKind::kLiveIfNotEqualOrUnknown:
          os << "LiveIfNotEqualOrUnknown";
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

  explicit JsirStateRef(detail::JsirStateElement<T>* element,
                        mlir::DataFlowSolver* solver,
                        mlir::DataFlowAnalysis* analysis)
      : element_(element), solver_(solver), analysis_(analysis) {}

  bool operator==(std::nullptr_t) const { return element_ == nullptr; }
  bool operator!=(std::nullptr_t) const { return element_ != nullptr; }

  detail::JsirStateElement<T>* element() { return element_; }

  const T& value() const { return element_->value(); }

  // Marks a program point as depending on this state.
  // This means that whenever this state is updated, we trigger a visit() of
  // that program point.
  void AddDependent(mlir::ProgramPoint* point);

  // Writes the state and triggers visit()s of its dependents.
  void Write(absl::FunctionRef<mlir::ChangeResult(T*)> write_fn);

  // Writes the state and triggers visit()s of its dependents.
  void Write(T&& lattice);
  void Write(const T& lattice);

  // Joins the state and triggers visit()s of its dependents.
  void Join(const T& lattice);

 private:
  // Points to the actual data attached to the program point.
  detail::JsirStateElement<T>* element_;

  // The solver that drives the worklist algorithm.
  // We need this to access the solver APIs to propagate changes.
  mlir::DataFlowSolver* solver_;

  // The analysis that this state belongs to.
  // When we schedule a new program point to be visited, we need to specify the
  // analysis, hence the need of this field.
  mlir::DataFlowAnalysis* analysis_;
};

// A lattice that represents if a piece of code is executable.
// Join(executable, non-executable) = executable
class JsirExecutable : public JsirState<JsirExecutable> {
 public:
  explicit JsirExecutable(bool executable = false) : executable_(executable) {}

  mlir::ChangeResult Join(const JsirExecutable& other) override;

  const bool& operator*() const { return executable_; }

  bool operator==(const JsirExecutable& rhs) const override {
    return executable_ == rhs.executable_;
  }

  bool operator!=(const JsirExecutable& rhs) const override {
    return !(operator==(rhs));
  }

  void print(llvm::raw_ostream& os) const override;

 private:
  bool executable_ = false;
};

template <typename T>
class JsirDenseStates {
 public:
  virtual ~JsirDenseStates() = default;

  // Gets the state attached before an op.
  virtual T GetStateBefore(mlir::Operation* op) = 0;

  // Gets the state attached after an op.
  virtual T GetStateAfter(mlir::Operation* op) = 0;

  // Gets the state attached at the entry of a block.
  virtual T GetStateAtEntryOf(mlir::Block* block) = 0;

  // Gets the state attached at the end of a block.
  virtual T GetStateAtEndOf(mlir::Block* block) = 0;
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
  virtual void PrintOp(mlir::Operation* op, size_t num_indents,
                       mlir::AsmState& asm_state, llvm::raw_ostream& os) = 0;

  std::string PrintOp(mlir::Operation* op) {
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
// JsirDataFlowAnalysis
// =============================================================================
// A dataflow analysis API that attaches lattices to both values and operations.
// This analysis supports both forward and backward analysis.
template <typename ValueT, typename StateT, DataflowDirection direction>
class JsirDataFlowAnalysis : public mlir::DataFlowAnalysis,
                             public JsirDataFlowAnalysisPrinter,
                             public JsirDenseStates<JsirStateRef<StateT>>,
                             public JsirSparseStates<JsirStateRef<ValueT>> {
 public:
  explicit JsirDataFlowAnalysis(mlir::DataFlowSolver& solver)
      : mlir::DataFlowAnalysis(solver), solver_(solver) {
    registerAnchorKind<JsirGeneralCfgEdge>();
  }

  // Set the initial state of an entry block for forward analysis or exit block
  // for backward analysis.
  virtual void InitializeBoundaryBlock(mlir::Block* block,
                                       JsirStateRef<StateT> boundary_state) {
    std::vector<JsirStateRef<ValueT>> arg_states;
    for (mlir::Value arg : block->getArguments()) {
      arg_states.push_back(GetStateAt(arg));
    }
    return InitializeBoundaryBlock(block, boundary_state, arg_states);
  }

  // The initial state on a boundary `mlir::Value`, e.g. a parameter of an entry
  // block. This is used in both backward and forward analysis, when visiting
  // the CFG edges.
  virtual ValueT BoundaryInitialValue() const = 0;

  // Sets the initial state on a boundary `mlir::Block`, i.e. the entry state of
  // an entry block for a forward analysis, or the exit state of an exit block
  // for a backward analysis.
  virtual void InitializeBoundaryBlock(
      mlir::Block* block, JsirStateRef<StateT> boundary_state,
      llvm::MutableArrayRef<JsirStateRef<ValueT>> arg_states) = 0;

  // Gets the state attached before an op.
  JsirStateRef<StateT> GetStateBefore(mlir::Operation* op) final;

  // Gets the state attached after an op.
  JsirStateRef<StateT> GetStateAfter(mlir::Operation* op) final;

  // Gets the state attached at the entry of a block.
  JsirStateRef<StateT> GetStateAtEntryOf(mlir::Block* block) final;

  // Gets the state attached at the end of a block.
  JsirStateRef<StateT> GetStateAtEndOf(mlir::Block* block) final;

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
      mlir::Operation* op, llvm::ArrayRef<const ValueT*> sparse_input,
      const StateT* dense_input,
      llvm::MutableArrayRef<JsirStateRef<ValueT>> sparse_output,
      JsirStateRef<StateT> dense_output) = 0;

  // Gets the state at an SSA value.
  JsirStateRef<ValueT> GetStateAt(mlir::Value value) final;

  // Format:
  //
  // ^block_name:
  //   <AtBlockEntry>%result0 = an_op (%arg0, %arg1, ...)<AfterOp>
  //   %result1 = another_op (%arg0, %arg1, ...)<AfterOp>
  //   ...
  void PrintOp(mlir::Operation* op, size_t num_indents,
               mlir::AsmState& asm_state, llvm::raw_ostream& os) override;

  using JsirDataFlowAnalysisPrinter::PrintOp;

  void PrintRegion(mlir::Region& region, size_t num_indents,
                   mlir::AsmState& asm_state, llvm::raw_ostream& os);

  // Callbacks for `PrintOp`. See comments of `PrintOp` for the format.
  virtual void PrintAtBlockEntry(mlir::Block& block, size_t num_indents,
                                 llvm::raw_ostream& os);
  virtual void PrintAfterOp(mlir::Operation* op, size_t num_indents,
                            mlir::AsmState& asm_state, llvm::raw_ostream& os);

  bool IsEntryBlock(mlir::Block* block);

  // When we visit the op, visit all the CFG edges associated with that op.
  absl::flat_hash_map<mlir::Operation*, std::vector<JsirGeneralCfgEdge*>>
      op_to_cfg_edges_;

  // TODO(b/425421947) Consider merging this with `op_to_cfg_edges_`.
  absl::flat_hash_map<mlir::Block*, std::vector<JsirGeneralCfgEdge*>>
      block_to_cfg_edges_;

 protected:
  struct CfgEdgeOptions {
    llvm::SmallVector<mlir::ProgramPoint*> from;
    llvm::SmallVector<mlir::ProgramPoint*> to;
    llvm::PointerUnion<mlir::Operation*, mlir::Block*> owner;
    std::optional<LivenessInfo> liveness_info;
    std::variant<mlir::ValueRange,
                 absl::FunctionRef<mlir::ValueRange(mlir::Block*)>>
        pred_values;
    mlir::ValueRange succ_values;
  };

  void MaybeEmplaceCfgEdges(CfgEdgeOptions options) {
    for (auto& from : options.from) {
      for (auto& to : options.to) {
        for (auto& op : *from->getBlock()) {
          if (getProgramPointBefore(&op) == from) {
            break;
          }
          if (llvm::isa<JshirBreakStatementOp>(op) ||
              llvm::isa<JshirContinueStatementOp>(op)) {
            return;
          }
        }

        mlir::ValueRange pred_values;
        if (std::holds_alternative<mlir::ValueRange>(options.pred_values)) {
          pred_values = std::get<mlir::ValueRange>(options.pred_values);
        } else {
          pred_values =
              std::get<absl::FunctionRef<mlir::ValueRange(mlir::Block*)>>(
                  options.pred_values)(from->getBlock());
        }

        JsirGeneralCfgEdge* edge = getLatticeAnchor<JsirGeneralCfgEdge>(
            from, to, pred_values, options.succ_values, options.liveness_info);

        mlir::ProgramPoint* dependent = nullptr;
        if (auto* owner_op = llvm::dyn_cast<mlir::Operation*>(options.owner)) {
          op_to_cfg_edges_[owner_op].push_back(edge);
          dependent = getProgramPointAfter(owner_op);
        } else if (auto* owner_block =
                       llvm::dyn_cast<mlir::Block*>(options.owner)) {
          block_to_cfg_edges_[owner_block].push_back(edge);
          dependent = getProgramPointAfter(owner_block);
        }

        auto from_state = GetStateImpl<StateT>(from);
        from_state.AddDependent(dependent);
        for (auto pred_value : pred_values) {
          auto pred_state = GetStateImpl<ValueT>(pred_value);
          pred_state.AddDependent(dependent);
        }
      }
    }
  }

  void InitializeBlock(mlir::Block* block);

  // Since our analysis algorithm is based on MLIR's dataflow analysis, we need
  // to set up the dependency information between basic blocks so that the
  // fixpoint algorithm works.
  // Different analyses may have different strategies. For instance, conditional
  // forward analysis requires to mark whether each successor basic block is
  // executable and selectively add executable basic blocks as successors. Here,
  // we provide a vanilla (unconditional) dependency initialization that
  // provides all successors as dependencies.
  // This method is called inside `InitializeBlock`.
  virtual void InitializeBlockDependencies(mlir::Block* block);

  virtual void VisitBlock(mlir::Block* block);

  // This method mainly serves to "join" states from blocks. i.e., this method
  // should implement the "join" operation in a dataflow analysis. It should
  // join the states from the end of the predecessor into the entry of the
  // successor for a forward analysis, or join the states from the entry of a
  // block to the end of the predecessor for a backward analysis.
  virtual void VisitCfgEdge(JsirGeneralCfgEdge* edge);

  // Gets the state at the program point.
  template <typename T>
  JsirStateRef<T> GetStateImpl(mlir::LatticeAnchor anchor);

  // Helper function to get the `StateRef`s for the operands and results of an
  // op. For forward analysis, the input should be the operands and the output
  // should be the results. For backward analysis, the input should be the
  // results and the output should be the operands.
  struct ValueStateRefs {
    std::vector<const ValueT*> inputs;
    std::vector<JsirStateRef<ValueT>> outputs;
  };
  ValueStateRefs GetValueStateRefs(mlir::Operation* op);

  llvm::SmallVector<mlir::ProgramPoint*> Before(mlir::Operation* op) {
    return {getProgramPointBefore(op)};
  }

  llvm::SmallVector<mlir::ProgramPoint*> After(mlir::Operation* op) {
    return {getProgramPointAfter(op)};
  }

  llvm::SmallVector<mlir::ProgramPoint*> Before(mlir::Block* block) {
    return {getProgramPointBefore(block)};
  }

  llvm::SmallVector<mlir::ProgramPoint*> After(mlir::Block* block) {
    return {getProgramPointAfter(block)};
  }

  llvm::SmallVector<mlir::ProgramPoint*> Before(mlir::Region& region) {
    CHECK(!region.empty());
    return {getProgramPointBefore(&region.front())};
  }

  llvm::SmallVector<mlir::ProgramPoint*> After(mlir::Region& region) {
    llvm::SmallVector<mlir::ProgramPoint*> after_points;
    for (mlir::Block& block : region) {
      if (block.getSuccessors().empty()) {
        after_points.push_back(getProgramPointAfter(&block));
      }
    }
    return after_points;
  }

  LivenessInfo LiveIfTruthyOrUnknown(mlir::Value value) {
    return {
        .kind = LivenessKind::kLiveIfTruthyOrUnknown,
        .values = {value},
    };
  }

  LivenessInfo LiveIfFalsyOrUnknown(mlir::Value value) {
    return {
        .kind = LivenessKind::kLiveIfFalsyOrUnknown,
        .values = {value},
    };
  }

  LivenessInfo LiveIfEqualOrUnknown(
      mlir::Value lhs, llvm::PointerUnion<mlir::Value, mlir::Attribute> rhs) {
    return {
        .kind = LivenessKind::kLiveIfEqualOrUnknown,
        .values = {lhs, rhs},
    };
  }

  LivenessInfo LiveIfNotEqualOrUnknown(
      mlir::Value lhs, llvm::PointerUnion<mlir::Value, mlir::Attribute> rhs) {
    return {
        .kind = LivenessKind::kLiveIfNotEqualOrUnknown,
        .values = {lhs, rhs},
    };
  }

  static mlir::ValueRange GetExprRegionEndValues(mlir::Block* block) {
    auto term_op = block->getTerminator();
    if (auto expr_region_end_op =
            llvm::dyn_cast<JsirExprRegionEndOp>(term_op)) {
      return expr_region_end_op->getOperands();
    }
    return {};
  }

  static mlir::ValueRange GetExprRegionEndValuesFromRegion(
      mlir::Region& region) {
    for (auto& block : region.getBlocks()) {
      if (block.hasNoSuccessors()) {
        auto end_values = GetExprRegionEndValues(&block);
        if (!end_values.empty()) {
          return end_values;
        }
      }
    }
    return {};
  }

  static mlir::Region& GetForStatementContinueTargetRegion(
      JshirForStatementOp for_stmt) {
    if (!for_stmt.getUpdate().empty()) {
      return for_stmt.getUpdate();
    }
    if (!for_stmt.getTest().empty()) {
      return for_stmt.getTest();
    }
    return for_stmt.getBody();
  }

 private:
  // TODO(b/425421947) Could this be not a member variable?
  JumpEnv jump_env_;

  std::optional<decltype(jump_env_.WithJumpTargets({}))> WithJumpTargets(
      std::optional<JumpTargets> maybe_jump_targets) {
    if (maybe_jump_targets.has_value()) {
      return jump_env_.WithJumpTargets(maybe_jump_targets.value());
    }
    return std::nullopt;
  }

  std::optional<decltype(jump_env_.WithLabel({}))> WithLabel(
      mlir::Operation* op) {
    if (auto labeled_stmt = llvm::dyn_cast<JshirLabeledStatementOp>(op);
        labeled_stmt != nullptr) {
      return jump_env_.WithLabel(labeled_stmt.getLabel().getName());
    }
    return std::nullopt;
  }

  mlir::LogicalResult initialize(mlir::Operation* op) override;

  void VisitOp(mlir::Operation* op, const StateT* input,
               JsirStateRef<StateT> output);

  virtual void VisitOp(mlir::Operation* op);

  mlir::DataFlowSolver& solver_;

  // Override `mlir::DataFlowAnalysis::visit` and redirect to `Visit{Op,Block}`.
  mlir::LogicalResult visit(mlir::ProgramPoint* point) override;
};

template <typename ValueT, typename StateT>
using JsirForwardDataFlowAnalysis =
    JsirDataFlowAnalysis<ValueT, StateT, DataflowDirection::kForward>;

template <typename ValueT, typename StateT>
using JsirBackwardDataFlowAnalysis =
    JsirDataFlowAnalysis<ValueT, StateT, DataflowDirection::kBackward>;

// =============================================================================
// JsirStateRef
// =============================================================================

template <typename T>
void JsirStateRef<T>::AddDependent(mlir::ProgramPoint* point) {
  element_->addDependency(point, analysis_);
}

template <typename T>
void JsirStateRef<T>::Write(
    absl::FunctionRef<mlir::ChangeResult(T*)> write_fn) {
  mlir::ChangeResult changed = write_fn(&element_->value_);
  solver_->propagateIfChanged(element_, changed);
}

template <typename T>
void JsirStateRef<T>::Write(T&& lattice) {
  if (element_->value_ == lattice) {
    return;
  }

  element_->value_ = std::move(lattice);
  solver_->propagateIfChanged(element_, mlir::ChangeResult::Change);
}

template <typename T>
void JsirStateRef<T>::Write(const T& lattice) {
  T lattice_copy = lattice;
  Write(std::move(lattice_copy));
}

template <typename T>
void JsirStateRef<T>::Join(const T& lattice) {
  mlir::ChangeResult changed = element_->value_.Join(lattice);
  solver_->propagateIfChanged(element_, changed);
}

// =============================================================================
// JsirDataFlowAnalysis
// =============================================================================

template <typename ValueT, typename StateT, DataflowDirection direction>
template <typename T>
JsirStateRef<T> JsirDataFlowAnalysis<ValueT, StateT, direction>::GetStateImpl(
    mlir::LatticeAnchor anchor) {
  auto* element =
      mlir::DataFlowAnalysis::getOrCreate<detail::JsirStateElement<T>>(anchor);
  return JsirStateRef<T>{element, &solver_, this};
}

template <typename ValueT, typename StateT, DataflowDirection direction>
JsirStateRef<StateT>
JsirDataFlowAnalysis<ValueT, StateT, direction>::GetStateBefore(
    mlir::Operation* op) {
  if (auto* prev_op = op->getPrevNode()) {
    return GetStateAfter(prev_op);
  } else {
    return GetStateImpl<StateT>(getProgramPointBefore(op->getBlock()));
  }
}

template <typename ValueT, typename StateT, DataflowDirection direction>
JsirStateRef<StateT>
JsirDataFlowAnalysis<ValueT, StateT, direction>::GetStateAfter(
    mlir::Operation* op) {
  return GetStateImpl<StateT>(getProgramPointAfter(op));
}

template <typename ValueT, typename StateT, DataflowDirection direction>
JsirStateRef<StateT>
JsirDataFlowAnalysis<ValueT, StateT, direction>::GetStateAtEntryOf(
    mlir::Block* block) {
  return GetStateImpl<StateT>(getProgramPointBefore(block));
}

template <typename ValueT, typename StateT, DataflowDirection direction>
JsirStateRef<StateT>
JsirDataFlowAnalysis<ValueT, StateT, direction>::GetStateAtEndOf(
    mlir::Block* block) {
  if (block->empty()) {
    return GetStateAtEntryOf(block);
  } else {
    return GetStateAfter(&block->back());
  }
}

template <typename ValueT, typename StateT, DataflowDirection direction>
void JsirDataFlowAnalysis<ValueT, StateT, direction>::PrintOp(
    mlir::Operation* op, size_t num_indents, mlir::AsmState& asm_state,
    llvm::raw_ostream& os) {
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

template <typename ValueT, typename StateT, DataflowDirection direction>
void JsirDataFlowAnalysis<ValueT, StateT, direction>::PrintRegion(
    mlir::Region& region, size_t num_indents, mlir::AsmState& asm_state,
    llvm::raw_ostream& os) {
  os << "{\n";
  {
    llvm::SaveAndRestore<size_t> num_indents_in_region{num_indents,
                                                       num_indents + 2};

    for (mlir::Block& block : region.getBlocks()) {
      os.indent(num_indents);
      block.printAsOperand(os, asm_state);
      os << ":\n";

      llvm::SaveAndRestore<size_t> num_indents_in_block{num_indents,
                                                        num_indents + 2};

      PrintAtBlockEntry(block, num_indents, os);

      for (mlir::Operation& op : block) {
        os.indent(num_indents);
        PrintOp(&op, num_indents, asm_state, os);
        os << "\n";
      }
    }
  }
  os.indent(num_indents);
  os << "}";
}

template <typename ValueT, typename StateT, DataflowDirection direction>
bool JsirDataFlowAnalysis<ValueT, StateT, direction>::IsEntryBlock(
    mlir::Block* block) {
  mlir::Operation* parent_op = block->getParentOp();

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
template <typename ValueT, typename StateT, DataflowDirection direction>
mlir::LogicalResult JsirDataFlowAnalysis<ValueT, StateT, direction>::initialize(
    mlir::Operation* op) {
  // The op depends on its input operands.
  for (mlir::Value operand : op->getOperands()) {
    JsirStateRef<ValueT> operand_state_ref = GetStateAt(operand);
    operand_state_ref.AddDependent(getProgramPointAfter(op));
  }

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

  std::optional<JumpTargets> maybe_jump_targets;

  if (auto branch = llvm::dyn_cast<mlir::cf::BranchOp>(op); branch != nullptr) {
    MaybeEmplaceCfgEdges({
        .from = After(branch),
        .to = Before(branch.getDest()),
        .owner = branch.getDest(),
        .pred_values = branch.getDestOperands(),
        .succ_values = branch.getDest()->getArguments(),
    });
  }

  if (auto cond_branch = llvm::dyn_cast<mlir::cf::CondBranchOp>(op);
      cond_branch != nullptr) {
    MaybeEmplaceCfgEdges({
        .from = After(cond_branch),
        .to = Before(cond_branch.getTrueDest()),
        .owner = cond_branch.getTrueDest(),
        .liveness_info = LiveIfTruthyOrUnknown(cond_branch.getCondition()),
        .pred_values = cond_branch.getTrueDestOperands(),
        .succ_values = cond_branch.getTrueDest()->getArguments(),
    });

    MaybeEmplaceCfgEdges({
        .from = After(cond_branch),
        .to = Before(cond_branch.getFalseDest()),
        .owner = cond_branch.getFalseDest(),
        .liveness_info = LiveIfFalsyOrUnknown(cond_branch.getCondition()),
        .pred_values = cond_branch.getFalseDestOperands(),
        .succ_values = cond_branch.getFalseDest()->getArguments(),
    });
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
      llvm::isa<JsirClassDeclarationOp>(op) /* TODO Should this be here? */
      || llvm::isa<JsirClassExpressionOp>(op) ||
      llvm::isa<JsirExportNamedDeclarationOp>(op)) {
    if (llvm::isa<JshirWithStatementOp>(op)) {
      maybe_jump_targets = {
          .labeled_break_target = getProgramPointAfter(op),
          .unlabeled_break_target = std::nullopt,
          .continue_target = std::nullopt,
      };
    }
    if (!op->getRegion(0).empty()) {
      MaybeEmplaceCfgEdges({
          .from = Before(op),
          .to = Before(op->getRegion(0)),
          .owner = &*op,
      });
      MaybeEmplaceCfgEdges({
          .from = After(op->getRegion(0)),
          .to = After(op),
          .owner = &*op,
      });
    } else {
      MaybeEmplaceCfgEdges({
          .from = Before(op),
          .to = After(op),
          .owner = &*op,
      });
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
    maybe_jump_targets = {
        .labeled_break_target = getProgramPointAfter(if_stmt),
        .unlabeled_break_target = std::nullopt,
        .continue_target = std::nullopt,
    };
    MaybeEmplaceCfgEdges({
        .from = Before(if_stmt),
        .to = Before(if_stmt.getConsequent()),
        .owner = &*if_stmt,
        .liveness_info = LiveIfTruthyOrUnknown(if_stmt.getTest()),
    });
    MaybeEmplaceCfgEdges({
        .from = After(if_stmt.getConsequent()),
        .to = After(if_stmt),
        .owner = &*if_stmt,
    });

    if (!if_stmt.getAlternate().empty()) {
      MaybeEmplaceCfgEdges({
          .from = Before(if_stmt),
          .to = Before(if_stmt.getAlternate()),
          .owner = &*if_stmt,
          .liveness_info = LiveIfFalsyOrUnknown(if_stmt.getTest()),
      });
      MaybeEmplaceCfgEdges({
          .from = After(if_stmt.getAlternate()),
          .to = After(if_stmt),
          .owner = &*if_stmt,
      });
    } else {
      MaybeEmplaceCfgEdges({
          .from = Before(if_stmt),
          .to = After(if_stmt),
          .owner = &*if_stmt,
          .liveness_info = LiveIfFalsyOrUnknown(if_stmt.getTest()),
      });
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
    MaybeEmplaceCfgEdges({
        .from = Before(block_stmt),
        .to = Before(block_stmt.getDirectives()),
        .owner = &*block_stmt,
    });
    MaybeEmplaceCfgEdges({
        .from = After(block_stmt.getDirectives()),
        .to = Before(block_stmt.getBody()),
        .owner = &*block_stmt,
    });
    MaybeEmplaceCfgEdges({
        .from = After(block_stmt.getBody()),
        .to = After(block_stmt),
        .owner = &*block_stmt,
    });
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
    maybe_jump_targets = {
        .labeled_break_target = getProgramPointAfter(while_stmt),
        .unlabeled_break_target = getProgramPointAfter(while_stmt),
        .continue_target = getProgramPointBefore(&while_stmt.getTest().front()),
    };
    MaybeEmplaceCfgEdges({
        .from = Before(while_stmt),
        .to = Before(while_stmt.getTest()),
        .owner = &*while_stmt,
    });
    MaybeEmplaceCfgEdges({
        .from = After(while_stmt.getTest()),
        .to = Before(while_stmt.getBody()),
        .owner = &*while_stmt,
        .liveness_info = LiveIfTruthyOrUnknown(
            GetExprRegionEndValuesFromRegion(while_stmt.getTest())[0]),
    });
    MaybeEmplaceCfgEdges({
        .from = After(while_stmt.getBody()),
        .to = Before(while_stmt.getTest()),
        .owner = &*while_stmt,
    });
    MaybeEmplaceCfgEdges({
        .from = After(while_stmt.getTest()),
        .to = After(while_stmt),
        .owner = &*while_stmt,
        .liveness_info = LiveIfFalsyOrUnknown(
            GetExprRegionEndValuesFromRegion(while_stmt.getTest())[0]),
    });
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
    maybe_jump_targets = {
        .labeled_break_target = getProgramPointAfter(do_while_stmt),
        .unlabeled_break_target = getProgramPointAfter(do_while_stmt),
        .continue_target =
            getProgramPointBefore(&do_while_stmt.getTest().front()),
    };
    MaybeEmplaceCfgEdges({
        .from = Before(do_while_stmt),
        .to = Before(do_while_stmt.getBody()),
        .owner = &*do_while_stmt,
    });
    MaybeEmplaceCfgEdges({
        .from = After(do_while_stmt.getBody()),
        .to = Before(do_while_stmt.getTest()),
        .owner = &*do_while_stmt,
    });
    MaybeEmplaceCfgEdges({
        .from = After(do_while_stmt.getTest()),
        .to = Before(do_while_stmt.getBody()),
        .owner = &*do_while_stmt,
        .liveness_info = LiveIfTruthyOrUnknown(
            GetExprRegionEndValuesFromRegion(do_while_stmt.getTest())[0]),
    });
    MaybeEmplaceCfgEdges({
        .from = After(do_while_stmt.getTest()),
        .to = After(do_while_stmt),
        .owner = &*do_while_stmt,
        .liveness_info = LiveIfFalsyOrUnknown(
            GetExprRegionEndValuesFromRegion(do_while_stmt.getTest())[0]),
    });
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
    maybe_jump_targets = {
        .labeled_break_target = getProgramPointAfter(for_stmt),
        .unlabeled_break_target = getProgramPointAfter(for_stmt),
        .continue_target = getProgramPointBefore(
            &GetForStatementContinueTargetRegion(for_stmt).front()),
    };
    // Emplace an edge into the first non-empty region of the for-statement.
    mlir::Region& first_region =
        !for_stmt.getInit().empty()
            ? for_stmt.getInit()
            : (!for_stmt.getTest().empty() ? for_stmt.getTest()
                                           : for_stmt.getBody());
    MaybeEmplaceCfgEdges({
        .from = Before(for_stmt),
        .to = Before(first_region),
        .owner = &*for_stmt,
    });

    if (!for_stmt.getInit().empty()) {
      mlir::Region& successor =
          !for_stmt.getTest().empty() ? for_stmt.getTest() : for_stmt.getBody();

      MaybeEmplaceCfgEdges({
          .from = After(for_stmt.getInit()),
          .to = Before(successor),
          .owner = &*for_stmt,
      });
    }

    if (!for_stmt.getTest().empty()) {
      MaybeEmplaceCfgEdges({
          .from = After(for_stmt.getTest()),
          .to = Before(for_stmt.getBody()),
          .owner = &*for_stmt,
          .liveness_info = LiveIfTruthyOrUnknown(
              GetExprRegionEndValuesFromRegion(for_stmt.getTest())[0]),
      });
      MaybeEmplaceCfgEdges({
          .from = After(for_stmt.getTest()),
          .to = After(for_stmt),
          .owner = &*for_stmt,
          .liveness_info = LiveIfFalsyOrUnknown(
              GetExprRegionEndValuesFromRegion(for_stmt.getTest())[0]),
      });
    }

    {
      MaybeEmplaceCfgEdges({
          .from = After(for_stmt.getBody()),
          .to = Before(GetForStatementContinueTargetRegion(for_stmt)),
          .owner = &*for_stmt,
      });
    }

    if (!for_stmt.getUpdate().empty()) {
      mlir::Region& successor =
          !for_stmt.getTest().empty() ? for_stmt.getTest() : for_stmt.getBody();

      MaybeEmplaceCfgEdges({
          .from = After(for_stmt.getUpdate()),
          .to = Before(successor),
          .owner = &*for_stmt,
      });
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
    maybe_jump_targets = {
        .labeled_break_target = getProgramPointAfter(for_in_stmt),
        .unlabeled_break_target = getProgramPointAfter(for_in_stmt),
        .continue_target =
            getProgramPointBefore(&for_in_stmt.getBody().front()),
    };
    MaybeEmplaceCfgEdges({
        .from = Before(for_in_stmt),
        .to = Before(for_in_stmt.getBody()),
        .owner = &*for_in_stmt,
    });
    MaybeEmplaceCfgEdges({
        .from = After(for_in_stmt.getBody()),
        .to = Before(for_in_stmt.getBody()),
        .owner = &*for_in_stmt,
    });
    MaybeEmplaceCfgEdges({
        .from = After(for_in_stmt.getBody()),
        .to = After(for_in_stmt),
        .owner = &*for_in_stmt,
    });
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
    maybe_jump_targets = {
        .labeled_break_target = getProgramPointAfter(for_of_stmt),
        .unlabeled_break_target = getProgramPointAfter(for_of_stmt),
        .continue_target =
            getProgramPointBefore(&for_of_stmt.getBody().front()),
    };
    MaybeEmplaceCfgEdges({
        .from = Before(for_of_stmt),
        .to = Before(for_of_stmt.getBody()),
        .owner = &*for_of_stmt,
    });
    MaybeEmplaceCfgEdges({
        .from = After(for_of_stmt.getBody()),
        .to = Before(for_of_stmt.getBody()),
        .owner = &*for_of_stmt,
    });
    MaybeEmplaceCfgEdges({
        .from = After(for_of_stmt.getBody()),
        .to = After(for_of_stmt),
        .owner = &*for_of_stmt,
    });
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
    mlir::Attribute comparison_attr;
    switch (*StringToJsLogicalOperator(logical_expr.getOperator_())) {
      case JsLogicalOperator::kAnd:
        // left && right => left ? right : left
        comparison_attr = mlir::BoolAttr::get(logical_expr.getContext(), true);
        break;
      case JsLogicalOperator::kOr:
        // left || right => left ? left : right
        comparison_attr = mlir::BoolAttr::get(logical_expr.getContext(), false);
        break;
      case JsLogicalOperator::kNullishCoalesce:
        // left ?? right => (left == null) ? right : left
        comparison_attr = JsirNullLiteralAttr::get(logical_expr.getContext());
        break;
    }

    mlir::Value left_value = logical_expr.getLeft();

    MaybeEmplaceCfgEdges({
        .from = Before(logical_expr),
        .to = After(logical_expr),
        .owner = &*logical_expr,
        .liveness_info = LiveIfNotEqualOrUnknown(left_value, comparison_attr),
        .pred_values = mlir::ValueRange{left_value},
        .succ_values = logical_expr->getResults(),
    });
    MaybeEmplaceCfgEdges({
        .from = Before(logical_expr),
        .to = Before(logical_expr.getRight()),
        .owner = &*logical_expr,
        .liveness_info = LiveIfEqualOrUnknown(left_value, comparison_attr),
    });
    MaybeEmplaceCfgEdges({
        .from = After(logical_expr.getRight()),
        .to = After(logical_expr),
        .owner = &*logical_expr,
        .pred_values = GetExprRegionEndValues,
        .succ_values = logical_expr->getResults(),
    });
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
    MaybeEmplaceCfgEdges({
        .from = Before(conditional_expr),
        .to = Before(conditional_expr.getConsequent()),
        .owner = &*conditional_expr,
        .liveness_info = LiveIfTruthyOrUnknown(conditional_expr.getTest()),
    });
    MaybeEmplaceCfgEdges({
        .from = Before(conditional_expr),
        .to = Before(conditional_expr.getAlternate()),
        .owner = &*conditional_expr,
        .liveness_info = LiveIfFalsyOrUnknown(conditional_expr.getTest()),
    });
    MaybeEmplaceCfgEdges({
        .from = After(conditional_expr.getConsequent()),
        .to = After(conditional_expr),
        .owner = &*conditional_expr,
        .pred_values = GetExprRegionEndValues,
        .succ_values = conditional_expr->getResults(),
    });
    MaybeEmplaceCfgEdges({
        .from = After(conditional_expr.getAlternate()),
        .to = After(conditional_expr),
        .owner = &*conditional_expr,
        .pred_values = GetExprRegionEndValues,
        .succ_values = conditional_expr->getResults(),
    });
  }

  if (auto break_stmt = llvm::dyn_cast<JshirBreakStatementOp>(op);
      break_stmt != nullptr) {
    absl::StatusOr<mlir::ProgramPoint*> break_target;

    JsirIdentifierAttr label = break_stmt.getLabelAttr();
    if (label == nullptr) {
      break_target = jump_env_.break_target();
    } else {
      break_target = jump_env_.break_target(label.getName());
    }

    if (break_target.ok()) {
      MaybeEmplaceCfgEdges({
          .from = Before(break_stmt),
          .to = {break_target.value()},
          .owner = &*break_stmt,
      });
    }
  }

  if (auto continue_stmt = llvm::dyn_cast<JshirContinueStatementOp>(op);
      continue_stmt != nullptr) {
    absl::StatusOr<mlir::ProgramPoint*> continue_target;

    JsirIdentifierAttr label = continue_stmt.getLabelAttr();
    if (label == nullptr) {
      continue_target = jump_env_.continue_target();
    } else {
      continue_target = jump_env_.continue_target(label.getName());
    }

    if (continue_target.ok()) {
      MaybeEmplaceCfgEdges({
          .from = Before(continue_stmt),
          .to = {continue_target.value()},
          .owner = &*continue_stmt,
      });
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
    MaybeEmplaceCfgEdges({
        .from = Before(try_stmt),
        .to = Before(try_stmt.getBlock()),
        .owner = &*try_stmt,
    });
    if (!try_stmt.getFinalizer().empty()) {
      MaybeEmplaceCfgEdges({
          .from = After(try_stmt.getBlock()),
          .to = Before(try_stmt.getFinalizer()),
          .owner = &*try_stmt,
      });
      MaybeEmplaceCfgEdges({
          .from = After(try_stmt.getFinalizer()),
          .to = After(try_stmt),
          .owner = &*try_stmt,
      });
    } else {
      MaybeEmplaceCfgEdges({
          .from = After(try_stmt.getBlock()),
          .to = After(try_stmt),
          .owner = &*try_stmt,
      });
    }
  }

  // ┌─────◄
  // │     jshir.switch_statement (
  // └─────► ┌───────────────┐
  //         │ cases region  │
  // ┌─────◄ └───────────────┘
  // │     );
  // └─────►
  if (auto switch_stmt = llvm::dyn_cast<JshirSwitchStatementOp>(op);
      switch_stmt != nullptr) {
    maybe_jump_targets = {
        .labeled_break_target = getProgramPointAfter(switch_stmt),
        .unlabeled_break_target = getProgramPointAfter(switch_stmt),
        .continue_target = std::nullopt,
    };
    MaybeEmplaceCfgEdges({
        .from = Before(switch_stmt),
        .to = Before(switch_stmt.getCases()),
        .owner = &*switch_stmt,
    });
    MaybeEmplaceCfgEdges({
        .from = After(switch_stmt.getCases()),
        .to = After(switch_stmt),
        .owner = &*switch_stmt,
    });
  }

  // ┌─────◄
  // │     jshir.switch_case (
  // ├─────► ┌───────────────┐
  // │       │ test region   │
  // │  ┌──◄ └───────────────┘
  // └──┴──► ┌───────────────┐
  //         │ consequent    │
  // ┌─────◄ └───────────────┘
  // │     );
  // └─────►
  if (auto switch_case = llvm::dyn_cast<JshirSwitchCaseOp>(op);
      switch_case != nullptr) {
    if (switch_case.getTest().empty()) {
      MaybeEmplaceCfgEdges({
          .from = Before(switch_case),
          .to = Before(switch_case.getConsequent()),
          .owner = &*switch_case,
      });
    } else {
      MaybeEmplaceCfgEdges({
          .from = Before(switch_case),
          .to = Before(switch_case.getTest()),
          .owner = &*switch_case,
      });
      MaybeEmplaceCfgEdges({
          .from = After(switch_case.getTest()),
          .to = Before(switch_case.getConsequent()),
          .owner = &*switch_case,
          .liveness_info = LiveIfEqualOrUnknown(
              switch_case->getParentOfType<JshirSwitchStatementOp>()
                  .getDiscriminant(),
              GetExprRegionEndValuesFromRegion(switch_case.getTest())[0]),
      });

      MaybeEmplaceCfgEdges({
          .from = After(switch_case.getTest()),
          .to = After(switch_case),
          .owner = &*switch_case,
          .liveness_info = LiveIfNotEqualOrUnknown(
              switch_case->getParentOfType<JshirSwitchStatementOp>()
                  .getDiscriminant(),
              GetExprRegionEndValuesFromRegion(switch_case.getTest())[0]),
      });
    }

    // If this is not the last case, we need fall-through to the next case.
    if (auto* next_node = switch_case->getNextNode(); next_node != nullptr) {
      if (auto successor_case = llvm::dyn_cast<JshirSwitchCaseOp>(next_node);
          successor_case != nullptr) {
        MaybeEmplaceCfgEdges({
            .from = After(switch_case.getConsequent()),
            .to = Before(successor_case.getConsequent()),
            .owner = &*switch_case,
        });
      }
    } else {
      MaybeEmplaceCfgEdges({
          .from = After(switch_case.getConsequent()),
          .to = After(switch_case),
          .owner = &*switch_case,
      });
    }
  }

  // Get optional jump targets and label to be used during recursive
  // initialization. These variables use RAII.
  auto with_jump_targets = WithJumpTargets(maybe_jump_targets);
  auto with_label = WithLabel(op);

  // Recursively initialize.
  for (mlir::Region& region : op->getRegions()) {
    for (mlir::Block& block : region.getBlocks()) {
      InitializeBlock(&block);
    }
  }

  return mlir::success();
}

template <typename ValueT, typename StateT, DataflowDirection direction>
void JsirDataFlowAnalysis<ValueT, StateT, direction>::InitializeBlock(
    mlir::Block* block) {
  // Initialize all inner ops.
  for (mlir::Operation& op : *block) {
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
    auto is_exit_block = [](mlir::Block* b) {
      // Treat empty and terminator-less blocks as exit blocks.
      if (b->empty() ||
          !b->back().mightHaveTrait<mlir::OpTrait::IsTerminator>())
        return true;

      // There may be a weird case where a terminator may be transferring
      // control either to the parent or to another block, so exit blocks and
      // successors are not mutually exclusive.
      mlir::Operation* terminator = b->getTerminator();
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

template <typename ValueT, typename StateT, DataflowDirection direction>
void JsirDataFlowAnalysis<ValueT, StateT, direction>::
    InitializeBlockDependencies(mlir::Block* block) {
  if constexpr (direction == DataflowDirection::kForward) {
    // For each block, we should update its successor blocks when the state
    // at the end of the block updates. Thus, we enumerate each predecessor's
    // end state and link it to the block.
    for (mlir::Block* pred : block->getPredecessors()) {
      JsirStateRef<StateT> pred_state_ref = GetStateAtEndOf(pred);
      pred_state_ref.AddDependent(getProgramPointBefore(block));
    }
  } else if constexpr (direction == DataflowDirection::kBackward) {
    // For each block, we should update its predecessor blocks when the state
    // at the end of the block updates. Thus, we enumerate each successor's
    // end state and link it to the block.
    for (mlir::Block* succ : block->getSuccessors()) {
      JsirStateRef<StateT> succ_state_ref = GetStateAtEntryOf(succ);
      succ_state_ref.AddDependent(getProgramPointBefore(block));
    }
  }
}

template <typename ValueT, typename StateT, DataflowDirection direction>
mlir::LogicalResult JsirDataFlowAnalysis<ValueT, StateT, direction>::visit(
    mlir::ProgramPoint* point) {
  if (!point->isBlockStart()) {
    VisitOp(point->getPrevOp());
  } else if (!point->isNull()) {
    VisitBlock(point->getBlock());
  }
  return mlir::success();
}

template <typename ValueT, typename StateT, DataflowDirection direction>
void JsirDataFlowAnalysis<ValueT, StateT, direction>::VisitOp(
    mlir::Operation* op) {
  if constexpr (direction == DataflowDirection::kForward) {
    JsirStateRef<StateT> before_state_ref = GetStateBefore(op);
    const StateT* before = &before_state_ref.value();

    JsirStateRef after_state_ref = GetStateAfter(op);

    VisitOp(op, before, after_state_ref);
  } else if constexpr (direction == DataflowDirection::kBackward) {
    JsirStateRef<StateT> after_state_ref = GetStateAfter(op);
    const StateT* after = &after_state_ref.value();

    JsirStateRef before_state_ref = GetStateBefore(op);

    VisitOp(op, after, before_state_ref);
  }
}

template <typename ValueT, typename StateT, DataflowDirection direction>
void JsirDataFlowAnalysis<ValueT, StateT, direction>::VisitBlock(
    mlir::Block* block) {
  for (auto* edge : block_to_cfg_edges_[block]) {
    VisitCfgEdge(edge);
  }
}

template <typename ValueT, typename StateT, DataflowDirection direction>
void JsirDataFlowAnalysis<ValueT, StateT, direction>::PrintAtBlockEntry(
    mlir::Block& block, size_t num_indents, llvm::raw_ostream& os) {
  os.indent(num_indents + 2);
  os << "// ";
  GetStateAtEntryOf(&block).value().print(os);
  os << "\n";
}

template <typename ValueT, typename StateT, DataflowDirection direction>
JsirStateRef<ValueT>
JsirDataFlowAnalysis<ValueT, StateT, direction>::GetStateAt(mlir::Value value) {
  return GetStateImpl<ValueT>(value);
}

template <typename ValueT, typename StateT, DataflowDirection direction>
void JsirDataFlowAnalysis<ValueT, StateT, direction>::PrintAfterOp(
    mlir::Operation* op, size_t num_indents, mlir::AsmState& asm_state,
    llvm::raw_ostream& os) {
  for (mlir::Value result : op->getResults()) {
    auto result_state_ref = GetStateAt(result);

    os << "\n";
    os.indent(num_indents + 2);
    os << "// ";
    result.printAsOperand(os, asm_state);
    os << " = ";
    result_state_ref.value().print(os);
  }

  os << "\n";
  os.indent(num_indents + 2);
  os << "// ";
  GetStateAfter(op).value().print(os);
}

template <typename ValueT, typename StateT, DataflowDirection direction>
typename JsirDataFlowAnalysis<ValueT, StateT, direction>::ValueStateRefs
JsirDataFlowAnalysis<ValueT, StateT, direction>::GetValueStateRefs(
    mlir::Operation* op) {
  if constexpr (direction == DataflowDirection::kForward) {
    std::vector<const ValueT*> operands;
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
    std::vector<const ValueT*> results;
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
    mlir::Operation* op, const StateT* input, JsirStateRef<StateT> output) {
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
    JsirGeneralCfgEdge* edge) {
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

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_ANALYSES_DATAFLOW_ANALYSIS_H_
