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

#ifndef MALDOCA_JS_IR_ANALYSES_JUMP_ENV_H_
#define MALDOCA_JS_IR_ANALYSES_JUMP_ENV_H_

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace maldoca {

struct JumpTargets {
  mlir::ProgramPoint *labeled_break_target;
  std::optional<mlir::ProgramPoint *> unlabeled_break_target;
  std::optional<mlir::ProgramPoint *> continue_target;
};

class JumpEnvScope {
 public:
  auto WithLabel(mlir::StringAttr label) {
    unmatched_labels_.insert(label);

    return absl::MakeCleanup([=, this] { unmatched_labels_.erase(label); });
  }

  auto WithJumpTargets(JumpTargets targets) {
    targets_stack_.push_back(UnlabeledJumpTargets{
        .break_target = targets.unlabeled_break_target,
        .continue_target = targets.continue_target,
    });

    auto labels = std::move(unmatched_labels_);
    for (const auto &label : labels) {
      matched_labels_.insert({
          label,
          LabeledJumpTargets{
              .break_target = targets.labeled_break_target,
              .continue_target = targets.continue_target,
          },
      });
    }

    return absl::MakeCleanup([this, labels = std::move(labels)] {
      targets_stack_.pop_back();

      unmatched_labels_ = std::move(labels);
      for (const auto &label : unmatched_labels_) {
        matched_labels_.erase(label);
      }
    });
  }

  const llvm::DenseSet<mlir::StringAttr> &unmatched_labels() const {
    return unmatched_labels_;
  }

  absl::StatusOr<mlir::ProgramPoint *> break_target() const {
    for (const auto &info : llvm::reverse(targets_stack_)) {
      if (info.break_target.has_value()) {
        return info.break_target.value();
      }
    }
    return absl::NotFoundError("Cannot find unlabeled break target.");
  }

  absl::StatusOr<mlir::ProgramPoint *> break_target(
      mlir::StringAttr label) const {
    auto it = matched_labels_.find(label);
    if (it == matched_labels_.end()) {
      if (unmatched_labels_.contains(label)) {
        return absl::NotFoundError(
            "Label is not matched to a control flow structure.");
      }
      return absl::NotFoundError("Cannot find label.");
    }

    const LabeledJumpTargets &info = it->second;
    return info.break_target;
  }

  absl::StatusOr<mlir::ProgramPoint *> continue_target() const {
    for (const auto &info : llvm::reverse(targets_stack_)) {
      if (info.continue_target.has_value()) {
        return info.continue_target.value();
      }
    }
    return absl::NotFoundError("Cannot find unlabeled break target.");
  }

  absl::StatusOr<mlir::ProgramPoint *> continue_target(
      mlir::StringAttr label) const {
    auto it = matched_labels_.find(label);
    if (it == matched_labels_.end()) {
      if (unmatched_labels_.contains(label)) {
        return absl::NotFoundError(
            "Label is not matched to a control flow structure.");
      }
      return absl::NotFoundError("Cannot find label.");
    }

    LabeledJumpTargets info = it->second;
    if (!info.continue_target.has_value()) {
      return absl::NotFoundError(
          "Labeled statement does not support continue.");
    }
    return info.continue_target.value();
  }

 private:
  struct UnlabeledJumpTargets {
    std::optional<mlir::ProgramPoint *> break_target;
    std::optional<mlir::ProgramPoint *> continue_target;
  };

  struct LabeledJumpTargets {
    mlir::ProgramPoint *break_target;
    std::optional<mlir::ProgramPoint *> continue_target;
  };

  std::vector<UnlabeledJumpTargets> targets_stack_;
  llvm::DenseMap<mlir::StringAttr, LabeledJumpTargets> matched_labels_;
  llvm::DenseSet<mlir::StringAttr> unmatched_labels_;
};

class JumpEnv {
 public:
  explicit JumpEnv() {
    // Push the global scope.
    scopes_.push_back(std::make_unique<JumpEnvScope>());
  }

  JumpEnv(const JumpEnv &) = delete;
  JumpEnv &operator=(const JumpEnv &) = delete;

  // Adds a label to the current scope.
  // Returns an object that, on destruction, deletes the label.
  auto WithLabel(mlir::StringAttr label) {
    return scopes_.back()->WithLabel(label);
  }

  // Adds a control flow structure that defines jump targets.
  // Returns an object that, on destruction, deletes the jump targets.
  auto WithJumpTargets(JumpTargets info) {
    return scopes_.back()->WithJumpTargets(info);
  }

  // Adds a new scope of labels and jump targets.
  // The new scope hides all the existing labels and jump targets.
  // Returns an object that, on destruction, deletes the scope.
  auto WithScope() {
    scopes_.push_back(std::make_unique<JumpEnvScope>());
    return absl::MakeCleanup([this] { scopes_.pop_back(); });
  }

  // The current set of labels not matched to a statement.
  const llvm::DenseSet<mlir::StringAttr> &unmatched_labels() const {
    return scopes_.back()->unmatched_labels();
  }

  // The target of a "break;" statement.
  auto break_target() const { return scopes_.back()->break_target(); }

  // The target of a "break <label>;" statement.
  auto break_target(mlir::StringAttr label) const {
    return scopes_.back()->break_target(label);
  }

  // The target of a "continue;" statement.
  auto continue_target() const { return scopes_.back()->continue_target(); }

  // The target of a "continue <label>;" statement.
  auto continue_target(mlir::StringAttr label) const {
    return scopes_.back()->continue_target(label);
  }

 private:
  std::vector<std::unique_ptr<JumpEnvScope>> scopes_;
};

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_ANALYSES_JUMP_ENV_H_
