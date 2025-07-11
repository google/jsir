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

// A util library to keep track of labels and break/continue targets in JSLIR.
//
// A JavaScript control flow structure might allow nesting these kinds of jumps:
// - break;
// - break <label>;
// - continue;
// - continue <label>;
//
// Unlabeled Jumps
// ===============
//
// The target of a jump is determined by the inner-most control flow structure
// that allow that kind of jump.
//
// For example, in the code below, the `break;` statement jumps out of the
// `while` loop instead of the `if` statement, even though the `if` statement is
// nested inside the `while` loop. This is because `if` statements do not allow
// unlabeled breaks.
//
// ```
// while (...) {
//   ...
//   if (...) {
//     break; ---+
//   }           |
//   ...         |
// }             |
// ...  <--------+
// ```
//
// Labeled Jumps
// =============
//
// JavaScript labels are "structured". A label effectively annotates a statement
// (instead of a program point), and is only visible within that statement.
//
// For example, in the code below, `lbl` annotates the `if` statement, rather
// than representing the program point right before the `if` statement.
//
// ```
// lbl: if (...) {
//   ...
// } else {
//   ...
// }
// ```
//
// Moreover, since a label is only visible within the annotated statement, two
// labels with the same name are fine as long as one isn't nested within the
// other.
//
// ```This is legal.
// lbl: if (...) { ... }
// lbl: if (...) { ... }
// ```
//
// ```This is illegal.
// lbl: if (...) {
//   ...
//   lbl: if (...) { ... }
//   ...
// }
// ```
//
// However, in the code below, the `break lbl;` statement jumps out of the `if`
// statement. This is because `if` statements allow labeled breaks.
//
// ```
// while (...) {
//   ...
//   lbl: if (...) {
//     break lbl; ---+
//     ...           |
//   }               |
//   ...  <----------+
// }
// ...
// ```

#ifndef MALDOCA_JS_IR_CONVERSION_JSLIR_JUMP_ENV_H_
#define MALDOCA_JS_IR_CONVERSION_JSLIR_JUMP_ENV_H_

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "absl/cleanup/cleanup.h"
#include "absl/status/statusor.h"

namespace maldoca {

// Specifies the targets of different kinds of jumps. `nullopt` means this kind
// of jump is not allowed.
struct JslirJumpTargets {
  // The target of "break <label>;"
  mlir::Block *labeled_break_target;

  // The target of "break;"
  std::optional<mlir::Block *> unlabeled_break_target;

  // The target of "continue;" and "continue <label>;"
  //
  // Note that for a control flow statement, it either supports both or neither
  // of labeled and unlabeled continue statements.
  std::optional<mlir::Block *> continue_target;
};

namespace internal {

// A function-level scope of labels and jump targets.
//
// JavaScript doesn't allow jumping out of the current function.
// ```
// lbl:
// function foo() {
//   break lbl;  // Error: undefined label.
// }
// ```
class JslirJumpEnvScope {
 public:
  auto WithLabel(mlir::StringAttr label) {
    unmatched_labels_.insert(label);

    return absl::MakeCleanup([=] { unmatched_labels_.erase(label); });
  }

  auto WithJumpTargets(JslirJumpTargets targets) {
    targets_stack_.push_back(JslirUnlabeledJumpTargets{
        .break_target = targets.unlabeled_break_target,
        .continue_target = targets.continue_target,
    });

    auto labels = std::move(unmatched_labels_);
    for (const auto &label : labels) {
      matched_labels_.insert({
          label,
          JslirLabeledJumpTargets{
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

  absl::StatusOr<mlir::Block *> break_target() const;

  absl::StatusOr<mlir::Block *> break_target(mlir::StringAttr label) const;

  absl::StatusOr<mlir::Block *> continue_target() const;

  absl::StatusOr<mlir::Block *> continue_target(mlir::StringAttr label) const;

 private:
  struct JslirUnlabeledJumpTargets {
    std::optional<mlir::Block *> break_target;
    std::optional<mlir::Block *> continue_target;
  };

  struct JslirLabeledJumpTargets {
    mlir::Block *break_target;
    std::optional<mlir::Block *> continue_target;
  };

  std::vector<JslirUnlabeledJumpTargets> targets_stack_;
  llvm::DenseMap<mlir::StringAttr, JslirLabeledJumpTargets> matched_labels_;
  llvm::DenseSet<mlir::StringAttr> unmatched_labels_;
};

}  // namespace internal

class JslirJumpEnv {
 public:
  explicit JslirJumpEnv() {
    // Push the global scope.
    scopes_.push_back(std::make_unique<internal::JslirJumpEnvScope>());
  }

  JslirJumpEnv(const JslirJumpEnv &) = delete;
  JslirJumpEnv &operator=(const JslirJumpEnv &) = delete;

  // Adds a label to the current scope.
  // Returns an object that, on destruction, deletes the label.
  auto WithLabel(mlir::StringAttr label) {
    return scopes_.back()->WithLabel(label);
  }

  // Adds a control flow structure that defines jump targets.
  // Returns an object that, on destruction, deletes the jump targets.
  auto WithJumpTargets(JslirJumpTargets info) {
    return scopes_.back()->WithJumpTargets(info);
  }

  // Adds a new scope of labels and jump targets.
  // The new scope hides all the existing labels and jump targets.
  // Returns an object that, on destruction, deletes the scope.
  auto WithScope() {
    scopes_.push_back(std::make_unique<internal::JslirJumpEnvScope>());
    return absl::MakeCleanup([this] { scopes_.pop_back(); });
  }

  // The current set of labels not matched to a statement.
  const llvm::DenseSet<mlir::StringAttr> &unmatched_labels() const {
    return scopes_.back()->unmatched_labels();
  }

  // The target of a "break;" statement.
  absl::StatusOr<mlir::Block *> break_target() const {
    return scopes_.back()->break_target();
  }

  // The target of a "break <label>;" statement.
  absl::StatusOr<mlir::Block *> break_target(mlir::StringAttr label) const {
    return scopes_.back()->break_target(label);
  }

  // The target of a "continue;" statement.
  absl::StatusOr<mlir::Block *> continue_target() const {
    return scopes_.back()->continue_target();
  }

  // The target of a "continue <label>;" statement.
  absl::StatusOr<mlir::Block *> continue_target(mlir::StringAttr label) const {
    return scopes_.back()->continue_target(label);
  }

 private:
  std::vector<std::unique_ptr<internal::JslirJumpEnvScope>> scopes_;
};

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_CONVERSION_JSLIR_JUMP_ENV_H_
