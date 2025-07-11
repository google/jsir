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

#ifndef MALDOCA_JS_IR_ANALYSES_PER_VAR_STATE_H_
#define MALDOCA_JS_IR_ANALYSES_PER_VAR_STATE_H_

#include <utility>
#include <vector>

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/ir/analyses/scope.h"
#include "maldoca/js/ir/analyses/state.h"

namespace maldoca {

// A PerVarState is conceptually a map from variable names to `ValueT`s.
//
// - The PerVarState has a default value that we assign to all variables.
//   This means that, even if a variable does not exist in the map, Get() still
//   returns a valid value - the default value.
//
// - We provide a Join() API that conforms to the definition of a lattice.
//   Join() updates the current state to the result of joining it with another
//   state. The algorithm joins the states on each variable in both
//   PerVarStates, as well as the default values in both PerVarStates:
//
//   Join(
//       State {
//           default: dflt_1,
//           a: val_1
//       },
//       State {
//           default: dflt_2,
//           a: val_2
//       }
//   ) = State {
//       default: Join(dflt_1, dflt_2),
//       a: Join(val_1, val_2)
//   }
//
//   When the key sets are different in the two PerVarStates, the default
//   value(s) are used:
//
//   Join(
//       State {
//           default: dflt_1,
//           a: val_1
//       },
//       State {
//           default: dflt_2,
//           b: val_2
//       }
//   ) = State {
//       default: Join(dflt_1, dflt_2),
//       a: Join(val_1, dflt_2),
//       b: Join(dflt_1, val_2)
//   }
//
// Example:
// ```
// <--------- State: {default: Unknown}
// if (...)
//   <------- State: {default: Unknown}
//   a = 1;
//   <------- State: {a: 1, default: Unknown}
// else
//   <------- State: {default: Unknown}
//   a = 1;
//   <------- State: {a: 1, default: Unknown}
//   b = 2;
//   <------- State: {a: 1, b: 2, default: Unknown}
// <--------- State: {a: 1, b: Unknown, default: Unknown}
//            (Note: For b, Join(Unknown, 2) = Unknown.)
// ```
template <typename ValueT>
class JsirPerVarState : public absl::flat_hash_map<JsSymbolId, ValueT>,
                        public JsirState<JsirPerVarState<ValueT>> {
 public:
  static_assert(std::is_base_of_v<JsirState<ValueT>, ValueT>,
                "Must use the CRTP type JsirState. "
                "E.g. class MyState : public JsirState<MyState> {};");

  explicit JsirPerVarState(ValueT default_value)
      : default_value_(std::move(default_value)) {}

  explicit JsirPerVarState() : JsirPerVarState(ValueT()) {}

  using absl::flat_hash_map<JsSymbolId, ValueT>::find;
  using absl::flat_hash_map<JsSymbolId, ValueT>::begin;
  using absl::flat_hash_map<JsSymbolId, ValueT>::end;
  using absl::flat_hash_map<JsSymbolId, ValueT>::insert;
  using absl::flat_hash_map<JsSymbolId, ValueT>::erase;

  // If the var does not exist in the map, returns the default value.
  const ValueT &Get(const JsSymbolId &var) const;

  mlir::ChangeResult Set(
      const JsSymbolId &var,
      llvm::function_ref<mlir::ChangeResult(ValueT &)> setter);

  mlir::ChangeResult Set(const JsSymbolId &var, ValueT new_value);

  mlir::ChangeResult Join(const JsSymbolId &var, ValueT other_value);

  mlir::ChangeResult Join(const JsirPerVarState<ValueT> &other) override;

  bool operator==(const JsirPerVarState<ValueT> &rhs) const override;

  bool operator!=(const JsirPerVarState<ValueT> &rhs) const override;

  void print(llvm::raw_ostream &os) const override;

 private:
  ValueT default_value_;
};

template <typename ValueT>
const ValueT &JsirPerVarState<ValueT>::Get(const JsSymbolId &var) const {
  auto it = find(var);
  if (it == end()) {
    return default_value_;
  }
  return it->second;
}

template <typename ValueT>
mlir::ChangeResult JsirPerVarState<ValueT>::Set(
    const JsSymbolId &var,
    llvm::function_ref<mlir::ChangeResult(ValueT &)> setter) {
  auto it = find(var);
  if (it == end()) {
    ValueT value = default_value_;
    mlir::ChangeResult result = setter(value);
    if (result == mlir::ChangeResult::NoChange) {
      return mlir::ChangeResult::NoChange;
    }
    // If we reach here, then `value` must be something non-default.
    insert({var, std::move(value)});
    return mlir::ChangeResult::Change;
  }
  mlir::ChangeResult result = setter(it->second);
  if (it->second == default_value_) {
    erase(it);
  }
  return result;
}

template <typename ValueT>
mlir::ChangeResult JsirPerVarState<ValueT>::Set(const JsSymbolId &var,
                                                ValueT new_value) {
  return Set(var, [&new_value](ValueT &value) {
    if (value == new_value) {
      return mlir::ChangeResult::NoChange;
    }
    value = new_value;
    return mlir::ChangeResult::Change;
  });
}

template <typename ValueT>
mlir::ChangeResult JsirPerVarState<ValueT>::Join(const JsSymbolId &var,
                                                 ValueT other_value) {
  return Set(var,
             [&other_value](ValueT &value) { return value.Join(other_value); });
}

template <typename ValueT>
mlir::ChangeResult JsirPerVarState<ValueT>::Join(
    const JsirPerVarState<ValueT> &other) {
  mlir::ChangeResult result = mlir::ChangeResult::NoChange;
  for (const auto &[symbol, other_value] : other) {
    result |= Join(symbol, other_value);
  }

  for (auto &[symbol, value] : *this) {
    if (other.find(symbol) != other.end()) {
      continue;
    }
    result |= value.Join(other.default_value_);
  }

  result |= default_value_.Join(other.default_value_);

  absl::erase_if(*this, [&](auto &kv) { return kv.second == default_value_; });
  return result;
}

template <typename ValueT>
bool JsirPerVarState<ValueT>::operator==(
    const JsirPerVarState<ValueT> &rhs) const {
  if (default_value_ != rhs.default_value_) {
    return false;
  }

  // 1. The key appears in both maps.
  for (const auto &[symbol, value] : *this) {
    auto rhs_it = rhs.find(symbol);
    if (rhs_it == rhs.end()) {
      continue;
    }

    if (value != rhs_it->second) {
      return false;
    }
  }

  // 2. The key only appears in this map.
  for (const auto &[symbol, value] : *this) {
    auto rhs_it = rhs.find(symbol);
    if (rhs_it != rhs.end()) {
      continue;
    }

    if (value != default_value_) {
      return false;
    }
  }

  // 3. The key only appears in the other map.
  for (const auto &[symbol, value] : rhs) {
    auto it = find(symbol);
    if (it != end()) {
      continue;
    }

    if (value != default_value_) {
      return false;
    }
  }

  return true;
}

template <typename ValueT>
bool JsirPerVarState<ValueT>::operator!=(
    const JsirPerVarState<ValueT> &rhs) const {
  return !(*this == rhs);
}

template <typename ValueT>
void JsirPerVarState<ValueT>::print(llvm::raw_ostream &os) const {
  os << "State [default = ";
  default_value_.print(os);
  os << "] { ";

  std::vector<std::pair<JsSymbolId, ValueT>> pairs{begin(), end()};
  absl::c_sort(pairs, [&](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });

  for (auto &[symbol, value] : pairs) {
    os << "<" << symbol << " : ";
    value.print(os);
    os << "> ";
  }
  os << "}";
}

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_ANALYSES_PER_VAR_STATE_H_
