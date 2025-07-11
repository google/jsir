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

#ifndef MALDOCA_JS_IR_ANALYSES_STATE_H_
#define MALDOCA_JS_IR_ANALYSES_STATE_H_

#include "llvm/Support/raw_ostream.h"
#include "mlir/Analysis/DataFlowFramework.h"

namespace maldoca {

// A state that can be attached to program points. Must conform to the rules of
// a semi-lattice.
template <typename T>
class JsirState {
 public:
  virtual ~JsirState() = default;

  // Joins this state with another, and updates this state to the result.
  //
  // This function is called when program paths merge.
  //
  // Constant propagation example:
  // ```
  // if (...) {
  //   a = 1;
  //   <- At this point, a == 1.
  // } else {
  //   a = 2;
  //   <- At this point, a == 2.
  // }
  // <- At this point, a == join(1, 2) == `Unknown`.
  // ```
  virtual mlir::ChangeResult Join(const T &other) = 0;

  virtual bool operator==(const T &rhs) const = 0;
  virtual bool operator!=(const T &rhs) const { return !(operator==(rhs)); }

  virtual void print(llvm::raw_ostream &os) const = 0;
};

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_ANALYSES_STATE_H_
