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

#ifndef MALDOCA_JS_IR_JSIR_TD_
#define MALDOCA_JS_IR_JSIR_TD_

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "maldoca/js/ir/interfaces.td"
include "maldoca/js/ir/jsir_attrs.td"
include "maldoca/js/ir/jsir_dialect.td"

// Import the generated ops.
include "maldoca/js/ir/jsir_ops.generated.td"

def JsirNoneOp : Jsir_Op<"none", [Pure]> {
  let results = (outs
    JsirAnyType
  );
}

#endif  // MALDOCA_JS_IR_JSIR_TD_
