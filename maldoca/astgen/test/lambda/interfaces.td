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

// Definition of op interfaces used by the lir dialect.
//
// Just like we model leaf classes as MLIR ops, we model non-leaf classes as
// MLIR interfaces.
//
// For example, `FunctionDefinition` inherits from `Expression`, so we define an
// interface `LairExpressionOpInterface`.
//
// This way, we can implicitly convert an `LairFunctionDefinitionOp` to an
// `LairExpressionOpInterface`:
//
// ```
// LairFunctionDefinitionOp function_definition = ...;
// LairExpressionOpInterface expression = function_definition;
// ```
//
// We can also type check and explicitly convert an `LairExpressionOpInterface`
// to an `LairFunctionDefinitionOp`:
//
// ```
// LairExpressionOpInterface expression = ...;
// if (llvm::isa<LairFunctionDefinitionOp>(expression)) {
//   auto function_definition =
//       llvm::cast<LairFunctionDefinitionOp>(expression);
//   ...
// }
// ```

include "mlir/IR/OpBase.td"

def LairExpressionOpInterface : OpInterface<"LairExpressionOpInterface"> {
  let cppNamespace = "::maldoca";

  let extraClassDeclaration = [{
    operator mlir::Value() {  // NOLINT
      return getOperation()->getResult(0);
    }
  }];
}

def LairExpressionOpInterfaceTraits : TraitList<[
    DeclareOpInterfaceMethods<LairExpressionOpInterface>
]>;
