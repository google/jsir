// Copyright 2026 Google LLC
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

#ifndef MALDOCA_ASTGEN_TEST_SPLIT_INTERFACES_TD_
#define MALDOCA_ASTGEN_TEST_SPLIT_INTERFACES_TD_

include "mlir/IR/OpBase.td"

def SirExpressionOpInterface : OpInterface<"SirExpressionOpInterface"> {
  let cppNamespace = "::maldoca";

  let extraClassDeclaration = [{
    operator mlir::Value() {  // NOLINT
      return getOperation()->getResult(0);
    }
  }];
}

def SirExpressionOpInterfaceTraits : TraitList<[
  DeclareOpInterfaceMethods<SirExpressionOpInterface>
]>;

def SirPatternRefOpInterface : OpInterface<"SirPatternRefOpInterface"> {
  let cppNamespace = "::maldoca";

  let extraClassDeclaration = [{
    operator mlir::Value() {  // NOLINT
      return getOperation()->getResult(0);
    }
  }];
}

def SirPatternRefOpInterfaceTraits : TraitList<[
  DeclareOpInterfaceMethods<SirPatternRefOpInterface>
]>;

def SirStatementOpInterface : OpInterface<"SirStatementOpInterface"> {
  let cppNamespace = "::maldoca";
}

def SirStatementOpInterfaceTraits : TraitList<[
  DeclareOpInterfaceMethods<SirStatementOpInterface>
]>;

#endif  // MALDOCA_ASTGEN_TEST_SPLIT_INTERFACES_TD_
