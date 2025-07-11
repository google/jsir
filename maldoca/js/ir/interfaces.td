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

// The op interfaces for JSIR ops.
//
// We can implicitly cast a concrete op to an interface:
//
// ```
// JsirBinaryExpressionOp binary_expression = ...;
// JsirExpressionOpInterface expression = binary_expression;
// ```
//
// We can also explicitly cast an interface to a concrete op:
//
// ```
// JsirExpressionOpInterface expression = ...;
// if (llvm::isa<JsirBinaryExpressionOp>(expression)) {
//   auto binary_expression = llvm::cast<JsirBinaryExpressionOp>(expression);
//   ...
// }
// ```
//
// This way we are effectively defining base classes.
//
// Note that MLIR does not support interface hierarchies. In other words, we
// cannot specify that JsirLiteralOpInterface "is a" JsirExpressionOpInterface.
//
// As a result, for each interface, we define a "traits" list that contains the
// interface itself and all the ancestor interfaces.
//
// For example, JsirLiteralTraits = [JsirExpressionOpInterface,
// JsirLiteralOpInterface].

#ifndef MALDOCA_JS_IR_INTERFACES_TD_
#define MALDOCA_JS_IR_INTERFACES_TD_

include "mlir/IR/OpBase.td"

def JsirExpressionOpInterface : OpInterface<"JsirExpressionOpInterface"> {
  let cppNamespace = "::maldoca";

  let extraClassDeclaration = [{
    operator mlir::Value() {  // NOLINT
      return getOperation()->getResult(0);
    }
  }];
}

def JsirExpressionOpInterfaceTraits : TraitList<[
  DeclareOpInterfaceMethods<JsirExpressionOpInterface>
]>;

def JsirPatternRefOpInterface : OpInterface<"JsirPatternRefOpInterface"> {
  let cppNamespace = "::maldoca";

  let extraClassDeclaration = [{
    operator mlir::Value() {  // NOLINT
      return getOperation()->getResult(0);
    }
  }];
}

def JsirPatternRefOpInterfaceTraits : TraitList<[
  DeclareOpInterfaceMethods<JsirPatternRefOpInterface>
]>;

def JsirLiteralOpInterface : OpInterface<"JsirLiteralOpInterface"> {
  let cppNamespace = "::maldoca";

  let extraClassDeclaration = [{
    operator mlir::Value() {  // NOLINT
      return getOperation()->getResult(0);
    }
  }];
}

def JsirLiteralOpInterfaceTraits : TraitList<[
  ConstantLike,
  DeclareOpInterfaceMethods<JsirLiteralOpInterface>,
  DeclareOpInterfaceMethods<JsirExpressionOpInterface>
]>;

def JsirLValRefOpInterface : OpInterface<"JsirLValRefOpInterface"> {
  let cppNamespace = "::maldoca";

  let extraClassDeclaration = [{
    operator mlir::Value() {  // NOLINT
      return getOperation()->getResult(0);
    }
  }];
}

def JsirLValRefOpInterfaceTraits : TraitList<[
  DeclareOpInterfaceMethods<JsirLValRefOpInterface>
]>;

def JsirStatementOpInterface : OpInterface<"JsirStatementOpInterface"> {
  let cppNamespace = "::maldoca";
}

def JsirStatementOpInterfaceTraits : TraitList<[
  DeclareOpInterfaceMethods<JsirStatementOpInterface>
]>;

def JsirDeclarationOpInterface : OpInterface<"JsirDeclarationOpInterface"> {
  let cppNamespace = "::maldoca";
}

def JsirDeclarationOpInterfaceTraits : TraitList<[
  DeclareOpInterfaceMethods<JsirStatementOpInterface>,
  DeclareOpInterfaceMethods<JsirDeclarationOpInterface>
]>;

def JsirModuleDeclarationOpInterface : OpInterface<"JsirModuleDeclarationOpInterface"> {
  let cppNamespace = "::maldoca";
}

def JsirModuleDeclarationOpInterfaceTraits : TraitList<[
  DeclareOpInterfaceMethods<JsirModuleDeclarationOpInterface>
]>;

def JsirModuleSpecifierAttrInterface : AttrInterface<"JsirModuleSpecifierAttrInterface"> {
  let cppNamespace = "::maldoca";
}

def JsirCommentAttrInterface : AttrInterface<"JsirCommentAttrInterface"> {
  let cppNamespace = "::maldoca";
}

#endif  // MALDOCA_JS_IR_INTERFACES_TD_
