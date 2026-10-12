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

#ifndef MALDOCA_JS_IR_JSIR_ATTRS_TD_
#define MALDOCA_JS_IR_JSIR_ATTRS_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "maldoca/js/ir/interfaces.td"
include "maldoca/js/ir/jsir_dialect.td"

def JsirPositionAttr : AttrDef<Jsir_Dialect, "JsirPosition", []> {
  let mnemonic = "position";
  let parameters = (
    ins
      "int64_t": $line,
      "int64_t": $column
  );
  let assemblyFormat = "`<` `L` $line `C` $column `>`";
}

// Import the generated attributes.
include "maldoca/js/ir/jsir_attrs.generated.td"

// Generates the C++ enum JsirForInOfKind.
def JsirForInOfKind : I32EnumAttr<
    "JsirForInOfKind", "ForIn or ForOf.",
    [
      I32EnumAttrCase<"ForIn", 0>,
      I32EnumAttrCase<"ForOf", 1>,
    ]
> {
  let cppNamespace = "maldoca";

  let stringToSymbolFnName = "StringToJsirForInOfKind";
  let symbolToStringFnName = "JsirForInOfKindToString";

  let genSpecializedAttr = 0;
}

// Generates the attribute JsirForInOfKindAttr.
// This attribute can be used as a parameter in an op.
def JsirForInOfKindAttr
    : EnumAttr<Jsir_Dialect, JsirForInOfKind, "for_in_of_kind">;

// ```
// for (let a in b) ...
//      ~~~~~
// ```
//
// `let a` is a `JsVariableDeclaration { JsVariableDeclarator { JsLVal }}` in
// the original AST, but we only store the `JsLVal` part in the IR, which means
// we lose the JsTrivia (source location, comments, etc.) information.
//
// Therefore, we add `declaration_loc` and `declarator_loc` to store that.
def JsirForInOfDeclarationAttr : AttrDef<Jsir_Dialect,
                                         "JsirForInOfDeclaration", []> {
  let mnemonic = "for_in_of_declaration";
  let parameters = (ins
    OptionalParameter<"JsirTriviaAttr">: $declaration_loc,
    OptionalParameter<"JsirTriviaAttr">: $declarator_loc,
    "::mlir::StringAttr": $kind
  );
  let assemblyFormat = "params";
}

#endif  // MALDOCA_JS_IR_JSIR_ATTRS_TD_
