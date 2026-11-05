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

// BreakStatement-Related Ops
//
// jshir.break_statement (%label optional argument)

def JshirBreakStatementOp : Jshir_Op<
    "break_statement", [
        JsirStatementOpInterfaceTraits
    ]> {
  let arguments = (ins
    OptionalAttr<JsirIdentifierAttr>: $label
  );
}

// ContinueStatement-Related Ops
//
// jshir.continue_statement (%label optional argument)

def JshirContinueStatementOp : Jshir_Op<
    "continue_statement", [
        JsirStatementOpInterfaceTraits
    ]> {
  let arguments = (ins
    OptionalAttr<JsirIdentifierAttr>: $label
  );
}


def JsirObjectExpressionOp : Jsir_Op<
    "object_expression", [
        JsirExpressionOpInterfaceTraits,
        NoTerminator
    ]> {
  let regions = (region
    ExprsRegion: $region
  );

  let results = (outs
    JsirAnyType
  );
}

def JsirParenthesizedExpressionOp : Jsir_Op<
    "parenthesized_expression", [
        JsirExpressionOpInterfaceTraits,
        Pure
    ]> {
  let hasFolder = 1;

  let arguments = (ins
    AnyType: $expression
  );

  let results = (outs
    JsirAnyType
  );
}

def JsirParenthesizedExpressionRefOp : Jsir_Op<
    "parenthesized_expression_ref", [
        JsirPatternRefOpInterfaceTraits,
        JsirLValRefOpInterfaceTraits,
        Pure
    ]> {
  let hasFolder = 1;

  let arguments = (ins
    AnyType: $expression
  );

  let results = (outs
    JsirAnyType
  );
}

def JsirClassPrivateMethodOp : Jsir_Op<
    "class_private_method", [
        NoTerminator,
        JsirFunctionOpInterfaceTraits
    ]> {
  let arguments = (ins
    OptionalAttr<JsirIdentifierAttr>: $id,
    BoolAttr: $generator,
    BoolAttr: $async,
    JsirPrivateNameAttr: $key,
    StrAttr: $kind,
    BoolAttr: $static_
  );

  let regions = (region
    ExprsRegion: $params,
    StmtRegion: $body
  );
}

// A specifier in `export {local as exported}`.
//
// `exported` is the name visible to importers, so it is an attribute.
//
// `local` is an SSA operand:
// - Identifier: `jsir.identifier_ref`, a reference to the exported binding.
// - StringLiteral: `jsir.string_literal`. Only valid in re-exports such as
//   `export {"a-b" as c} from "m"`.
// In a re-export (`source` set on the parent), `local` names a binding in the
// source module, so it has no binding in this file.
//
// Handwritten because astgen's printers don't honor
// `ir_mapping.lval_member_types` yet: the Identifier member is an lvalue, but
// the StringLiteral member is an rvalue.
def JsirExportSpecifierOp : Jsir_Op<"export_specifier", []> {
  let arguments = (ins
    AnyAttrOf<[JsirIdentifierAttr, JsirStringLiteralAttr]>: $exported,
    Optional<AnyType>: $local
  );
}

#endif  // MALDOCA_JS_IR_JSIR_HANDWRITTEN_TD_
