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
  let assemblyFormat = "attr-dict";
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
  let assemblyFormat = "attr-dict";
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
  let assemblyFormat = "attr-dict";
}

// ForStatement-Related Ops
//
// jshir.for_statement ( {
//   %init = ...
//   jsir.expr_region_end (%init)
// },  {
//   %test = ...
//   jsir.expr_region_end (%test)
// },  {
//   %update = ...
//   jsir.expr_region_update (%update)
// },  {
//   ...  // body
// })

def JshirForStatementOp : Jshir_Op<
    "for_statement", [
        JsirStatementOpInterfaceTraits,
        NoTerminator
    ]> {
  let regions = (region
    OptionalRegion<UnknownRegion>: $init,
    OptionalRegion<ExprRegion>: $test,
    OptionalRegion<ExprRegion>: $update,
    StmtRegion: $body
  );
  let assemblyFormat = "attr-dict-with-keyword `(` $init `,` $test `,` $update `,` $body `)`";
}

// For{In,Of}Statement-Related Ops
//
// %left = ...
// %right = ...
// jshir.for_in_statement (%left, %right) {
//   ...  // body
// })
//                                   - OR -
// jshir.for_of_statement (%left, %right) {
//   ...  // body
// }) {await = true}

def JshirForInStatementOp : Jshir_Op<
    "for_in_statement", [
        JsirStatementOpInterfaceTraits,
        NoTerminator
    ]> {
  let arguments = (ins
    OptionalAttr<JsirForInOfDeclarationAttr>: $left_declaration,
    AnyType: $left_lval,  // JsLVal
    AnyType: $right  // JsExpression
  );

  let regions = (region
    StmtRegion: $body
  );
  let assemblyFormat = "`(` $left_lval `,` $right `)` attr-dict $body";
}

def JshirForOfStatementOp : Jshir_Op<
    "for_of_statement", [
        JsirStatementOpInterfaceTraits,
        NoTerminator
    ]> {
  let arguments = (ins
    OptionalAttr<JsirForInOfDeclarationAttr>: $left_declaration,
    AnyType: $left_lval,  // JsLVal
    AnyType: $right,  // JsExpression
    BoolAttr: $await
  );

  let regions = (region
    StmtRegion: $body
  );
  let assemblyFormat = "`(` $left_lval `,` $right `)` attr-dict $body";
}

def JsirArrowFunctionExpressionOp : Jsir_Op<
    "arrow_function_expression", [
        JsirExpressionOpInterfaceTraits,
        AttrSizedOperandSegments,
        NoTerminator
    ]> {
  let arguments = (ins
    Optional<AnyType>: $id,
    Variadic<AnyType>: $params,
    BoolAttr: $generator,
    BoolAttr: $async
  );

  let regions = (region
    UnknownRegion: $body
  );

  let results = (outs
    JsirAnyType
  );
  let assemblyFormat = "`(` $id `,` $params `)` attr-dict $body";
}

def JsirObjectLiteralKeyType : OptionalAttr<AnyAttrOf<[
    JsirIdentifierAttr,
    JsirStringLiteralAttr,
    JsirNumericLiteralAttr,
    JsirBigIntLiteralAttr
]>>;

def JsirObjectPropertyOp : Jsir_Op<"object_property", []> {
  let arguments = (ins
    JsirObjectLiteralKeyType: $literal_key,
    Optional<AnyType>: $computed_key,
    BoolAttr: $shorthand,
    AnyType: $value  // Expression
  );

  let results = (outs
    JsirAnyType
  );
  let assemblyFormat = "`(` $computed_key `,` $value `)` attr-dict";
}

def JsirObjectPropertyRefOp : Jsir_Op<"object_property_ref", []> {
  let arguments = (ins
    JsirObjectLiteralKeyType: $literal_key,
    Optional<AnyType>: $computed_key,
    BoolAttr: $shorthand,
    AnyType: $value  // Pattern
  );

  let results = (outs
    JsirAnyType
  );
  let assemblyFormat = "`(` $computed_key `,` $value `)` attr-dict";
}

def JsirObjectMethodOp : Jsir_Op<
    "object_method", [
        AttrSizedOperandSegments,
        NoTerminator
    ]> {
  let arguments = (ins
    JsirObjectLiteralKeyType: $literal_key,
    Optional<AnyType>: $computed_key,
    OptionalAttr<JsirIdentifierAttr>: $id,
    Variadic<AnyType>: $params,
    BoolAttr: $generator,
    BoolAttr: $async,
    StrAttr: $kind
  );

  let regions = (region
    StmtRegion: $body
  );

  let results = (outs
    JsirAnyType
  );
  let assemblyFormat = "`(` $computed_key `,` $params `)` attr-dict $body";
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
  let assemblyFormat = "attr-dict $region";
}

// - If `computed` is `true`, the op corresponds to a computed (`a[b]`) member
//   expression.
//   - `literal_property` is nullptr.
//   - `computed_property` is an `Expression`.
// - If `computed` is `false`, the op corresponds to a static (`a.b`) member
//   expression.
//   - `literal_property` is an `Identifier` or a `PrivateName`.
//   - `computed_property` is nullptr.
def JsirMemberExpressionOp : Jsir_Op<
    "member_expression", [
        JsirExpressionOpInterfaceTraits
    ]> {
  let hasFolder = 1;

  let arguments = (ins
    AnyType: $object,
    OptionalAttr<AnyAttrOf<[JsirIdentifierAttr, JsirPrivateNameAttr]>>: $literal_property,
    Optional<AnyType>: $computed_property
  );

  let results = (outs
    JsirAnyType
  );
  let assemblyFormat = "`(` $object `,` $computed_property `)` attr-dict";
}

def JsirMemberExpressionRefOp : Jsir_Op<
    "member_expression_ref", [
        JsirPatternRefOpInterfaceTraits,
        JsirLValRefOpInterfaceTraits
    ]> {
  let arguments = (ins
    AnyType: $object,
    OptionalAttr<AnyAttrOf<[JsirIdentifierAttr, JsirPrivateNameAttr]>>: $literal_property,
    Optional<AnyType>: $computed_property
  );

  let results = (outs
    JsirAnyType
  );
  let assemblyFormat = "`(` $object `,` $computed_property `)` attr-dict";
}

def JsirOptionalMemberExpressionOp : Jsir_Op<
    "optional_member_expression", [
        JsirExpressionOpInterfaceTraits
    ]> {
  let arguments = (ins
    AnyType: $object,
    OptionalAttr<AnyAttrOf<[JsirIdentifierAttr, JsirPrivateNameAttr]>>: $literal_property,
    Optional<AnyType>: $computed_property,
    BoolAttr: $optional
  );

  let results = (outs
    JsirAnyType
  );
  let assemblyFormat = "`(` $object `,` $computed_property `)` attr-dict";
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
  let assemblyFormat = "`(` $expression `)` attr-dict";
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
  let assemblyFormat = "`(` $expression `)` attr-dict";
}

def JsirClassMethodOp : Jsir_Op<
    "class_method", [
        AttrSizedOperandSegments,
        NoTerminator
    ]> {
  let arguments = (ins
    OptionalAttr<JsirIdentifierAttr>: $id,
    Variadic<AnyType>: $params,
    BoolAttr: $generator,
    BoolAttr: $async,
    JsirObjectLiteralKeyType: $literal_key,
    Optional<AnyType>: $computed_key,
    StrAttr: $kind,
    BoolAttr: $static_
  );

  let regions = (region
    StmtRegion: $body
  );
  let assemblyFormat = "`(` $params `,` $computed_key `)` attr-dict $body";
}

def JsirClassPrivateMethodOp : Jsir_Op<
    "class_private_method", [
        NoTerminator
    ]> {
  let arguments = (ins
    OptionalAttr<JsirIdentifierAttr>: $id,
    Variadic<AnyType>: $params,
    BoolAttr: $generator,
    BoolAttr: $async,
    JsirPrivateNameAttr: $key,
    StrAttr: $kind,
    BoolAttr: $static_
  );

  let regions = (region
    StmtRegion: $body
  );
  let assemblyFormat = "`(` $params `)` attr-dict $body";
}

def JsirClassPropertyOp : Jsir_Op<"class_property", [NoTerminator]> {
  let arguments = (ins
    JsirObjectLiteralKeyType: $literal_key,
    Optional<AnyType>: $computed_key,
    BoolAttr: $static_
  );

  let regions = (region
    OptionalRegion<ExprRegion>: $value
  );
  let assemblyFormat = "`(` $computed_key `)` attr-dict $value";
}

def JsirExportDefaultDeclarationOp : Jsir_Op<
    "export_default_declaration", [
        JsirModuleDeclarationOpInterfaceTraits,
        NoTerminator
    ]> {
  let regions = (region
    UnknownRegion: $declaration
  );
  let assemblyFormat = "attr-dict $declaration";
}

#endif  // MALDOCA_JS_IR_JSIR_HANDWRITTEN_TD_
