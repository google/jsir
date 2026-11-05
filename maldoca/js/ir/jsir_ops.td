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

// ForStatement-Related Ops
//
// `init` is either a `VariableDeclaration` (a statement, held in the
// `init_declaration` region) or an expression (held in the `init_expression`
// region). At most one of them is set.
//
// jshir.for_statement ( {
//   jsir.variable_declaration ...
// },  {
// },  {
//   %test = ...
//   jsir.expr_region_end (%test)
// },  {
//   %update = ...
//   jsir.expr_region_end (%update)
// },  {
//   ...  // body
// })

def JshirForStatementOp : Jshir_Op<
    "for_statement", [
        JsirStatementOpInterfaceTraits,
        NoTerminator,
        PredOpTrait<"at most one of init_declaration, init_expression is set", CPred<"(int(!::llvm::cast<JshirForStatementOp>(&$_op).getInitDeclaration().empty()) + int(!::llvm::cast<JshirForStatementOp>(&$_op).getInitExpression().empty())) <= 1">>
    ]> {
  let regions = (region
    OptionalRegion<StmtRegion>: $init_declaration,
    OptionalRegion<ExprRegion>: $init_expression,
    OptionalRegion<ExprRegion>: $test,
    OptionalRegion<ExprRegion>: $update,
    StmtRegion: $body
  );
}

// For{In,Of}Statement-Related Ops
//
// `left` is either a `VariableDeclaration` (a statement, held in the
// `left_declaration` region, e.g. `for (const x of xs)`) or an `LVal` (held in
// the `left_lval` operand, e.g. `for (x of xs)`). Exactly one of them is set.
//
// %right = ...
// jshir.for_in_statement (%right) ({
//   jsir.variable_declaration ...
// }, {
//   ...  // body
// })
//                                   - OR -
// %left = ...
// %right = ...
// jshir.for_of_statement (%left, %right) ({
// }, {
//   ...  // body
// }) {await = true}

def JshirForInStatementOp : Jshir_Op<
    "for_in_statement", [
        JsirStatementOpInterfaceTraits,
        NoTerminator,
        PredOpTrait<"exactly one of left_declaration, left_lval is set", CPred<"(int(!::llvm::cast<JshirForInStatementOp>(&$_op).getLeftDeclaration().empty()) + int(::llvm::cast<JshirForInStatementOp>(&$_op).getLeftLval() != nullptr)) == 1">>
    ]> {
  let arguments = (ins
    Optional<AnyType>: $left_lval,
    AnyType: $right
  );

  let regions = (region
    OptionalRegion<StmtRegion>: $left_declaration,
    StmtRegion: $body
  );
}

def JshirForOfStatementOp : Jshir_Op<
    "for_of_statement", [
        JsirStatementOpInterfaceTraits,
        NoTerminator,
        PredOpTrait<"exactly one of left_declaration, left_lval is set", CPred<"(int(!::llvm::cast<JshirForOfStatementOp>(&$_op).getLeftDeclaration().empty()) + int(::llvm::cast<JshirForOfStatementOp>(&$_op).getLeftLval() != nullptr)) == 1">>
    ]> {
  let arguments = (ins
    Optional<AnyType>: $left_lval,
    AnyType: $right,
    BoolAttr: $await
  );

  let regions = (region
    OptionalRegion<StmtRegion>: $left_declaration,
    StmtRegion: $body
  );
}

// The body is either a `BlockStatement` (held in the `body_statement` region)
// or an expression (held in the `body_expression` region). Exactly one of them
// is set.
def JsirArrowFunctionExpressionOp : Jsir_Op<
    "arrow_function_expression", [
        JsirExpressionOpInterfaceTraits,
        NoTerminator,
        JsirFunctionOpInterfaceTraits,
        PredOpTrait<"exactly one of body_statement, body_expression is set", CPred<"(int(!::llvm::cast<JsirArrowFunctionExpressionOp>(&$_op).getBodyStatement().empty()) + int(!::llvm::cast<JsirArrowFunctionExpressionOp>(&$_op).getBodyExpression().empty())) == 1">>
    ]> {
  let arguments = (ins
    OptionalAttr<JsirIdentifierAttr>: $id,
    BoolAttr: $generator,
    BoolAttr: $async
  );

  let regions = (region
    ExprsRegion: $params,
    OptionalRegion<StmtRegion>: $body_statement,
    OptionalRegion<ExprRegion>: $body_expression
  );

  let results = (outs
    JsirAnyType
  );
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
}

def JsirObjectMethodOp : Jsir_Op<
    "object_method", [
        NoTerminator,
        JsirFunctionOpInterfaceTraits
    ]> {
  let arguments = (ins
    JsirObjectLiteralKeyType: $literal_key,
    Optional<AnyType>: $computed_key,
    OptionalAttr<JsirIdentifierAttr>: $id,
    BoolAttr: $generator,
    BoolAttr: $async,
    StrAttr: $kind
  );

  let regions = (region
    ExprsRegion: $params,
    StmtRegion: $body
  );

  let results = (outs
    JsirAnyType
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

def JsirClassMethodOp : Jsir_Op<
    "class_method", [
        NoTerminator,
        JsirFunctionOpInterfaceTraits
    ]> {
  let arguments = (ins
    OptionalAttr<JsirIdentifierAttr>: $id,
    BoolAttr: $generator,
    BoolAttr: $async,
    JsirObjectLiteralKeyType: $literal_key,
    Optional<AnyType>: $computed_key,
    StrAttr: $kind,
    BoolAttr: $static_
  );

  let regions = (region
    ExprsRegion: $params,
    StmtRegion: $body
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

def JsirClassPropertyOp : Jsir_Op<"class_property", [NoTerminator]> {
  let arguments = (ins
    JsirObjectLiteralKeyType: $literal_key,
    Optional<AnyType>: $computed_key,
    BoolAttr: $static_
  );

  let regions = (region
    OptionalRegion<ExprRegion>: $value
  );
}

// `declaration` is either a function/class declaration (held in the
// `declaration_statement` region) or an expression (held in the
// `declaration_expression` region). Exactly one of them is set.
def JsirExportDefaultDeclarationOp : Jsir_Op<
    "export_default_declaration", [
        JsirModuleDeclarationOpInterfaceTraits,
        NoTerminator,
        PredOpTrait<"exactly one of declaration_statement, declaration_expression is set", CPred<"(int(!::llvm::cast<JsirExportDefaultDeclarationOp>(&$_op).getDeclarationStatement().empty()) + int(!::llvm::cast<JsirExportDefaultDeclarationOp>(&$_op).getDeclarationExpression().empty())) == 1">>
    ]> {
  let regions = (region
    OptionalRegion<StmtRegion>: $declaration_statement,
    OptionalRegion<ExprRegion>: $declaration_expression
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
// Handwritten because astgen applies one FieldKind to every variant member.
def JsirExportSpecifierOp : Jsir_Op<"export_specifier", []> {
  let arguments = (ins
    AnyAttrOf<[JsirIdentifierAttr, JsirStringLiteralAttr]>: $exported,
    Optional<AnyType>: $local
  );
}

#endif  // MALDOCA_JS_IR_JSIR_HANDWRITTEN_TD_
