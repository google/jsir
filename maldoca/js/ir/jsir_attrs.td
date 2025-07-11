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

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "maldoca/js/ir/interfaces.td"
include "maldoca/js/ir/jsir_dialect.td"

// Constant values for control flow structure kinds.
//
// We want to group the elements of the enum JsirControlFlowMarkerKind by
// control flow structure kinds, so we define these bitmasks. Here we leave 4
// bits for each control flow kind, so each of them has 16 kinds of markers
// available.
defvar block_statement_kind = 0x0;
defvar with_statement_kind = 0x10;
defvar labeled_statement_kind = 0x20;
defvar if_statement_kind = 0x30;
defvar switch_statement_kind = 0x40;
defvar try_statement_kind = 0x50;
defvar break_statement_kind = 0x60;
defvar continue_statement_kind = 0x70;
defvar while_statement_kind = 0x80;
defvar do_while_statement_kind = 0x90;
defvar for_statement_kind = 0xA0;
defvar for_in_statement_kind = 0xB0;
defvar for_of_statement_kind = 0xC0;
defvar logical_expression_kind = 0xD0;
defvar conditional_expression_kind = 0xE0;

// Generates the C++ enum JsirControlFlowKind.
def JsirControlFlowKind : I32EnumAttr<
    "JsirControlFlowKind", "The kind of JSIR control flow structure.",
    [
      I32EnumAttrCase<"BlockStatement", block_statement_kind>,
      I32EnumAttrCase<"WithStatement", with_statement_kind>,
      I32EnumAttrCase<"LabeledStatement", labeled_statement_kind>,
      I32EnumAttrCase<"IfStatement", if_statement_kind>,
      I32EnumAttrCase<"SwitchStatement", switch_statement_kind>,
      I32EnumAttrCase<"TryStatement", try_statement_kind>,
      I32EnumAttrCase<"BreakStatement", break_statement_kind>,
      I32EnumAttrCase<"ContinueStatement", continue_statement_kind>,
      I32EnumAttrCase<"WhileStatement", while_statement_kind>,
      I32EnumAttrCase<"DoWhileStatement", do_while_statement_kind>,
      I32EnumAttrCase<"ForStatement", for_statement_kind>,
      I32EnumAttrCase<"ForInStatement", for_in_statement_kind>,
      I32EnumAttrCase<"ForOfStatement", for_of_statement_kind>,
      I32EnumAttrCase<"LogicalExpression", logical_expression_kind>,
      I32EnumAttrCase<"ConditionalExpression", conditional_expression_kind>,
    ]
> {
  let cppNamespace = "maldoca";

  let stringToSymbolFnName = "StringToJsirControlFlowKind";
  let symbolToStringFnName = "JsirControlFlowKindToString";

  let genSpecializedAttr = 0;
}

// Generates the attribute JsirControlFlowKindAttr.
// This attribute can be used as a parameter in an op.
def JsirControlFlowKindAttr
    : EnumAttr<Jsir_Dialect, JsirControlFlowKind, "cf_kind">;

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

// Generates the C++ enum JsirControlFlowMarkerKind.

def JsirControlFlowMarkerKind : I32EnumAttr<
    "JsirControlFlowMarkerKind", "The kind of JSIR control flow marker.",
    [
      // BlockStatement
      I32EnumAttrCase<"BlockStatementDirectives",
                      !add(block_statement_kind, 0)>,
      I32EnumAttrCase<"BlockStatementBody",
                      !add(block_statement_kind, 1)>,
      I32EnumAttrCase<"BlockStatementEnd",
                      !add(block_statement_kind, 2)>,

      // WithStatement
      I32EnumAttrCase<"WithStatementBody",
                      !add(with_statement_kind, 0)>,
      I32EnumAttrCase<"WithStatementEnd",
                      !add(with_statement_kind, 1)>,

      // LabeledStatement
      I32EnumAttrCase<"LabeledStatementEnd",
                      !add(labeled_statement_kind, 0)>,

      // IfStatement
      I32EnumAttrCase<"IfStatementConsequent",
                      !add(if_statement_kind, 0)>,
      I32EnumAttrCase<"IfStatementAlternate",
                      !add(if_statement_kind, 1)>,
      I32EnumAttrCase<"IfStatementEnd",
                      !add(if_statement_kind, 2)>,

      // SwitchStatement
      I32EnumAttrCase<"SwitchStatementCaseBody",
                      !add(switch_statement_kind, 1)>,
      I32EnumAttrCase<"SwitchStatementEnd",
                      !add(switch_statement_kind, 2)>,

      // TryStatement
      I32EnumAttrCase<"TryStatementBody",
                      !add(try_statement_kind, 0)>,
      I32EnumAttrCase<"TryStatementHandler",
                      !add(try_statement_kind, 1)>,
      I32EnumAttrCase<"TryStatementFinalizer",
                      !add(try_statement_kind, 2)>,
      I32EnumAttrCase<"TryStatementEnd",
                      !add(try_statement_kind, 3)>,

      // WhileStatement
      I32EnumAttrCase<"WhileStatementTest",
                      !add(while_statement_kind, 0)>,
      I32EnumAttrCase<"WhileStatementBody",
                      !add(while_statement_kind, 1)>,
      I32EnumAttrCase<"WhileStatementEnd",
                      !add(while_statement_kind, 2)>,

      // DoWhileStatement
      I32EnumAttrCase<"DoWhileStatementBody",
                      !add(do_while_statement_kind, 0)>,
      I32EnumAttrCase<"DoWhileStatementTest",
                      !add(do_while_statement_kind, 1)>,
      I32EnumAttrCase<"DoWhileStatementEnd",
                      !add(do_while_statement_kind, 2)>,

      // ForStatement
      I32EnumAttrCase<"ForStatementInit",
                      !add(for_statement_kind, 0)>,
      I32EnumAttrCase<"ForStatementTest",
                      !add(for_statement_kind, 1)>,
      I32EnumAttrCase<"ForStatementBody",
                      !add(for_statement_kind, 2)>,
      I32EnumAttrCase<"ForStatementUpdate",
                      !add(for_statement_kind, 3)>,
      I32EnumAttrCase<"ForStatementEnd",
                      !add(for_statement_kind, 4)>,

      // ForInStatement
      I32EnumAttrCase<"ForInStatementBody",
                      !add(for_in_statement_kind, 0)>,
      I32EnumAttrCase<"ForInStatementEnd",
                      !add(for_in_statement_kind, 1)>,

      // ForOfStatement
      I32EnumAttrCase<"ForOfStatementBody",
                      !add(for_of_statement_kind, 0)>,
      I32EnumAttrCase<"ForOfStatementEnd",
                      !add(for_of_statement_kind, 1)>,

      // LogicalExpression
      I32EnumAttrCase<"LogicalExpressionRight",
                      !add(logical_expression_kind, 0)>,
      I32EnumAttrCase<"LogicalExpressionEnd",
                      !add(logical_expression_kind, 1)>,

      // ConditionalExpression
      I32EnumAttrCase<"ConditionalExpressionAlternate",
                      !add(conditional_expression_kind, 0)>,
      I32EnumAttrCase<"ConditionalExpressionConsequent",
                      !add(conditional_expression_kind, 1)>,
      I32EnumAttrCase<"ConditionalExpressionEnd",
                      !add(conditional_expression_kind, 2)>,
    ]
> {
  let cppNamespace = "maldoca";

  let stringToSymbolFnName = "StringToJsirControlFlowMarkerKind";
  let symbolToStringFnName = "JsirControlFlowMarkerKindToString";

  let genSpecializedAttr = 0;
}

// Generates the attribute JsirControlFlowMarkerKindAttr.
// This attribute can be used as a parameter in an op.
def JsirControlFlowMarkerKindAttr
    : EnumAttr<Jsir_Dialect, JsirControlFlowMarkerKind,
               "cf_marker">;

def JsirPositionAttr : AttrDef<Jsir_Dialect, "JsirPosition", []> {
  let mnemonic = "position";
  let parameters = (
    ins
      "int64_t": $line,
      "int64_t": $column
  );
  let assemblyFormat = "`<` `L` $line `C` $column `>`";
}

def JsirLocationAttr : AttrDef<Jsir_Dialect, "JsirLocation", [],
                               "::mlir::LocationAttr"> {
  let mnemonic = "location";
  let parameters = (
    ins
      OptionalParameter<"JsirPositionAttr">: $start,
      OptionalParameter<"JsirPositionAttr">: $end,
      OptionalParameter<"::mlir::StringAttr">: $identifier_name,
      OptionalParameter<"std::optional<int64_t>">: $start_index,
      OptionalParameter<"std::optional<int64_t>">: $end_index,
      OptionalParameter<"std::optional<int64_t>">: $scope_uid
  );
  let assemblyFormat = "params";
}

def JsirCommentBlockAttr : AttrDef<
    Jsir_Dialect, "JsirCommentBlock", [
        DeclareAttrInterfaceMethods<JsirCommentAttrInterface>
    ]> {
  let mnemonic = "comment_block";
  let parameters = (ins
    "JsirLocationAttr": $loc,
    "::mlir::StringAttr": $value
  );
  let assemblyFormat = "params";
}

def JsirCommentLineAttr : AttrDef<
    Jsir_Dialect, "JsirCommentLine", [
        DeclareAttrInterfaceMethods<JsirCommentAttrInterface>
    ]> {
  let mnemonic = "comment_line";
  let parameters = (ins
    "JsirLocationAttr": $loc,
    "::mlir::StringAttr": $value
  );
  let assemblyFormat = "params";
}

def JsirSymbolIdAttr : AttrDef<Jsir_Dialect, "JsirSymbolId", []> {
  let mnemonic = "symbol_id";
  let parameters = (ins
    "::mlir::StringAttr": $name,
    "std::optional<int64_t>": $def_scope_id
  );
  let assemblyFormat = "params";
}

def JsirTriviaAttr : AttrDef<Jsir_Dialect, "JsirTrivia", [],
                             "::mlir::LocationAttr"> {
  let mnemonic = "trivia";
  let parameters = (
    ins
      OptionalParameter<"JsirLocationAttr">: $loc,
      OptionalArrayRefParameter<"int64_t">: $leading_comment_uids,
      OptionalArrayRefParameter<"int64_t">: $trailing_comment_uids,
      OptionalArrayRefParameter<"int64_t">: $inner_comment_uids,
      OptionalParameter<"JsirSymbolIdAttr">: $referenced_symbol,
      OptionalArrayRefParameter<"JsirSymbolIdAttr">: $defined_symbols
  );
  let assemblyFormat = "params";
}

def JsirInterpreterDirectiveAttr : AttrDef<Jsir_Dialect, "JsirInterpreterDirective", []> {
  let mnemonic = "interpreter_directive";
  let parameters = (ins
    OptionalParameter<"JsirTriviaAttr">: $loc,
    "::mlir::StringAttr": $value
  );
  let assemblyFormat = "params";
}

def JsirDirectiveLiteralExtraAttr : AttrDef<Jsir_Dialect,
                                            "JsirDirectiveLiteralExtra", []> {
  let mnemonic = "directive_literal_extra";
  let parameters = (ins
    "::mlir::StringAttr": $raw,
    "::mlir::StringAttr": $raw_value
  );
  let assemblyFormat = "params";
}

def JsirIdentifierAttr : AttrDef<Jsir_Dialect, "JsirIdentifier", []> {
  let mnemonic = "identifier";
  let parameters = (ins
    OptionalParameter<"JsirTriviaAttr">: $loc,
    "::mlir::StringAttr": $name
  );
  let assemblyFormat = "params";
}

def JsirPrivateNameAttr : AttrDef<Jsir_Dialect, "JsirPrivateName", []> {
  let mnemonic = "private_name";
  let parameters = (ins
    OptionalParameter<"JsirTriviaAttr">: $loc,
    JsirIdentifierAttr: $id
  );
  let assemblyFormat = "params";
}

def JsirRegExpLiteralAttr : AttrDef<Jsir_Dialect, "JsirRegExpLiteral", []> {
  let mnemonic = "reg_exp_literal";
  let parameters = (ins
    "::mlir::StringAttr": $pattern,
    "::mlir::StringAttr": $flags
  );
  let assemblyFormat = "params";
}

def JsirRegExpLiteralExtraAttr : AttrDef<Jsir_Dialect,
                                         "JsirRegExpLiteralExtra", []> {
  let mnemonic = "reg_exp_literal_extra";
  let parameters = (ins
    "::mlir::StringAttr": $raw
  );
  let assemblyFormat = "params";
}

def JsirNullLiteralAttr : AttrDef<Jsir_Dialect, "JsirNullLiteral", []> {
  let mnemonic = "null_literal";
}

def JsirStringLiteralExtraAttr : AttrDef<Jsir_Dialect,
                                         "JsirStringLiteralExtra", []> {
  let mnemonic = "string_literal_extra";
  let parameters = (ins
    "::mlir::StringAttr": $raw,
    "::mlir::StringAttr": $raw_value
  );
  let assemblyFormat = "params";
}

def JsirStringLiteralAttr : AttrDef<Jsir_Dialect, "JsirStringLiteral", []> {
  let mnemonic = "string_literal";
  let parameters = (ins
    OptionalParameter<"JsirTriviaAttr">: $loc,
    "::mlir::StringAttr": $value,
    JsirStringLiteralExtraAttr: $extra
  );
  let assemblyFormat = "params";
}

def JsirNumericLiteralExtraAttr : AttrDef<Jsir_Dialect,
                                         "JsirNumericLiteralExtra", []> {
  let mnemonic = "numeric_literal_extra";
  let parameters = (ins
    "::mlir::StringAttr": $raw,
    "::mlir::FloatAttr": $raw_value
  );
  let assemblyFormat = "params";
}

def JsirNumericLiteralAttr : AttrDef<Jsir_Dialect, "JsirNumericLiteral", []> {
  let mnemonic = "numeric_literal";
  let parameters = (ins
    OptionalParameter<"JsirTriviaAttr">: $loc,
    "::mlir::FloatAttr": $value,
    JsirNumericLiteralExtraAttr: $extra
  );
  let assemblyFormat = "params";
}

def JsirBigIntLiteralExtraAttr : AttrDef<Jsir_Dialect,
                                         "JsirBigIntLiteralExtra", []> {
  let mnemonic = "big_int_literal_extra";
  let parameters = (ins
    "::mlir::StringAttr": $raw,
    "::mlir::StringAttr": $raw_value
  );
  let assemblyFormat = "params";
}

def JsirBigIntLiteralAttr : AttrDef<Jsir_Dialect, "JsirBigIntLiteral", []> {
  let mnemonic = "big_int_literal";
  let parameters = (ins
    "::mlir::StringAttr": $value,
    JsirBigIntLiteralExtraAttr: $extra
  );
  let assemblyFormat = "params";
}

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

def JsirImportSpecifierAttr : AttrDef<
    Jsir_Dialect, "JsirImportSpecifier", [
        DeclareAttrInterfaceMethods<JsirModuleSpecifierAttrInterface>
    ]> {
  let mnemonic = "import_specifier";
  let parameters = (ins
    OptionalParameter<"JsirTriviaAttr">: $loc,
    "::mlir::Attribute": $imported,
    JsirIdentifierAttr: $local
  );
  let assemblyFormat = "params";
}

def JsirImportDefaultSpecifierAttr : AttrDef<
    Jsir_Dialect, "JsirImportDefaultSpecifier", [
        DeclareAttrInterfaceMethods<JsirModuleSpecifierAttrInterface>
    ]> {
  let mnemonic = "import_default_specifier";
  let parameters = (ins
    OptionalParameter<"JsirTriviaAttr">: $loc,
    JsirIdentifierAttr: $local
  );
  let assemblyFormat = "params";
}

def JsirImportNamespaceSpecifierAttr : AttrDef<
    Jsir_Dialect, "JsirImportNamespaceSpecifier", [
        DeclareAttrInterfaceMethods<JsirModuleSpecifierAttrInterface>
    ]> {
  let mnemonic = "import_namespace_specifier";
  let parameters = (ins
    OptionalParameter<"JsirTriviaAttr">: $loc,
    JsirIdentifierAttr: $local
  );
  let assemblyFormat = "params";
}

def JsirImportAttributeAttr : AttrDef<Jsir_Dialect, "JsirImportAttribute", []> {
  let mnemonic = "import_attribute";
  let parameters = (ins
    JsirIdentifierAttr: $key,
    JsirStringLiteralAttr: $value
  );
  let assemblyFormat = "params";
}

def JsirExportSpecifierAttr : AttrDef<
    Jsir_Dialect, "JsirExportSpecifier", [
        DeclareAttrInterfaceMethods<JsirModuleSpecifierAttrInterface>
    ]> {
  let mnemonic = "export_specifier";
  let parameters = (ins
    OptionalParameter<"JsirTriviaAttr">: $loc,
    "::mlir::Attribute": $exported,
    "::mlir::Attribute": $local
  );
  let assemblyFormat = "params";
}
