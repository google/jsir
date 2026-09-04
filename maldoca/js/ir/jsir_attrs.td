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
    "std::optional<int64_t>": $binding_uid
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
