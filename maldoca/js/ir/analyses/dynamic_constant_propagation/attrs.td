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
include "maldoca/js/ir/jsir_attrs.td"
include "maldoca/js/ir/analyses/dynamic_constant_propagation/jsir_builtin_dialect.td"

def JsirBuiltinFunctionAttr : AttrDef<JsirBuiltin_Dialect,
                                      "JsirBuiltinFunction", []> {
  let mnemonic = "builtin_function";
  let parameters = (ins
    "::mlir::StringAttr": $name
  );
  let assemblyFormat = "params";
}

def JsirInlineExpressionBinaryExpressionAttr : AttrDef<JsirBuiltin_Dialect,
                                                       "JsirInlineExpressionBinaryExpression", []> {
  let mnemonic = "inline_expression_binary_expression";
  let parameters = (ins
    "::mlir::Attribute": $left,
    "::mlir::Attribute": $right,
    "::mlir::StringAttr": $op
  );
  let assemblyFormat = "params";
}

def JsirInlinExpressionUnaryExpressionAttr : AttrDef<JsirBuiltin_Dialect,
                                                     "JsirInlineExpressionUnaryExpression", []> {
  let mnemonic = "inline_expression_unary_expression";
  let parameters = (ins
    "::mlir::Attribute": $argument,
    "::mlir::StringAttr": $op
  );
  let assemblyFormat = "params";
}

def JsirInlineExpressionFunctionAttr : AttrDef<JsirBuiltin_Dialect,
                                               "JsirInlineExpressionFunction", []> {
  let mnemonic = "inline_expression_function";
  let parameters = (ins
    ArrayRefParameter<"JsirSymbolIdAttr">: $params,
    "::mlir::Attribute": $body
  );
  let assemblyFormat = "params";
}

def JsirInlineExpressionCallAttr : AttrDef<JsirBuiltin_Dialect,
                                           "JsirInlineExpressionCall", []> {
  let mnemonic = "inline_expression_call";
  let parameters = (ins
    "::mlir::Attribute": $callee,
    ArrayRefParameter<"::mlir::Attribute">: $arguments
  );
  let assemblyFormat = "params";
}

def JsirInlineExpressionObjectAttr : AttrDef<JsirBuiltin_Dialect,
                                             "JsirInlineExpressionObject", []> {
  let mnemonic = "inline_expression_object";
  let parameters = (ins
    ArrayRefParameter<"JsirInlineExpressionPropertyAttr">: $properties
  );
  let assemblyFormat = "params";
}

def JsirInlineExpressionPropertyAttr : AttrDef<JsirBuiltin_Dialect,
                                               "JsirInlineExpressionProperty", []> {
  let mnemonic = "inline_expression_property";
  let parameters = (ins
    "::mlir::StringAttr": $key,
    "::mlir::Attribute": $value
  );
  let assemblyFormat = "params";
}

def JsirInlineExpressionMemberExpressionAttr : AttrDef<JsirBuiltin_Dialect,
                                                        "JsirInlineExpressionMemberExpression", []> {
  let mnemonic = "inline_expression_member_expression";
  let parameters = (ins
    "::mlir::Attribute": $object,
    "::mlir::Attribute": $computed_key
  );
  let assemblyFormat = "params";
}
