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

#ifndef MALDOCA_JS_IR_IR_H_
#define MALDOCA_JS_IR_IR_H_

#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "maldoca/js/quickjs/quickjs.h"
#include "quickjs/quickjs.h"

// A useful macro to pass to FOR_EACH_JSIR_CLASS to ignore a kind of JSIR class.
#define JSIR_CLASS_IGNORE(...)

// A list of all JSIR and JSHIR ops and attributes:
//
// - CIR_OP: Jsir<TYPE>Op
// - HIR_OP: Jshir<TYPE>Op
// - REF_OP: Jsir<TYPE>RefOp
// - ATTRIB: Jsir<TYPE>Attr
#define FOR_EACH_JSIR_CLASS(CIR_OP, HIR_OP, REF_OP, ATTRIB) \
  /* File */                                                        \
  CIR_OP(File)                                                      \
                                                                    \
  /* Identifiers */                                                 \
  CIR_OP(Identifier)                                                \
  REF_OP(Identifier)                                                \
  ATTRIB(Identifier)                                                \
  CIR_OP(PrivateName)                                               \
  ATTRIB(PrivateName)                                               \
                                                                    \
  /* Literals */                                                    \
  ATTRIB(RegExpLiteralExtra)                                        \
  CIR_OP(RegExpLiteral)                                             \
  CIR_OP(NullLiteral)                                               \
  ATTRIB(StringLiteralExtra)                                        \
  CIR_OP(StringLiteral)                                             \
  ATTRIB(StringLiteral)                                             \
  CIR_OP(BooleanLiteral)                                            \
  ATTRIB(NumericLiteralExtra)                                       \
  CIR_OP(NumericLiteral)                                            \
  ATTRIB(NumericLiteral)                                            \
  ATTRIB(BigIntLiteralExtra)                                        \
  CIR_OP(BigIntLiteral)                                             \
  ATTRIB(BigIntLiteral)                                             \
                                                                    \
  /* Program */                                                     \
  CIR_OP(Program)                                                   \
                                                                    \
  /* Statements */                                                  \
  CIR_OP(ExpressionStatement)                                       \
  HIR_OP(BlockStatement)                                            \
  CIR_OP(EmptyStatement)                                            \
  CIR_OP(DebuggerStatement)                                         \
  HIR_OP(WithStatement)                                             \
  CIR_OP(ReturnStatement)                                           \
  HIR_OP(LabeledStatement)                                          \
  HIR_OP(BreakStatement)                                            \
  HIR_OP(ContinueStatement)                                         \
  HIR_OP(IfStatement)                                               \
  HIR_OP(SwitchStatement)                                           \
  HIR_OP(SwitchCase)                                                \
  CIR_OP(ThrowStatement)                                            \
  HIR_OP(TryStatement)                                              \
  HIR_OP(CatchClause)                                               \
  HIR_OP(WhileStatement)                                            \
  HIR_OP(DoWhileStatement)                                          \
  HIR_OP(ForStatement)                                              \
  HIR_OP(ForInStatement)                                            \
  HIR_OP(ForOfStatement)                                            \
                                                                    \
  /* Declarations */                                                \
  CIR_OP(FunctionDeclaration)                                       \
  CIR_OP(VariableDeclaration)                                       \
  CIR_OP(VariableDeclarator)                                        \
                                                                    \
  /* Directives */                                                  \
  CIR_OP(Directive)                                                 \
  ATTRIB(DirectiveLiteralExtra)                                     \
  CIR_OP(DirectiveLiteral)                                          \
  ATTRIB(InterpreterDirective)                                      \
                                                                    \
  /* Expressions */                                                 \
  CIR_OP(Super)                                                     \
  CIR_OP(Import)                                                    \
  CIR_OP(ThisExpression)                                            \
  CIR_OP(ArrowFunctionExpression)                                   \
  CIR_OP(YieldExpression)                                           \
  CIR_OP(AwaitExpression)                                           \
  CIR_OP(ArrayExpression)                                           \
  CIR_OP(ObjectExpression)                                          \
  CIR_OP(ObjectProperty)                                            \
  REF_OP(ObjectProperty)                                            \
  CIR_OP(ObjectMethod)                                              \
  CIR_OP(FunctionExpression)                                        \
  CIR_OP(UnaryExpression)                                           \
  CIR_OP(UpdateExpression)                                          \
  CIR_OP(BinaryExpression)                                          \
  CIR_OP(AssignmentExpression)                                      \
  HIR_OP(LogicalExpression)                                         \
  CIR_OP(SpreadElement)                                             \
  CIR_OP(MemberExpression)                                          \
  REF_OP(MemberExpression)                                          \
  CIR_OP(OptionalMemberExpression)                                  \
  HIR_OP(ConditionalExpression)                                     \
  CIR_OP(CallExpression)                                            \
  CIR_OP(OptionalCallExpression)                                    \
  CIR_OP(NewExpression)                                             \
  CIR_OP(SequenceExpression)                                        \
  CIR_OP(ParenthesizedExpression)                                   \
  REF_OP(ParenthesizedExpression)                                   \
  CIR_OP(TemplateLiteral)                                           \
  CIR_OP(TaggedTemplateExpression)                                  \
  CIR_OP(TemplateElementValue)                                      \
  CIR_OP(TemplateElement)                                           \
                                                                    \
  /* Patterns */                                                    \
  REF_OP(ObjectPattern)                                             \
  REF_OP(ArrayPattern)                                              \
  REF_OP(RestElement)                                               \
  REF_OP(AssignmentPattern)                                         \
                                                                    \
  /* Classes */                                                     \
  CIR_OP(ClassBody)                                                 \
  CIR_OP(ClassMethod)                                               \
  CIR_OP(ClassPrivateMethod)                                        \
  CIR_OP(ClassProperty)                                             \
  CIR_OP(ClassPrivateProperty)                                      \
  CIR_OP(ClassDeclaration)                                          \
  CIR_OP(ClassExpression)                                           \
                                                                    \
  /* Modules */                                                     \
  CIR_OP(MetaProperty)                                              \
  CIR_OP(ImportDeclaration)                                         \
  ATTRIB(ImportSpecifier)                                           \
  ATTRIB(ImportDefaultSpecifier)                                    \
  ATTRIB(ImportNamespaceSpecifier)                                  \
  ATTRIB(ImportAttribute)                                           \
  CIR_OP(ExportNamedDeclaration)                                    \
  ATTRIB(ExportSpecifier)                                           \
  CIR_OP(ExportDefaultDeclaration)                                  \
  CIR_OP(ExportAllDeclaration)

// JsirStatementOpInterface
#define FOR_EACH_JSIR_STATEMENT_OP(CIR, HIR) \
  CIR(ExpressionStatement)                   \
  HIR(BlockStatement)                        \
  CIR(EmptyStatement)                        \
  CIR(DebuggerStatement)                     \
  HIR(WithStatement)                         \
  CIR(ReturnStatement)                       \
  HIR(LabeledStatement)                      \
  HIR(BreakStatement)                        \
  HIR(ContinueStatement)                     \
  HIR(IfStatement)                           \
  HIR(SwitchStatement)                       \
  CIR(ThrowStatement)                        \
  HIR(TryStatement)                          \
  HIR(WhileStatement)                        \
  HIR(DoWhileStatement)                      \
  HIR(ForStatement)                          \
  HIR(ForInStatement)                        \
  HIR(ForOfStatement)                        \
  CIR(FunctionDeclaration)                   \
  CIR(VariableDeclaration)                   \
  CIR(ClassDeclaration)

// JsirExpressionOpInterface
#define FOR_EACH_JSIR_EXPRESSION_OP(CIR, HIR) \
  CIR(Identifier)                             \
  CIR(RegExpLiteral)                          \
  CIR(NullLiteral)                            \
  CIR(StringLiteral)                          \
  CIR(BooleanLiteral)                         \
  CIR(NumericLiteral)                         \
  CIR(BigIntLiteral)                          \
  CIR(ThisExpression)                         \
  CIR(ArrowFunctionExpression)                \
  CIR(YieldExpression)                        \
  CIR(AwaitExpression)                        \
  CIR(ArrayExpression)                        \
  CIR(ObjectExpression)                       \
  CIR(FunctionExpression)                     \
  CIR(UnaryExpression)                        \
  CIR(UpdateExpression)                       \
  CIR(BinaryExpression)                       \
  CIR(AssignmentExpression)                   \
  HIR(LogicalExpression)                      \
  CIR(MemberExpression)                       \
  CIR(OptionalMemberExpression)               \
  HIR(ConditionalExpression)                  \
  CIR(CallExpression)                         \
  CIR(OptionalCallExpression)                 \
  CIR(NewExpression)                          \
  CIR(SequenceExpression)                     \
  CIR(ParenthesizedExpression)                \
  CIR(TemplateLiteral)                        \
  CIR(TaggedTemplateExpression)               \
  CIR(ClassExpression)                        \
  CIR(MetaProperty)

// JsirLiteralOpInterface
#define FOR_EACH_JSIR_LITERAL_OP(CIR, HIR) \
  CIR(RegExpLiteral)                       \
  CIR(NullLiteral)                         \
  CIR(StringLiteral)                       \
  CIR(BooleanLiteral)                      \
  CIR(NumericLiteral)                      \
  CIR(BigIntLiteral)

// JsirDeclarationOpInterface
#define FOR_EACH_JSIR_DECLARATION_OP(CIR, HIR) \
  CIR(FunctionDeclaration)                     \
  CIR(VariableDeclaration)                     \
  CIR(ClassDeclaration)

// JsirModuleDeclarationOpInterface
#define FOR_EACH_JSIR_MODULE_DECLARATION_OP(CIR, HIR) \
  CIR(ImportDeclaration)                              \
  CIR(ExportNamedDeclaration)                         \
  CIR(ExportDefaultDeclaration)                       \
  CIR(ExportAllDeclaration)

// JsirPatternRefOpInterface
#define FOR_EACH_JSIR_PATTERN_REF_OP(CIR, HIR) \
  CIR(IdentifierRef)                           \
  CIR(MemberExpressionRef)                     \
  CIR(ParenthesizedExpressionRef)              \
  CIR(ObjectPatternRef)                        \
  CIR(ArrayPatternRef)                         \
  CIR(RestElementRef)                          \
  CIR(AssignmentPatternRef)

// JsirLValRefOpInterface
#define FOR_EACH_JSIR_LVAL_REF_OP(CIR, HIR) \
  CIR(IdentifierRef)                        \
  CIR(MemberExpressionRef)                  \
  CIR(ParenthesizedExpressionRef)           \
  CIR(ObjectPatternRef)                     \
  CIR(ArrayPatternRef)                      \
  CIR(RestElementRef)                       \
  CIR(AssignmentPatternRef)

namespace maldoca {

// Checks that region contains a single block. There is no restriction on the
// block. This means that this region can be either of the four kinds below.
bool IsUnknownRegion(mlir::Region &region);

// Checks that the region contains a single block that terminates with
// JsirExprRegionEnd. This means that this region calculates a single
// expression.
bool IsExprRegion(mlir::Region &region);

// Checks that the region contains a single block that terminates with
// JsirExprsRegionEnd. This means that this region calculates a list of
// expressions.
bool IsExprsRegion(mlir::Region &region);

// Checks that the region contains a single block that's non empty. This means
// that this region contains a statement.
bool IsStmtRegion(mlir::Region &region);

// Checks that the region contains a single block. There is no restriction on
// the block. This means that this region contains a list of statements (the
// most unrestrictive case).
bool IsStmtsRegion(mlir::Region &region);

// Converts MLIR attribute to QuickJS value.
std::optional<QjsValue> MlirAttributeToQuickJsValue(JSContext *qjs_context,
                                                    mlir::Attribute val);

// Converts QuickJS value to MLIR attribute.
std::optional<mlir::Attribute> QuickJsValueToMlirAttribute(
    JSContext *qjs_context, mlir::MLIRContext *context, JSValue val);

// Emulates a binary operation in QuickJS.
std::optional<mlir::Attribute> EmulateBinOp(std::string op,
                                            mlir::MLIRContext *context,
                                            mlir::Attribute mlir_left,
                                            mlir::Attribute mlir_right);

// Emulates a binary operation in QuickJS.
QjsValue EmulateBinOp(JSContext *qjs_context, std::string op, QjsValue left,
                      QjsValue right);

// Emulates a Unary operation in QuickJS.
QjsValue EmulateUnaryOp(JSContext *qjs_context, std::string op,
                       QjsValue operand);

}  // namespace maldoca

// Include the auto-generated header file containing the declaration of the JSIR
// dialect.
#include "maldoca/js/ir/jshir_dialect.h.inc"
#include "maldoca/js/ir/jsir_dialect.h.inc"

// Include the auto-generated header file containing the declarations of the
// JSIR interfaces.
#include "maldoca/js/ir/attr_interfaces.h.inc"
#include "maldoca/js/ir/interfaces.h.inc"

// Include the auto-generated header file containing the declarations of the
// JSIR enums.
#include "maldoca/js/ir/jsir_enum_attrs.h.inc"

// Include the auto-generated header file containing the declarations of the
// JSIR attributes.
#define GET_ATTRDEF_CLASSES
#include "maldoca/js/ir/jsir_attrs.h.inc"

// Include the auto-generated header file containing the declarations of the
// JSIR types.
#define GET_TYPEDEF_CLASSES
#include "maldoca/js/ir/jsir_types.h.inc"

// Include the auto-generated header file containing the declarations of the
// JSIR operations.
#define GET_OP_CLASSES
#include "maldoca/js/ir/jsir_ops.h.inc"
#define GET_OP_CLASSES
#include "maldoca/js/ir/jshir_ops.h.inc"

#endif  // MALDOCA_JS_IR_IR_H_
