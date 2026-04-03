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

#ifndef MALDOCA_JS_IR_CONVERSION_AST_TO_JSIR_H_
#define MALDOCA_JS_IR_CONVERSION_AST_TO_JSIR_H_

#include <functional>
#include <variant>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/ir/ir.h"
#include "maldoca/js/ir/trivia.h"

namespace maldoca {

class AstToJsir {
 public:
// Example:
//
// static JsirFileOp VisitFile(mlir::OpBuilder &builder, const JsFile *node);
#define DECLARE_CIR_OP_VISIT_FUNCTION(TYPE)                   \
  static Jsir##TYPE##Op Visit##TYPE(mlir::OpBuilder& builder, \
                                    const Js##TYPE* node);

// Example:
//
// static JshirBlockStatementOp VisitBlockStatement(mlir::OpBuilder &builder,
// const JsBlockStatement *node);
#define DECLARE_HIR_OP_VISIT_FUNCTION(TYPE)                    \
  static Jshir##TYPE##Op Visit##TYPE(mlir::OpBuilder& builder, \
                                     const Js##TYPE* node);

// Example:
//
// static JsirIdentifierRefOp VisitIdentifierRef(mlir::OpBuilder &builder, const
// JsIdentifier *node);
#define DECLARE_REF_OP_VISIT_FUNCTION(TYPE)                           \
  static Jsir##TYPE##RefOp Visit##TYPE##Ref(mlir::OpBuilder& builder, \
                                            const Js##TYPE* node);

// Example:
//
// static JsirIdentifierAttr VisitIdentifierAttr(mlir::OpBuilder &builder, const
// JsIdentifier *node);
#define DECLARE_ATTRIB_VISIT_FUNCTION(TYPE)                           \
  static Jsir##TYPE##Attr Visit##TYPE##Attr(mlir::OpBuilder& builder, \
                                            const Js##TYPE* node);

  FOR_EACH_JSIR_CLASS(DECLARE_CIR_OP_VISIT_FUNCTION,
                      DECLARE_HIR_OP_VISIT_FUNCTION,
                      DECLARE_REF_OP_VISIT_FUNCTION,
                      DECLARE_ATTRIB_VISIT_FUNCTION)

#undef DECLARE_CIR_OP_VISIT_FUNCTION
#undef DECLARE_REF_OP_VISIT_FUNCTION
#undef DECLARE_HIR_OP_VISIT_FUNCTION
#undef DECLARE_ATTRIB_VISIT_FUNCTION

  static JsirProgramBodyElementOpInterface VisitProgramBodyElement(
      mlir::OpBuilder& builder, const JsProgramBodyElement* node);

  static JsirLiteralOpInterface VisitLiteral(mlir::OpBuilder& builder,
                                             const JsLiteral* node);

  static JsirStatementOpInterface VisitStatement(mlir::OpBuilder& builder,
                                                 const JsStatement* node);

  static JsirExpressionOpInterface VisitExpression(mlir::OpBuilder& builder,
                                                   const JsExpression* node);

  static JsirLValRefOpInterface VisitLValRef(mlir::OpBuilder& builder,
                                             const JsLVal* node);

  static JsirDeclarationOpInterface VisitDeclaration(mlir::OpBuilder& builder,
                                                     const JsDeclaration* node);

  static JsirPatternRefOpInterface VisitPatternRef(mlir::OpBuilder& builder,
                                                   const JsPattern* node);

  static JsirModuleSpecifierAttrInterface VisitModuleSpecifierAttr(
      mlir::OpBuilder& builder, const JsModuleSpecifier* node);

  static JsirModuleDeclarationOpInterface VisitModuleDeclaration(
      mlir::OpBuilder& builder, const JsModuleDeclaration* node);

 private:
  static JsirCommentAttrInterface VisitCommentAttr(mlir::OpBuilder& builder,
                                                   const JsComment* node);

  template <typename T, typename... Args>
  static T CreateExpr(mlir::OpBuilder& builder, const JsNode* node,
                      Args&&... args) {
    mlir::MLIRContext* context = builder.getContext();
    mlir::Location trivia_attr =
        (node != nullptr) ? GetJsirTriviaAttr(context, *node)
                          : builder.getUnknownLoc();
    return T::create(builder, trivia_attr,
                     std::forward<Args>(args)...);
  }

  // Overloads `CreateExpr` when the input does not implement `JsNode`.

  template <typename T, typename... Args>
  static T CreateExpr(mlir::OpBuilder& builder,
                      const JsTemplateElementValue* node, Args&&... args) {
    CHECK(node != nullptr) << "Node cannot be null.";
    return T::create(builder, builder.getUnknownLoc(),
                     std::forward<Args>(args)...);
  }

  template <typename T, typename... Args>
  static T CreateStmt(mlir::OpBuilder& builder, const JsNode* node,
                      Args&&... args) {
    mlir::MLIRContext* context = builder.getContext();
    mlir::Location trivia_attr =
        (node != nullptr) ? GetJsirTriviaAttr(context, *node)
                          : builder.getUnknownLoc();
    return T::create(builder, trivia_attr,
                     mlir::TypeRange(), std::forward<Args>(args)...);
  }

  static void AppendNewBlockAndPopulate(mlir::OpBuilder& builder,
                                        mlir::Region& region,
                                        std::function<void()> populate);

  // The key of an object property.
  //
  // Example:
  // {
  //   a: 0
  //   ~
  //
  //   "b": 1
  //   ~~~
  //
  //   ["b"]: 2
  //   ~~~~~
  // }
  //
  // The key can be either literal or computed. Therefore, only one of them is
  // non-null.
  struct ObjectPropertyKey {
    // JsirIdentifierAttr | JsirStringLiteralAttr | JsirNumericLiteralAttr
    //                    | JsirBigIntLiteralAttr
    mlir::Attribute literal;

    // JsirExpressionOpInterface
    mlir::Value computed;
  };

  // If computed == false:
  //   ObjectPropertyKey::literal is non-null.
  //   ObjectPropertyKey::computed is null.
  // If computed == true:
  //   ObjectPropertyKey::literal is null.
  //   ObjectPropertyKey::computed is non-null.
  static ObjectPropertyKey GetObjectPropertyKey(mlir::OpBuilder& builder,
                                                const JsExpression* node,
                                                bool computed);

  static mlir::Value VisitMemberExpressionObject(
      mlir::OpBuilder& builder,
      std::variant<const JsExpression*, const JsSuper*> object);

  struct MemberExpressionProperty {
    mlir::Attribute literal;
    mlir::Value computed;
  };

  static MemberExpressionProperty VisitMemberExpressionProperty(
      mlir::OpBuilder& builder,
      std::variant<const JsExpression*, const JsPrivateName*> property,
      bool computed);
};

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_CONVERSION_AST_TO_JSIR_H_
