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

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

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
  explicit AstToJsir(mlir::OpBuilder &builder) : builder_(builder) {}

// Example:
//
// JsirFileOp VisitFile(const JsFile *node);
#define DECLARE_CIR_OP_VISIT_FUNCTION(TYPE) \
  Jsir##TYPE##Op Visit##TYPE(const Js##TYPE *node);

// Example:
//
// JshirBlockStatementOp VisitBlockStatement(const JsBlockStatement *node);
#define DECLARE_HIR_OP_VISIT_FUNCTION(TYPE) \
  Jshir##TYPE##Op Visit##TYPE(const Js##TYPE *node);

// Example:
//
// JsirIdentifierRefOp VisitIdentifierRef(const JsIdentifier *node);
#define DECLARE_REF_OP_VISIT_FUNCTION(TYPE) \
  Jsir##TYPE##RefOp Visit##TYPE##Ref(const Js##TYPE *node);

// Example:
//
// JsirIdentifierAttr VisitIdentifierAttr(const JsIdentifier *node);
#define DECLARE_ATTRIB_VISIT_FUNCTION(TYPE) \
  Jsir##TYPE##Attr Visit##TYPE##Attr(const Js##TYPE *node);

  FOR_EACH_JSIR_CLASS(DECLARE_CIR_OP_VISIT_FUNCTION,
                      DECLARE_HIR_OP_VISIT_FUNCTION,
                      /*LIR_OP=*/JSIR_CLASS_IGNORE,
                      DECLARE_REF_OP_VISIT_FUNCTION,
                      DECLARE_ATTRIB_VISIT_FUNCTION)

#undef DECLARE_CIR_OP_VISIT_FUNCTION
#undef DECLARE_REF_OP_VISIT_FUNCTION
#undef DECLARE_HIR_OP_VISIT_FUNCTION
#undef DECLARE_ATTRIB_VISIT_FUNCTION

  JsirLiteralOpInterface VisitLiteral(const JsLiteral *node);

  JsirStatementOpInterface VisitStatement(const JsStatement *node);

  JsirExpressionOpInterface VisitExpression(const JsExpression *node);

  JsirLValRefOpInterface VisitLValRef(const JsLVal *node);

  JsirDeclarationOpInterface VisitDeclaration(const JsDeclaration *node);

  JsirPatternRefOpInterface VisitPatternRef(const JsPattern *node);

  JsirModuleSpecifierAttrInterface VisitModuleSpecifierAttr(
      const JsModuleSpecifier *node);

  JsirModuleDeclarationOpInterface VisitModuleDeclaration(
      const JsModuleDeclaration *node);

 private:
  JsirCommentAttrInterface VisitCommentAttr(const JsComment *node);

  template <typename T, typename... Args>
  T CreateExpr(const JsNode *node, Args &&...args) {
    CHECK(node != nullptr) << "Node cannot be null.";
    mlir::MLIRContext *context = builder_.getContext();
    return builder_.create<T>(GetJsirTriviaAttr(context, *node),
                              std::forward<Args>(args)...);
  }

  // Overloads `CreateExpr` when the input does not implement `JsNode`.

  template <typename T, typename... Args>
  T CreateExpr(const JsTemplateElementValue *node, Args &&...args) {
    CHECK(node != nullptr) << "Node cannot be null.";
    return builder_.create<T>(builder_.getUnknownLoc(),
                              std::forward<Args>(args)...);
  }

  template <typename T, typename... Args>
  T CreateStmt(const JsNode *node, Args &&...args) {
    CHECK(node != nullptr) << "Node cannot be null.";
    mlir::MLIRContext *context = builder_.getContext();
    return builder_.create<T>(GetJsirTriviaAttr(context, *node), std::nullopt,
                              std::forward<Args>(args)...);
  }

  void AppendNewBlockAndPopulate(mlir::Region &region,
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
  ObjectPropertyKey GetObjectPropertyKey(const JsExpression *node,
                                         bool computed);

  mlir::Value VisitMemberExpressionObject(
      std::variant<const JsExpression *, const JsSuper *> object);

  struct MemberExpressionProperty {
    mlir::Attribute literal;
    mlir::Value computed;
  };

  MemberExpressionProperty VisitMemberExpressionProperty(
      std::variant<const JsExpression *, const JsPrivateName *> property,
      bool computed);

  mlir::OpBuilder &builder_;
};

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_CONVERSION_AST_TO_JSIR_H_
