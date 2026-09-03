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

#ifndef MALDOCA_ASTGEN_AST_WALKER_HEADER_PRINTER_H_
#define MALDOCA_ASTGEN_AST_WALKER_HEADER_PRINTER_H_

#include <string>

#include "absl/strings/string_view.h"
#include "maldoca/astgen/ast_def.h"
#include "maldoca/astgen/cc_printer_base.h"
#include "google/protobuf/io/zero_copy_stream.h"

namespace maldoca {

// Printer of the C++ header for the AST walker ("ast_walker.generated.h").
class AstWalkerHeaderPrinter : public CcPrinterBase {
 public:
  explicit AstWalkerHeaderPrinter(google::protobuf::io::ZeroCopyOutputStream* os)
      : CcPrinterBase(os) {}

  // Prints the "ast_walker.generated.h" header file.
  //
  // - cc_namespace: The C++ namespace for all the AST node classes.
  //   Example: "maldoca::astgen".
  //
  // - ast_path: The directory for the AST code.
  //   "ast_walker.generated.h" is in that directory.
  //   This is used to generate the header guard and include paths.
  //
  // See test cases in test/ for examples.
  void PrintAstWalker(const AstDef& ast, absl::string_view cc_namespace,
                      absl::string_view ast_path);

  // Prints a walker class (JsAstWalker or MutableJsAstWalker).
  //
  // Format:
  //  class <Lang>AstWalker : public <Lang>AstVisitor<void> {
  //   public:
  //    explicit <Lang>AstWalker(<Lang>AstVisitor<void> *preorder_callback,
  //                             <Lang>AstVisitor<void> *postorder_callback)
  //        : preorder_callback_(preorder_callback),
  //          postorder_callback_(postorder_callback) {}
  //    void Visit<Leaf1>(const <Lang><Leaf1> &<leaf1>) override;
  //    ...
  //  };
  //
  // Example:
  //  class JsAstWalker : public JsAstVisitor<void> {
  //   public:
  //    explicit JsAstWalker(JsAstVisitor<void> *preorder_callback,
  //                         JsAstVisitor<void> *postorder_callback)
  //        : preorder_callback_(preorder_callback),
  //          postorder_callback_(postorder_callback) {}
  //    void VisitBinaryExpression(
  //        const JsBinaryExpression &binary_expression) override;
  //  };
  void PrintAstWalkerClass(const AstDef& ast, bool is_mutable);

  // Prints the Visit method for a concrete leaf node.
  //
  // Format:
  //  void Visit<Node>(const <Lang><Node> &<node>) override {
  //    if (preorder_callback_) {
  //      preorder_callback_->Visit<Node>(<node>);
  //    }
  //    <child_field_traversals>
  //    if (postorder_callback_) {
  //      postorder_callback_->Visit<Node>(<node>);
  //    }
  //  }
  //
  // Example:
  //  void VisitExpressionStatement(
  //      const JsExpressionStatement &expression_statement) override {
  //    if (preorder_callback_) {
  //      preorder_callback_->VisitExpressionStatement(expression_statement);
  //    }
  //    VisitExpression(*expression_statement.expression());
  //    if (postorder_callback_) {
  //      postorder_callback_->VisitExpressionStatement(expression_statement);
  //    }
  //  }
  void PrintVisitMethod(const NodeDef& node, const AstDef& ast,
                        bool is_mutable);

  // Prints the recursive child traversal for a specific field.
  //
  // Format:
  //  Visit<ChildNode>(*<node>.<field>());
  //
  // Example:
  //  VisitExpression(*expression_statement.expression());
  void PrintFieldTraversal(const FieldDef& field, absl::string_view node_var,
                           const AstDef& ast, bool is_mutable);
};

// Prints the "ast_walker.generated.h" header file.
//
// - cc_namespace: The C++ namespace for all the AST node classes.
//   Example: "maldoca::astgen".
//
// - ast_path: The directory for the AST code.
//   "ast_walker.generated.h" is in that directory.
//   This is used to generate the header guard.
std::string PrintAstWalkerHeader(const AstDef& ast,
                                absl::string_view cc_namespace,
                                absl::string_view ast_path);

}  // namespace maldoca

#endif  // MALDOCA_ASTGEN_AST_WALKER_HEADER_PRINTER_H_
