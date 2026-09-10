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

#ifndef MALDOCA_ASTGEN_AST_VISITOR_HEADER_PRINTER_H_
#define MALDOCA_ASTGEN_AST_VISITOR_HEADER_PRINTER_H_

#include <string>

#include "absl/strings/string_view.h"
#include "maldoca/astgen/ast_def.h"
#include "maldoca/astgen/cc_printer_base.h"
#include "google/protobuf/io/zero_copy_stream.h"

namespace maldoca {

// Printer of the C++ header for the AST visitor ("ast_visitor.generated.h").
class AstVisitorHeaderPrinter : public CcPrinterBase {
 public:
    explicit AstVisitorHeaderPrinter(google::protobuf::io::ZeroCopyOutputStream* os)
        : CcPrinterBase(os) {}

  // Prints the "ast_visitor.generated.h" header file.
  //
  // - cc_namespace: The C++ namespace for all the AST node classes.
  //   Example: "maldoca::astgen".
  //
  // - ast_path: The directory for the AST code.
  //   "ast_visitor.generated.h" is in that directory.
  //   This is used to generate the header guard and include paths.
  //
  // See test cases in test/ for examples.
  void PrintAstVisitor(const AstDef& ast, absl::string_view cc_namespace,
                       absl::string_view ast_path);

  // Prints base visitor interfaces for intermediate nodes with multiple
  // inheritance (e.g. Pattern, LVal).
  //
  // Format:
  //  template <typename R>
  //  class <Lang><Node>Visitor {
  //   public:
  //    virtual ~<Lang><Node>Visitor() = default;
  //    R Visit<Node>(const <Lang><Node> &<node>);
  //    virtual R Visit<Leaf1>(const <Lang><Leaf1> &<leaf1>) = 0;
  //  };
  //
  // Example:
  //  template <typename R>
  //  class JsPatternVisitor {
  //   public:
  //    virtual ~JsPatternVisitor() = default;
  //    R VisitPattern(const JsPattern &pattern);
  //    virtual R VisitIdentifier(const JsIdentifier &identifier) = 0;
  //  };
  void PrintBaseVisitor(const NodeDef& node, absl::string_view lang_name,
                        bool is_mutable);

  // Prints the core AST visitor class (JsAstVisitor or MutableJsAstVisitor).
  //
  // Format:
  //  template <typename R>
  //  class <Lang>AstVisitor : public virtual <Lang>PatternVisitor<R>,
  //                           public virtual <Lang>LValVisitor<R> {
  //   public:
  //    virtual ~<Lang>AstVisitor() = default;
  //    virtual R Visit<LeafNode>(const <Lang><LeafNode> &<leaf_node>) = 0;
  //    R Visit<NonLeafNode>(const <Lang><NonLeafNode> &<non_leaf_node>);
  //  };
  //
  // Example:
  //  template <typename R>
  //  class JsAstVisitor : public virtual JsPatternVisitor<R>,
  //                       public virtual JsLValVisitor<R> {
  //   public:
  //    virtual ~JsAstVisitor() = default;
  //    virtual R VisitBinaryExpression(
  //        const JsBinaryExpression &binary_expression) = 0;
  //    R VisitExpression(const JsExpression &expression);
  //  };
  void PrintAstVisitorClass(const AstDef& ast, bool is_mutable);

  // Prints dynamic_cast dispatch for a non-leaf node (e.g. VisitExpression).
  //
  // Format:
  //  R Visit<NonLeaf>(const <Lang><NonLeaf> &<non_leaf>) {
  //    const <Lang><NonLeaf> *<non_leaf>_ptr = &<non_leaf>;
  //    if (const <Lang><Child1> *<child1> =
  //            dynamic_cast<const <Lang><Child1> *>(<non_leaf>_ptr)) {
  //      return Visit<Child1>(*<child1>);
  //    }
  //    LOG(FATAL) << "Unreachable code.";
  //  }
  //
  // Example:
  //  R VisitLiteral(const JsLiteral &literal) {
  //    const JsLiteral *literal_ptr = &literal;
  //    if (const JsRegExpLiteral *reg_exp_literal =
  //            dynamic_cast<const JsRegExpLiteral *>(literal_ptr)) {
  //      return VisitRegExpLiteral(*reg_exp_literal);
  //    }
  //    LOG(FATAL) << "Unreachable code.";
  //  }
  void PrintNonLeafDispatch(const NodeDef& node, absl::string_view lang_name,
                            bool is_mutable);

  // Prints Empty AST visitor (EmptyJsAstVisitor, EmptyMutableJsAstVisitor).
  //
  // Format:
  //  class Empty<Lang>AstVisitor : public <Lang>AstVisitor<void> {
  //   public:
  //    ~Empty<Lang>AstVisitor() override = default;
  //    void Visit<LeafNode>(const <Lang><LeafNode> &<leaf_node>) override {}
  //  };
  //
  // Example:
  //  class EmptyJsAstVisitor : public JsAstVisitor<void> {
  //   public:
  //    ~EmptyJsAstVisitor() override = default;
  //    void VisitBinaryExpression(
  //        const JsBinaryExpression &binary_expression) override {}
  //  };
  void PrintEmptyVisitor(const AstDef& ast, bool is_mutable);

  // Prints Default AST visitor (DefaultJsAstVisitor,
  // DefaultMutableJsAstVisitor).
  //
  // Format:
  //  class Default<Lang>AstVisitor : public Empty<Lang>AstVisitor {
  //   public:
  //    ~Default<Lang>AstVisitor() override = default;
  //    void Visit<LeafNode>(const <Lang><LeafNode> &<leaf_node>) override {
  //      this->VisitNodeDefault(<leaf_node>);
  //    }
  //    virtual void VisitNodeDefault(const <Lang>Node &node) = 0;
  //  };
  //
  // Example:
  //  class DefaultJsAstVisitor : public EmptyJsAstVisitor {
  //   public:
  //    ~DefaultJsAstVisitor() override = default;
  //    void VisitBinaryExpression(
  //        const JsBinaryExpression &binary_expression) override {
  //      this->VisitNodeDefault(binary_expression);
  //    }
  //    virtual void VisitNodeDefault(const JsNode &node) = 0;
  //  };
  void PrintDefaultVisitor(const AstDef& ast, bool is_mutable);

  // Prints lambda wrapper classes for default AST visitors.
  //
  // Format:
  //  class Default<Lang>AstVisitorLambdaWrapper :
  //      public Default<Lang>AstVisitor {
  //   public:
  //    explicit Default<Lang>AstVisitorLambdaWrapper(
  //        std::function<void(const <Lang>Node &)> func)
  //        : func_(std::move(func)) {}
  //    void VisitNodeDefault(const <Lang>Node &node) override { func_(node); }
  //   private:
  //    std::function<void(const <Lang>Node &)> func_;
  //  };
  //
  // Example:
  //  class DefaultJsAstVisitorLambdaWrapper :
  //      public DefaultJsAstVisitor {
  //   public:
  //    explicit DefaultJsAstVisitorLambdaWrapper(
  //        std::function<void(const JsNode &)> func)
  //        : func_(std::move(func)) {}
  //    void VisitNodeDefault(const JsNode &node) override { func_(node); }
  //   private:
  //    std::function<void(const JsNode &)> func_;
  //  };
  void PrintLambdaWrappers(absl::string_view lang_name);
};

// Prints the "ast_visitor.generated.h" header file.
//
// - cc_namespace: The C++ namespace for all the AST node classes.
//   Example: "maldoca::astgen".
//
// - ast_path: The directory for the AST code.
//   "ast_visitor.generated.h" is in that directory.
//   This is used to generate the header guard.
std::string PrintAstVisitorHeader(const AstDef& ast,
                                 absl::string_view cc_namespace,
                                 absl::string_view ast_path);

}  // namespace maldoca

#endif  // MALDOCA_ASTGEN_AST_VISITOR_HEADER_PRINTER_H_
