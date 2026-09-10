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

#include "maldoca/astgen/ast_visitor_header_printer.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "maldoca/astgen/ast_def.h"
#include "maldoca/astgen/ast_gen_utils.h"
#include "maldoca/astgen/symbol.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"

namespace maldoca {
namespace {

bool IsAstNode(const NodeDef& node) {
  if (node.name() == "Node") return true;
  return absl::c_any_of(node.ancestors(), [](const NodeDef* ancestor) {
    return ancestor->name() == "Node";
  });
}

bool IsLeafAstNode(const NodeDef& node) {
  return IsAstNode(node) && node.type().has_value();
}

bool IsNonLeafAstNode(const NodeDef& node) {
  return IsAstNode(node) && !node.type().has_value() && node.name() != "Node";
}

bool IsInBaseVisitor(const NodeDef& node) {
  return absl::c_any_of(node.ancestors(), [](const NodeDef* ancestor) {
    return ancestor->name() == "Pattern" || ancestor->name() == "LVal";
  });
}

}  // namespace

void AstVisitorHeaderPrinter::PrintAstVisitor(const AstDef& ast,
                                             absl::string_view cc_namespace,
                                             absl::string_view ast_path) {
  auto header_path = GetAstVisitorHeaderPath(ast_path);

  PrintLicense();
  Println();

  PrintCodeGenerationWarning();
  Println();

  PrintEnterHeaderGuard(header_path);
  Println();

  Println("// IWYU pragma: begin_keep");
  Println("// NOLINTBEGIN(whitespace/line_length)");
  Println("// NOLINTBEGIN(google3-readability-absl-macros)");
  Println("// clang-format off");
  Println();

  Println("#include <functional>");
  Println("#include <utility>");
  Println();

  PrintIncludeHeader("absl/log/log.h");
  PrintIncludeHeader(GetAstHeaderPath(ast_path));
  Println();

  PrintEnterNamespace(cc_namespace);
  Println();

  // 1. Multiple-inheritance base visitors (e.g. JsPatternVisitor,
  //    JsLValVisitor)
  for (const NodeDef* node : ast.topological_sorted_nodes()) {
    if (node->name() == "Pattern" || node->name() == "LVal") {
      PrintBaseVisitor(*node, ast.lang_name(), /*is_mutable=*/false);
      Println();
      PrintBaseVisitor(*node, ast.lang_name(), /*is_mutable=*/true);
      Println();
    }
  }

  // 2. Const AstVisitor
  PrintAstVisitorClass(ast, /*is_mutable=*/false);
  Println();

  // 3. Mutable AstVisitor
  PrintAstVisitorClass(ast, /*is_mutable=*/true);
  Println();

  // 4. Empty Visitors
  PrintEmptyVisitor(ast, /*is_mutable=*/false);
  Println();
  PrintEmptyVisitor(ast, /*is_mutable=*/true);
  Println();

  // 5. Default Visitors
  PrintDefaultVisitor(ast, /*is_mutable=*/false);
  Println();
  PrintDefaultVisitor(ast, /*is_mutable=*/true);
  Println();

  // 6. Lambda Wrappers
  PrintLambdaWrappers(ast.lang_name());
  Println();

  Println("// clang-format on");
  Println("// NOLINTEND(google3-readability-absl-macros)");
  Println("// NOLINTEND(whitespace/line_length)");
  Println("// IWYU pragma: end_keep");
  Println();

  PrintExitNamespace(cc_namespace);
  Println();

  PrintExitHeaderGuard(header_path);
}

void AstVisitorHeaderPrinter::PrintBaseVisitor(const NodeDef& node,
                                              absl::string_view lang_name,
                                              bool is_mutable) {
  std::string lang = (Symbol(lang_name)).ToPascalCase();
  std::string class_name =
      is_mutable ? absl::StrCat("Mutable", lang, node.name(), "Visitor")
                 : absl::StrCat(lang, node.name(), "Visitor");
  std::string node_cc_name = (Symbol(lang_name) + node.name()).ToPascalCase();
  std::string node_var = Symbol(node.name()).ToCcVarName();
  std::string const_qual = is_mutable ? "" : "const ";
  std::string ref_type = absl::StrCat(const_qual, node_cc_name, " &");
  std::string ptr_type = absl::StrCat(const_qual, node_cc_name, " *");

  auto vars = WithVars({
      {"ClassName", class_name},
      {"NodeName", node.name()},
      {"RefType", ref_type},
      {"PtrType", ptr_type},
      {"node_var", node_var},
  });

  Println("template <typename R>");
  Println("class $ClassName$ {");
  Println(" public:");
  {
    auto indent = WithIndent();
    Println("virtual ~$ClassName$() = default;");
    Println();

    Println("R Visit$NodeName$($RefType$$node_var$) {");
    {
      auto indent2 = WithIndent();
      Println("$PtrType$$node_var$_ptr = &$node_var$;");

      if (!node.leaves().empty()) {
        bool first = true;
        for (const NodeDef* leaf : node.leaves()) {
          std::string leaf_cc_name =
              (Symbol(lang_name) + leaf->name()).ToPascalCase();
          std::string leaf_var = Symbol(leaf->name()).ToCcVarName();
          std::string leaf_ptr_type =
              absl::StrCat(const_qual, leaf_cc_name, " *");

          std::string if_keyword = first ? "if" : "} else if";
          first = false;

          auto leaf_vars = WithVars({
              {"if_keyword", if_keyword},
              {"LeafPtrType", leaf_ptr_type},
              {"leaf_var", leaf_var},
              {"LeafName", leaf->name()},
          });

          Println(
              "$if_keyword$ ($LeafPtrType$$leaf_var$ = "
              "dynamic_cast<$LeafPtrType$>($node_var$_ptr)) {");
          Println("  return Visit$LeafName$(*$leaf_var$);");
        }
        Println("}");
        Println();
      }
      Println("LOG(FATAL) << \"Unreachable code.\";");
    }
    Println("}");

    if (!node.leaves().empty()) {
      Println();
      for (const NodeDef* leaf : node.leaves()) {
        std::string leaf_cc_name =
            (Symbol(lang_name) + leaf->name()).ToPascalCase();
        std::string leaf_var = Symbol(leaf->name()).ToCcVarName();
        std::string leaf_ref_type =
            absl::StrCat(const_qual, leaf_cc_name, " &");

        auto leaf_vars = WithVars({
            {"LeafName", leaf->name()},
            {"LeafRefType", leaf_ref_type},
            {"leaf_var", leaf_var},
        });

        Println("virtual R Visit$LeafName$($LeafRefType$$leaf_var$) = 0;");
      }
    }
  }
  Println("};");
}

void AstVisitorHeaderPrinter::PrintAstVisitorClass(const AstDef& ast,
                                                  bool is_mutable) {
  std::string lang = (Symbol(ast.lang_name())).ToPascalCase();
  std::string class_name =
      is_mutable ? absl::StrCat("Mutable", lang, "AstVisitor")
                 : absl::StrCat(lang, "AstVisitor");
  std::vector<std::string> base_visitors;
  for (const NodeDef* node : ast.topological_sorted_nodes()) {
    if (node->name() == "Pattern" || node->name() == "LVal") {
      base_visitors.push_back(
          is_mutable ? absl::StrCat("public virtual Mutable", lang,
                                    node->name(), "Visitor<R>")
                     : absl::StrCat("public virtual ", lang,
                                    node->name(), "Visitor<R>"));
    }
  }

  auto vars = WithVars({
      {"ClassName", class_name},
  });

  Println("template <typename R>");
  if (base_visitors.empty()) {
    Println("class $ClassName$ {");
  } else {
    auto base_vars = WithVars({
        {"BaseVisitors",
         absl::StrJoin(base_visitors, ",\n                     ")},
    });
    Println("class $ClassName$ : $BaseVisitors$ {");
  }
  Println(" public:");
  {
    auto indent = WithIndent();
    if (base_visitors.empty()) {
      Println("virtual ~$ClassName$() = default;");
    } else {
      Println("~$ClassName$() override = default;");
    }
    Println();

    // 1. Pure virtual Visit<LeafNode> declarations for all leaf nodes
    for (const NodeDef* node : ast.topological_sorted_nodes()) {
      if (!IsLeafAstNode(*node)) continue;

      std::string node_cc_name =
          (Symbol(ast.lang_name()) + node->name()).ToPascalCase();
      std::string node_var = Symbol(node->name()).ToCcVarName();
      std::string const_qual = is_mutable ? "" : "const ";
      std::string ref_type = absl::StrCat(const_qual, node_cc_name, " &");

      auto leaf_vars = WithVars({
          {"NodeName", node->name()},
          {"RefType", ref_type},
          {"node_var", node_var},
      });

      if (IsInBaseVisitor(*node)) {
        Println("R Visit$NodeName$($RefType$$node_var$) override = 0;");
      } else {
        Println("virtual R Visit$NodeName$($RefType$$node_var$) = 0;");
      }
      Println();
    }

    // 2. Concrete dynamic_cast dispatch for all non-leaf nodes
    // Collect non-leaf nodes and sort them so derived/children come before
    // base/parents.
    std::vector<const NodeDef*> non_leaf_nodes;
    for (const NodeDef* node : ast.topological_sorted_nodes()) {
      if (!IsNonLeafAstNode(*node)) continue;
      if (node->name() == "Pattern" || node->name() == "LVal") {
        continue;
      }
      non_leaf_nodes.push_back(node);
    }
    std::reverse(non_leaf_nodes.begin(), non_leaf_nodes.end());

    for (size_t i = 0; i < non_leaf_nodes.size(); ++i) {
      if (i > 0) {
        Println();
      }
      PrintNonLeafDispatch(*non_leaf_nodes[i], ast.lang_name(), is_mutable);
    }
  }
  Println("};");
}

void AstVisitorHeaderPrinter::PrintNonLeafDispatch(const NodeDef& node,
                                                  absl::string_view lang_name,
                                                  bool is_mutable) {
  std::string node_cc_name = (Symbol(lang_name) + node.name()).ToPascalCase();
  std::string node_var = Symbol(node.name()).ToCcVarName();
  std::string const_qual = is_mutable ? "" : "const ";
  std::string ref_type = absl::StrCat(const_qual, node_cc_name, " &");
  std::string ptr_type = absl::StrCat(const_qual, node_cc_name, " *");

  std::string virtual_kw = (node.name() == "Statement") ? "virtual " : "";

  auto vars = WithVars({
      {"virtual_kw", virtual_kw},
      {"NodeName", node.name()},
      {"RefType", ref_type},
      {"PtrType", ptr_type},
      {"node_var", node_var},
  });

  Println("$virtual_kw$R Visit$NodeName$($RefType$$node_var$) {");
  {
    auto indent = WithIndent();
    Println("$PtrType$$node_var$_ptr = &$node_var$;");

    if (!node.children().empty()) {
      bool first = true;
      for (const NodeDef* child : node.children()) {
        std::string child_cc_name =
            (Symbol(lang_name) + child->name()).ToPascalCase();
        std::string child_var = Symbol(child->name()).ToCcVarName();
        std::string child_ptr_type =
            absl::StrCat(const_qual, child_cc_name, " *");

        std::string if_keyword = first ? "if" : "} else if";
        first = false;

        auto child_vars = WithVars({
            {"if_keyword", if_keyword},
            {"ChildPtrType", child_ptr_type},
            {"child_var", child_var},
            {"ChildName", child->name()},
        });

        Println(
            "$if_keyword$ ($ChildPtrType$$child_var$ = "
            "dynamic_cast<$ChildPtrType$>($node_var$_ptr)) {");
        Println("  return Visit$ChildName$(*$child_var$);");
      }
      Println("}");
      Println();
    }
    Println("LOG(FATAL) << \"Unreachable code.\";");
  }
  Println("}");
}

void AstVisitorHeaderPrinter::PrintEmptyVisitor(const AstDef& ast,
                                               bool is_mutable) {
  std::string lang = (Symbol(ast.lang_name())).ToPascalCase();
  std::string class_name =
      is_mutable ? absl::StrCat("EmptyMutable", lang, "AstVisitor")
                 : absl::StrCat("Empty", lang, "AstVisitor");
  std::string base_name =
      is_mutable ? absl::StrCat("Mutable", lang, "AstVisitor<void>")
                 : absl::StrCat(lang, "AstVisitor<void>");

  auto vars = WithVars({
      {"ClassName", class_name},
      {"BaseName", base_name},
  });

  Println("class $ClassName$ : public $BaseName$ {");
  Println(" public:");
  {
    auto indent = WithIndent();
    Println("~$ClassName$() override = default;");

    for (const NodeDef* node : ast.topological_sorted_nodes()) {
      if (!IsLeafAstNode(*node)) continue;

      std::string node_cc_name =
          (Symbol(ast.lang_name()) + node->name()).ToPascalCase();
      std::string node_var = Symbol(node->name()).ToCcVarName();
      std::string const_qual = is_mutable ? "" : "const ";
      std::string ref_type = absl::StrCat(const_qual, node_cc_name, " &");

      auto leaf_vars = WithVars({
          {"NodeName", node->name()},
          {"RefType", ref_type},
          {"node_var", node_var},
      });

      Println();
      Println("void Visit$NodeName$($RefType$$node_var$) override {}");
    }
  }
  Println("};");
}

void AstVisitorHeaderPrinter::PrintDefaultVisitor(const AstDef& ast,
                                                 bool is_mutable) {
  std::string lang = (Symbol(ast.lang_name())).ToPascalCase();
  std::string class_name =
      is_mutable ? absl::StrCat("DefaultMutable", lang, "AstVisitor")
                 : absl::StrCat("Default", lang, "AstVisitor");
  std::string base_name =
      is_mutable ? absl::StrCat("EmptyMutable", lang, "AstVisitor")
                 : absl::StrCat("Empty", lang, "AstVisitor");
  std::string node_base = is_mutable ? absl::StrCat(lang, "Node &")
                                     : absl::StrCat("const ", lang, "Node &");

  auto vars = WithVars({
      {"ClassName", class_name},
      {"BaseName", base_name},
      {"NodeBase", node_base},
  });

  Println("class $ClassName$ : public $BaseName$ {");
  Println(" public:");
  {
    auto indent = WithIndent();
    Println("~$ClassName$() override = default;");

    for (const NodeDef* node : ast.topological_sorted_nodes()) {
      if (!IsLeafAstNode(*node)) continue;

      std::string node_cc_name =
          (Symbol(ast.lang_name()) + node->name()).ToPascalCase();
      std::string node_var = Symbol(node->name()).ToCcVarName();
      std::string const_qual = is_mutable ? "" : "const ";
      std::string ref_type = absl::StrCat(const_qual, node_cc_name, " &");

      auto leaf_vars = WithVars({
          {"NodeName", node->name()},
          {"RefType", ref_type},
          {"node_var", node_var},
      });

      Println();
      Println("void Visit$NodeName$($RefType$$node_var$) override {");
      Println("  this->VisitNodeDefault($node_var$);");
      Println("}");
    }

    Println();
    Println(
        "// Override this method to customize the default behavior on each "
        "AST node.");
    Println("virtual void VisitNodeDefault($NodeBase$node) = 0;");
  }
  Println("};");
}

void AstVisitorHeaderPrinter::PrintLambdaWrappers(absl::string_view lang_name) {
  std::string lang = (Symbol(lang_name)).ToPascalCase();

  auto vars = WithVars({
      {"Lang", lang},
  });

  // 1. Const Lambda Wrapper
  Println(
      "class Default$Lang$AstVisitorLambdaWrapper : public "
      "Default$Lang$AstVisitor {");
  Println(" public:");
  {
    auto indent = WithIndent();
    Println("explicit Default$Lang$AstVisitorLambdaWrapper(");
    Println("    std::function<void(const $Lang$Node &)> func)");
    Println("    : func_(std::move(func)) {}");
    Println();
    Println(
        "void VisitNodeDefault(const $Lang$Node &node) override { "
        "func_(node); }");
  }
  Println();
  Println(" private:");
  {
    auto indent = WithIndent();
    Println("std::function<void(const $Lang$Node &)> func_;");
  }
  Println("};");
  Println();

  // 2. Mutable Lambda Wrapper
  Println(
      "class DefaultMutable$Lang$AstVisitorLambdaWrapper : public "
      "DefaultMutable$Lang$AstVisitor {");
  Println(" public:");
  {
    auto indent = WithIndent();
    Println("explicit DefaultMutable$Lang$AstVisitorLambdaWrapper(");
    Println("    std::function<void($Lang$Node &)> func)");
    Println("    : func_(std::move(func)) {}");
    Println();
    Println(
        "void VisitNodeDefault($Lang$Node &node) override { func_(node); }");
  }
  Println();
  Println(" private:");
  {
    auto indent = WithIndent();
    Println("std::function<void($Lang$Node &)> func_;");
  }
  Println("};");
}

std::string PrintAstVisitorHeader(const AstDef& ast,
                                 absl::string_view cc_namespace,
                                 absl::string_view ast_path) {
  std::string str;
  {
    google::protobuf::io::StringOutputStream os(&str);
    AstVisitorHeaderPrinter printer(&os);
    printer.PrintAstVisitor(ast, cc_namespace, ast_path);
  }
  return str;
}

}  // namespace maldoca
