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

#include "maldoca/astgen/ast_walker_header_printer.h"

#include <cstddef>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "maldoca/astgen/ast_def.h"
#include "maldoca/astgen/ast_gen_utils.h"
#include "maldoca/astgen/symbol.h"
#include "maldoca/astgen/type.h"
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

const NodeDef* GetAstNodeDef(const Type& type, const AstDef& ast) {
  if (type.IsA<ClassType>()) {
    const auto& class_type = static_cast<const ClassType&>(type);
    auto it = ast.nodes().find(class_type.name().ToPascalCase());
    if (it != ast.nodes().end() && IsAstNode(*it->second)) {
      return it->second.get();
    }
  }
  return nullptr;
}

bool HasAstNodes(const VariantType& variant_type, const AstDef& ast) {
  return absl::c_any_of(variant_type.types(), [&](const auto& t) {
    return GetAstNodeDef(*t, ast) != nullptr;
  });
}

std::vector<const FieldDef*> GetOrderedFieldsForWalker(const NodeDef& node) {
  std::vector<const FieldDef*> fields(node.aggregated_fields().begin(),
                                      node.aggregated_fields().end());
  if (node.name() == "ClassDeclaration" || node.name() == "ClassExpression") {
    auto it = absl::c_find_if(fields, [](const FieldDef* f) {
      return f->name().ToCcVarName() == "id";
    });
    if (it != fields.end()) {
      const FieldDef* id_field = *it;
      fields.erase(it);
      fields.insert(fields.begin(), id_field);
    }
  } else if (node.name() == "ClassMethod" ||
             node.name() == "ClassPrivateMethod") {
    auto it = absl::c_find_if(fields, [](const FieldDef* f) {
      return f->name().ToCcVarName() == "key";
    });
    if (it != fields.end()) {
      const FieldDef* key_field = *it;
      fields.erase(it);
      fields.insert(fields.begin(), key_field);
    }
  }
  return fields;
}

}  // namespace

void AstWalkerHeaderPrinter::PrintAstWalker(const AstDef& ast,
                                            absl::string_view cc_namespace,
                                            absl::string_view ast_path) {
  auto header_path = GetAstWalkerHeaderPath(ast_path);

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

  Println("#include <memory>");
  Println("#include <optional>");
  Println("#include <variant>");
  Println("#include <vector>");
  Println();

  PrintIncludeHeader("absl/log/log.h");
  PrintIncludeHeader(GetAstHeaderPath(ast_path));
  PrintIncludeHeader(GetAstVisitorHeaderPath(ast_path));
  Println();

  PrintEnterNamespace(cc_namespace);
  Println();

  // 1. Const AST Walker
  PrintAstWalkerClass(ast, /*is_mutable=*/false);
  Println();

  // 2. Mutable AST Walker
  PrintAstWalkerClass(ast, /*is_mutable=*/true);
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

void AstWalkerHeaderPrinter::PrintAstWalkerClass(const AstDef& ast,
                                                 bool is_mutable) {
  std::string lang = (Symbol(ast.lang_name())).ToPascalCase();
  std::string class_name = is_mutable
                               ? absl::StrCat("Mutable", lang, "AstWalker")
                               : absl::StrCat(lang, "AstWalker");
  std::string visitor_type =
      is_mutable ? absl::StrCat("Mutable", lang, "AstVisitor<void>")
                 : absl::StrCat(lang, "AstVisitor<void>");

  auto vars = WithVars({
      {"ClassName", class_name},
      {"VisitorType", visitor_type},
  });

  Println("class $ClassName$ : public $VisitorType$ {");
  Println(" public:");
  {
    auto indent = WithIndent();
    Println("explicit $ClassName$($VisitorType$ *preorder_callback,");
    Println("                     $VisitorType$ *postorder_callback)");
    Println("    : preorder_callback_(preorder_callback),");
    Println("      postorder_callback_(postorder_callback) {}");

    for (const NodeDef* node : ast.topological_sorted_nodes()) {
      if (!IsLeafAstNode(*node)) continue;
      Println();
      PrintVisitMethod(*node, ast, is_mutable);
    }
  }
  Println();
  Println(" private:");
  {
    auto indent = WithIndent();
    Println("[[maybe_unused]] $VisitorType$ *preorder_callback_;");
    Println("[[maybe_unused]] $VisitorType$ *postorder_callback_;");
  }
  Println("};");
}

void AstWalkerHeaderPrinter::PrintVisitMethod(const NodeDef& node,
                                              const AstDef& ast,
                                              bool is_mutable) {
  std::string node_cc_name =
      (Symbol(ast.lang_name()) + node.name()).ToPascalCase();
  std::string node_var = Symbol(node.name()).ToCcVarName();
  std::string const_qual = is_mutable ? "" : "const ";
  std::string ref_type = absl::StrCat(const_qual, node_cc_name, " &");

  auto vars = WithVars({
      {"NodeName", node.name()},
      {"RefType", ref_type},
      {"node_var", node_var},
  });

  Println("void Visit$NodeName$($RefType$$node_var$) override {");
  {
    auto indent = WithIndent();

    // 1. Preorder Callback Hook
    Println("if (preorder_callback_) {");
    Println("  preorder_callback_->Visit$NodeName$($node_var$);");
    Println("}");

    // 2. Traverse Child AST Nodes
    for (const FieldDef* field : GetOrderedFieldsForWalker(node)) {
      PrintFieldTraversal(*field, node_var, ast, is_mutable);
    }

    // 3. Postorder Callback Hook
    Println("if (postorder_callback_) {");
    Println("  postorder_callback_->Visit$NodeName$($node_var$);");
    Println("}");
  }
  Println("}");
}

void AstWalkerHeaderPrinter::PrintFieldTraversal(const FieldDef& field,
                                                 absl::string_view node_var,
                                                 const AstDef& ast,
                                                 bool is_mutable) {
  std::string getter_name = field.name().ToCcVarName();

  // Case A: Direct Scalar ClassType
  if (field.type().IsA<ClassType>()) {
    const NodeDef* target_node = GetAstNodeDef(field.type(), ast);
    if (target_node == nullptr) return;

    auto vars = WithVars({
        {"TargetName", target_node->name()},
        {"getter_name", getter_name},
        {"node_var", node_var},
    });

    if (field.optionalness() == OPTIONALNESS_REQUIRED) {
      Println("Visit$TargetName$(*$node_var$.$getter_name$());");
    } else {
      Println("if ($node_var$.$getter_name$().has_value()) {");
      Println("  Visit$TargetName$(*$node_var$.$getter_name$().value());");
      Println("}");
    }
    return;
  }

  // Case B: ListType
  if (field.type().IsA<ListType>()) {
    const auto& list_type = static_cast<const ListType&>(field.type());

    // Sub-case B1: List of ClassType
    if (list_type.element_type().IsA<ClassType>()) {
      const NodeDef* elem_node = GetAstNodeDef(list_type.element_type(), ast);
      if (elem_node == nullptr) return;

      bool is_optional_list = (field.optionalness() != OPTIONALNESS_REQUIRED);
      std::string list_expr =
          is_optional_list
              ? absl::StrCat("*", node_var, ".", getter_name, "().value()")
              : absl::StrCat("*", node_var, ".", getter_name, "()");

      auto vars = WithVars({
          {"ElemName", elem_node->name()},
          {"getter_name", getter_name},
          {"node_var", node_var},
          {"list_expr", list_expr},
      });

      if (is_optional_list) {
        Println("if ($node_var$.$getter_name$().has_value()) {");
        Indent();
      }
      if (list_type.element_maybe_null() == MaybeNull::kYes) {
        Println("for (const auto &elem : $list_expr$) {");
        Println("  if (elem.has_value()) {");
        Println("    Visit$ElemName$(*elem.value());");
        Println("  }");
        Println("}");
      } else {
        Println("for (const auto &elem : $list_expr$) {");
        Println("  Visit$ElemName$(*elem);");
        Println("}");
      }
      if (is_optional_list) {
        Outdent();
        Println("}");
      }
      return;
    }

    // Sub-case B2: List of VariantType
    if (list_type.element_type().IsA<VariantType>()) {
      const auto& var_type =
          static_cast<const VariantType&>(list_type.element_type());
      if (!HasAstNodes(var_type, ast)) return;

      bool is_optional_list = (field.optionalness() != OPTIONALNESS_REQUIRED);
      std::string list_expr =
          is_optional_list
              ? absl::StrCat("*", node_var, ".", getter_name, "().value()")
              : absl::StrCat("*", node_var, ".", getter_name, "()");

      auto vars = WithVars({
          {"getter_name", getter_name},
          {"node_var", node_var},
          {"list_expr", list_expr},
      });

      if (is_optional_list) {
        Println("if ($node_var$.$getter_name$().has_value()) {");
        Indent();
      }
      if (list_type.element_maybe_null() == MaybeNull::kYes) {
        Println("for (const auto &elem : $list_expr$) {");
        Println("  if (elem.has_value()) {");
        Println("    const auto &elem_val = elem.value();");
        Println("    switch (elem_val.index()) {");
        for (size_t i = 0; i < var_type.types().size(); ++i) {
          if (const NodeDef* alt = GetAstNodeDef(*var_type.types()[i], ast)) {
            auto alt_vars = WithVars({
                {"Index", absl::StrCat(i)},
                {"AltName", alt->name()},
            });
            Println("      case $Index$:");
            Println("        Visit$AltName$(*std::get<$Index$>(elem_val));");
            Println("        break;");
          }
        }
        Println("      default:");
        Println("        LOG(FATAL) << \"Unreachable code.\";");
        Println("    }");
        Println("  }");
        Println("}");
      } else {
        Println("for (const auto &elem : $list_expr$) {");
        Println("  switch (elem.index()) {");
        for (size_t i = 0; i < var_type.types().size(); ++i) {
          if (const NodeDef* alt = GetAstNodeDef(*var_type.types()[i], ast)) {
            auto alt_vars = WithVars({
                {"Index", absl::StrCat(i)},
                {"AltName", alt->name()},
            });
            Println("    case $Index$:");
            Println("      Visit$AltName$(*std::get<$Index$>(elem));");
            Println("      break;");
          }
        }
        Println("    default:");
        Println("      LOG(FATAL) << \"Unreachable code.\";");
        Println("  }");
        Println("}");
      }
      if (is_optional_list) {
        Outdent();
        Println("}");
      }
      return;
    }
    return;
  }

  // Case C: VariantType
  if (field.type().IsA<VariantType>()) {
    const auto& var_type = static_cast<const VariantType&>(field.type());
    if (!HasAstNodes(var_type, ast)) return;

    auto vars = WithVars({
        {"getter_name", getter_name},
        {"node_var", node_var},
    });

    if (field.optionalness() != OPTIONALNESS_REQUIRED) {
      Println("if ($node_var$.$getter_name$().has_value()) {");
      Println(
          "  auto $getter_name$ = "
          "$node_var$.$getter_name$().value();");
      Println("  switch ($getter_name$.index()) {");
      for (size_t i = 0; i < var_type.types().size(); ++i) {
        if (const NodeDef* alt = GetAstNodeDef(*var_type.types()[i], ast)) {
          auto alt_vars = WithVars({
              {"Index", absl::StrCat(i)},
              {"AltName", alt->name()},
              {"getter_name", getter_name},
          });
          Println("    case $Index$:");
          Println("      Visit$AltName$(*std::get<$Index$>($getter_name$));");
          Println("      break;");
        }
      }
      Println("    default:");
      Println("      LOG(FATAL) << \"Unreachable code.\";");
      Println("  }");
      Println("}");
    } else {
      Println("switch ($node_var$.$getter_name$().index()) {");
      for (size_t i = 0; i < var_type.types().size(); ++i) {
        if (const NodeDef* alt = GetAstNodeDef(*var_type.types()[i], ast)) {
          auto alt_vars = WithVars({
              {"Index", absl::StrCat(i)},
              {"AltName", alt->name()},
              {"getter_name", getter_name},
              {"node_var", node_var},
          });
          Println("  case $Index$:");
          Println(
              "    Visit$AltName$("
              "*std::get<$Index$>($node_var$.$getter_name$()));");
          Println("    break;");
        }
      }
      Println("  default:");
      Println("    LOG(FATAL) << \"Unreachable code.\";");
      Println("}");
    }
    return;
  }
}

std::string PrintAstWalkerHeader(const AstDef& ast,
                                 absl::string_view cc_namespace,
                                 absl::string_view ast_path) {
  std::string str;
  {
    google::protobuf::io::StringOutputStream os(&str);
    AstWalkerHeaderPrinter printer(&os);
    printer.PrintAstWalker(ast, cc_namespace, ast_path);
  }
  return str;
}

}  // namespace maldoca
