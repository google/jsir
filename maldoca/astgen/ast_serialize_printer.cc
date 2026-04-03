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

#include "maldoca/astgen/ast_serialize_printer.h"

#include <cstddef>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "maldoca/astgen/ast_def.h"
#include "maldoca/astgen/ast_gen_utils.h"
#include "maldoca/astgen/symbol.h"
#include "maldoca/astgen/type.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"

namespace maldoca {

void AstSerializePrinter::PrintAst(const AstDef& ast,
                                   absl::string_view cc_namespace,
                                   absl::string_view ast_path) {
  auto vars = WithVars({
      {"os_variable", kOsValueVariableName},
  });

  auto header_path = GetAstHeaderPath(ast_path);

  PrintLicense();
  Println();

  PrintCodeGenerationWarning();
  Println();

  Println("// IWYU pragma: begin_keep");
  Println("// NOLINTBEGIN(whitespace/line_length)");
  Println("// clang-format off");
  Println();

  Println("#include <cmath>");
  Println("#include <limits>");
  Println("#include <ostream>");
  Println("#include <string>");
  Println("#include <utility>");
  Println();

  PrintIncludeHeaders({
      std::string(header_path),
      "absl/log/log.h",
      "absl/memory/memory.h",
      "absl/status/status.h",
      "absl/strings/string_view.h",
      "nlohmann/json.hpp",
      "maldoca/base/status_macros.h",
  });
  Println();

  PrintEnterNamespace(cc_namespace);
  Println();

  Println(
      R"(void MaybeAddComma(std::ostream &$os_variable$, bool &needs_comma) {
  if (needs_comma) {
    $os_variable$ << ",";
  }
  needs_comma = true;
}
)");

  for (const auto& node : ast.topological_sorted_nodes()) {
    PrintTitle((Symbol(ast.lang_name()) + node->name()).ToPascalCase());
    Println();

    PrintSerializeFieldsFunction(*node, ast.lang_name());
    Println();

    if (node->children().empty()) {
      PrintSerializeFunction(*node, ast.lang_name());
      Println();
    }
  }

  Println("// clang-format on");
  Println("// NOLINTEND(whitespace/line_length)");
  Println("// IWYU pragma: end_keep");
  Println();

  PrintExitNamespace(cc_namespace);
}

void AstSerializePrinter::PrintBuiltinSerialize(const BuiltinType& type,
                                                const std::string& lhs,
                                                const std::string& rhs) {
  auto vars = WithVars({
      {"os_variable", kOsValueVariableName},
      {"lhs", lhs},
      {"rhs", rhs},
  });

  if (!lhs.empty()) {
    Println("$os_variable$ << $lhs$ << (nlohmann::json($rhs$)).dump();");
  } else {
    Println("$os_variable$ << (nlohmann::json($rhs$)).dump();");
  }
}

void AstSerializePrinter::PrintEnumSerialize(const EnumType& type,
                                             const std::string& lhs,
                                             const std::string& rhs,
                                             absl::string_view lang_name) {
  auto vars = WithVars({
      {"os_variable", kOsValueVariableName},
      {"lhs", lhs},
      {"rhs", rhs},
      {"EnumName", (Symbol(lang_name) + type.name()).ToPascalCase()},
  });

  if (!lhs.empty()) {
    Println(
        R"($os_variable$ << $lhs$ << "\"" << $EnumName$ToString($rhs$) << "\"";)");
  } else {
    Println(R"($os_variable$ << "\"" << $EnumName$ToString($rhs$) << "\"";)");
  }
}

void AstSerializePrinter::PrintClassSerialize(const ClassType& type,
                                              const std::string& lhs,
                                              const std::string& rhs) {
  auto vars = WithVars({
      {"os_variable", kOsValueVariableName},
      {"lhs", lhs},
      {"rhs", rhs},
  });

  if (!lhs.empty()) {
    Println("$os_variable$ << $lhs$;");
  }
  Println("$rhs$->Serialize($os_variable$);");
}

void AstSerializePrinter::PrintVariantSerialize(const VariantType& variant_type,
                                                const std::string& lhs,
                                                const std::string& rhs,
                                                absl::string_view lang_name) {
  auto vars = WithVars({
      {"lhs", lhs},
      {"rhs", rhs},
  });

  Println("switch ($rhs$.index()) {");
  {
    auto indent = WithIndent();
    for (size_t i = 0; i != variant_type.types().size(); ++i) {
      auto vars = WithVars({
          {"i", std::to_string(i)},
      });

      Println("case $i$: {");
      {
        auto indent = WithIndent();
        const ScalarType& type = *variant_type.types()[i];
        PrintSerialize(type, lhs, absl::StrFormat("std::get<%zu>(%s)", i, rhs),
                       lang_name);
        Println("break;");
      }

      Println("}");
    }

    Println("default:");
    Println("  LOG(FATAL) << \"Unreachable code.\";");
  }
  Println("}");
}

void AstSerializePrinter::PrintListSerialize(const ListType& list_type,
                                             const std::string& lhs,
                                             const std::string& rhs,
                                             absl::string_view lang_name) {
  constexpr char kRhsElement[] = "element";
  CHECK_NE(lhs, kRhsElement);
  CHECK_NE(rhs, kRhsElement);

  constexpr char kLhsElement[] = "element_json";
  CHECK_NE(lhs, kLhsElement);
  CHECK_NE(rhs, kLhsElement);

  auto vars = WithVars({
      {"os_variable", kOsValueVariableName},
      {"lhs", lhs},
      {"rhs", rhs},
      {"lhs_element", kLhsElement},
      {"rhs_element", kRhsElement},
  });

  if (!lhs.empty()) {
    Println(R"($os_variable$ << $lhs$ << "[";)");
  } else {
    Println(R"($os_variable$ << "[";)");
  }
  Println("{");
  {
    auto indent = WithIndent();

    Println("bool needs_comma = false;");
    Println("for (const auto& $rhs_element$ : $rhs$) {");
    {
      auto indent = WithIndent();
      Println("MaybeAddComma($os_variable$, needs_comma);");
      PrintNullableToJson(list_type.element_type(),
                          list_type.element_maybe_null(), "", kRhsElement,
                          lang_name);
    }
    Println("}");
  }
  Println("}");
  Println(R"($os_variable$ << "]";)");
}

void AstSerializePrinter::PrintSerialize(const Type& type,
                                         const std::string& lhs,
                                         const std::string& rhs,
                                         absl::string_view lang_name) {
  switch (type.kind()) {
    case TypeKind::kBuiltin: {
      const auto& builtin_type = static_cast<const BuiltinType&>(type);
      PrintBuiltinSerialize(builtin_type, lhs, rhs);
      break;
    }

    case TypeKind::kEnum: {
      const auto& enum_type = static_cast<const EnumType&>(type);
      PrintEnumSerialize(enum_type, lhs, rhs, lang_name);
      break;
    }

    case TypeKind::kClass: {
      const auto& class_type = static_cast<const ClassType&>(type);
      PrintClassSerialize(class_type, lhs, rhs);
      break;
    }

    case TypeKind::kVariant: {
      const auto& variant_type = static_cast<const VariantType&>(type);
      PrintVariantSerialize(variant_type, lhs, rhs, lang_name);
      break;
    }

    case TypeKind::kList: {
      const auto& list_type = static_cast<const ListType&>(type);
      PrintListSerialize(list_type, lhs, rhs, lang_name);
      break;
    }
  }
}

void AstSerializePrinter::PrintNullableToJson(const Type& type,
                                              MaybeNull maybe_null,
                                              const std::string& lhs,
                                              const std::string& rhs,
                                              absl::string_view lang_name) {
  switch (maybe_null) {
    case MaybeNull::kNo: {
      PrintSerialize(type, lhs, rhs, lang_name);
      break;
    }

    case MaybeNull::kYes: {
      auto vars = WithVars({
          {"os_variable", kOsValueVariableName},
          {"lhs", lhs},
          {"rhs", rhs},
      });

      Println("if ($rhs$.has_value()) {");
      {
        auto indent = WithIndent();
        auto rhs_value = absl::StrCat(rhs, ".value()");
        PrintSerialize(type, lhs, rhs_value, lang_name);
      }
      Println("} else {");
      {
        auto indent = WithIndent();

        if (!lhs.empty()) {
          Println(R"($os_variable$ << $lhs$ << "null";)");
        } else {
          Println(R"($os_variable$ << "null";)");
        }
      }
      Println("}");
      break;
    }
  }
}

void AstSerializePrinter::PrintSerializeFieldsFunction(
    const NodeDef& node, absl::string_view lang_name) {
  auto vars = WithVars({
      {"NodeType", (Symbol(lang_name) + node.name()).ToPascalCase()},
      {"os_variable", kOsValueVariableName},
  });

  Println(
      "void $NodeType$::SerializeFields(std::ostream& $os_variable$, "
      "bool &needs_comma) const {");
  {
    auto indent = WithIndent();

    for (const FieldDef& field : node.fields()) {
      // E.g. "\"fieldName\":"
      auto lhs = absl::StrFormat(R"("\"%s\":")", field.name().ToCamelCase());

      // E.g. field_name_
      auto rhs = absl::StrCat(field.name().ToCcVarName(), "_");

      switch (field.optionalness()) {
        case OPTIONALNESS_UNSPECIFIED: {
          LOG(FATAL) << "Invalid Optionalness. Should be a bug.";
          break;
        }

        case OPTIONALNESS_REQUIRED: {
          Println("MaybeAddComma($os_variable$, needs_comma);");
          PrintSerialize(field.type(), lhs, rhs, lang_name);
          break;
        }

        case OPTIONALNESS_MAYBE_UNDEFINED: {
          auto vars = WithVars({
              {"rhs", rhs},
          });

          // If <rhs> == std::nullopt, the assignment does not happen.
          Println("if ($rhs$.has_value()) {");
          {
            auto indent = WithIndent();
            auto rhs_value = absl::StrCat(rhs, ".value()");
            Println("MaybeAddComma($os_variable$, needs_comma);");
            PrintSerialize(field.type(), lhs, rhs_value, lang_name);
          }
          Println("}");

          break;
        }
        case OPTIONALNESS_MAYBE_NULL: {
          Println("MaybeAddComma($os_variable$, needs_comma);");
          PrintNullableToJson(field.type(), MaybeNull::kYes, lhs, rhs,
                              lang_name);
          break;
        }
      }
    }
  }
  Println("}");
}

void AstSerializePrinter::PrintSerializeFunction(const NodeDef& node,
                                                 absl::string_view lang_name) {
  auto vars = WithVars({
      {"NodeType", (Symbol(lang_name) + node.name()).ToPascalCase()},
      {"NodeTypeNoLangName", node.name()},
      {"os_variable", kOsValueVariableName},
  });

  Println("void $NodeType$::Serialize(std::ostream& $os_variable$) const {");
  {
    auto indent = WithIndent();

    Println(R"($os_variable$ << "{";)");
    Println("{");
    {
      auto indent = WithIndent();
      Println("bool needs_comma = false;");

      // The "type" field.
      if (!node.parents().empty() || !node.children().empty()) {
        Println("MaybeAddComma($os_variable$, needs_comma);");
        Println(R"($os_variable$ << "\"type\":\"$NodeTypeNoLangName$\"";)");
      }

      // Assign fields of ancestors of this node.
      for (const NodeDef* ancestor : node.ancestors()) {
        auto vars = WithVars({
            {"AncestorType",
             (Symbol(lang_name) + ancestor->name()).ToPascalCase()},
        });
        Println(
            "$AncestorType$::SerializeFields($os_variable$, "
            "needs_comma);");
      }

      // Assign fields of the node itself.
      Println("$NodeType$::SerializeFields($os_variable$, needs_comma);");
    }
    Println("}");

    Println(R"($os_variable$ << "}";)");
  }
  Println("}");
}

std::string PrintAstToJson(const AstDef& ast, absl::string_view cc_namespace,
                           absl::string_view ast_path) {
  std::string str;
  {
    google::protobuf::io::StringOutputStream os(&str);
    AstSerializePrinter printer(&os);
    printer.PrintAst(ast, cc_namespace, ast_path);
  }

  return str;
}

}  // namespace maldoca
