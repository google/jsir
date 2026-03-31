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

#include "maldoca/astgen/ast_source_printer.h"

#include <cstddef>
#include <string>

#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "maldoca/astgen/ast_def.h"
#include "maldoca/astgen/ast_def.pb.h"
#include "maldoca/astgen/ast_gen_utils.h"
#include "maldoca/astgen/symbol.h"
#include "maldoca/astgen/type.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"

namespace maldoca {

void AstSourcePrinter::PrintAst(const AstDef& ast,
                                absl::string_view cc_namespace,
                                absl::string_view ast_path) {
  auto header_path = GetAstHeaderPath(ast_path);

  PrintLicense();
  Println();

  PrintCodeGenerationWarning();
  Println();

  PrintIncludeHeader(header_path);
  Println();

  Println("// IWYU pragma: begin_keep");
  Println("// NOLINTBEGIN(whitespace/line_length)");
  Println("// clang-format off");
  Println();

  Println("#include <cstdint>");
  Println("#include <memory>");
  Println("#include <optional>");
  Println("#include <string>");
  Println("#include <utility>");
  Println("#include <variant>");
  Println("#include <vector>");
  Println();

  PrintIncludeHeader("absl/container/flat_hash_map.h");
  PrintIncludeHeader("absl/memory/memory.h");
  PrintIncludeHeader("absl/log/log.h");
  PrintIncludeHeader("absl/status/status.h");
  PrintIncludeHeader("absl/status/statusor.h");
  PrintIncludeHeader("absl/strings/str_cat.h");
  PrintIncludeHeader("absl/strings/string_view.h");
  PrintIncludeHeader("nlohmann/json.hpp");
  PrintIncludeHeader("maldoca/base/status_macros.h");
  Println();

  PrintEnterNamespace(cc_namespace);
  Println();

  for (const EnumDef& enum_def : ast.enum_defs()) {
    PrintEnum(enum_def, ast.lang_name());
    Println();
  }

  for (const NodeDef* node : ast.topological_sorted_nodes()) {
    PrintNode(*node, ast.lang_name());
  }

  Println("// clang-format on");
  Println("// NOLINTEND(whitespace/line_length)");
  Println("// IWYU pragma: end_keep");
  Println();

  PrintExitNamespace(cc_namespace);
}

void AstSourcePrinter::PrintEnum(const EnumDef& enum_def,
                                 absl::string_view lang_name) {
  auto vars = WithVars({
      {"EnumName", (Symbol(lang_name) + enum_def.name()).ToPascalCase()},
      {"enum_name", enum_def.name().ToSnakeCase()},
  });

  Println("absl::string_view $EnumName$ToString($EnumName$ $enum_name$) {");
  {
    auto indent = WithIndent();
    Println("switch ($enum_name$) {");
    {
      auto indent = WithIndent();
      for (const EnumMemberDef& member : enum_def.members()) {
        auto vars = WithVars({
            {"kMemberName", (Symbol("k") + member.name()).ToCamelCase()},
            {"string_value", absl::CEscape(member.string_value())},
        });

        Println("case $EnumName$::$kMemberName$:");
        Println("  return \"$string_value$\";");
      }
    }
    Println("}");
  }
  Println("}");
  Println();

  Println(
      "absl::StatusOr<$EnumName$> StringTo$EnumName$(absl::string_view s) {");
  {
    auto indent = WithIndent();

    Println(
        "static const auto *kMap = "
        "new absl::flat_hash_map<absl::string_view, $EnumName$> {");
    {
      auto indent = WithIndent(4);
      for (const EnumMemberDef& member : enum_def.members()) {
        auto vars = WithVars({
            {"kMemberName", (Symbol("k") + member.name()).ToCamelCase()},
            {"string_value", absl::CEscape(member.string_value())},
        });

        Println("{\"$string_value$\", $EnumName$::$kMemberName$},");
      }
    }
    Println("};");
    Println();

    const auto code = UnIndentedSource(R"(
auto it = kMap->find(s);
if (it == kMap->end()) {
  return absl::InvalidArgumentError(absl::StrCat("Invalid string for $EnumName$: ", s));
}
return it->second;
    )");
    Println(code);
  }
  Println("}");
}

void AstSourcePrinter::PrintNode(const NodeDef& node,
                                 absl::string_view lang_name) {
  PrintTitle((Symbol(lang_name) + node.name()).ToPascalCase());
  Println();

  auto vars = WithVars({
      {"NodeType", ClassType(Symbol(node.name()), lang_name).CcType()},
  });

  if (node.node_type_enum().has_value()) {
    PrintEnum(*node.node_type_enum().value(), lang_name);
    Println();
  }

  if (!node.aggregated_fields().empty()) {
    PrintConstructor(node, lang_name);
    Println();
  }

  for (const FieldDef& field : node.fields()) {
    const Type& type = field.type();
    bool is_optional = field.optionalness() != OPTIONALNESS_REQUIRED;

    std::string cc_getter_type = CcMutableGetterType(field);
    std::string cc_const_getter_type = CcConstGetterType(field);

    auto vars = WithVars({
        {"NodeType", (Symbol(lang_name) + node.name()).ToPascalCase()},
        {"cc_getter_type", cc_getter_type},
        {"cc_const_getter_type", cc_const_getter_type},
        {"cc_type", CcType(field)},
        {"field_name", field.name().ToCcVarName()},
    });

    // If both the mutable getter and const getter would have the same return
    // type, then we just skip the mutable getter and only keep the const
    // getter.
    if (cc_getter_type != cc_const_getter_type) {
      Println("$cc_getter_type$ $NodeType$::$field_name$() {");
      {
        auto indent = WithIndent();
        PrintGetterBody(field.name(), type, is_optional);
      }
      Println("}");
      Println();
    }

    Println("$cc_const_getter_type$ $NodeType$::$field_name$() const {");
    {
      auto indent = WithIndent();
      PrintGetterBody(field.name(), type, is_optional);
    }
    Println("}");
    Println();

    Println("void $NodeType$::set_$field_name$($cc_type$ $field_name$) {");
    {
      auto indent = WithIndent();
      PrintSetterBody(field.name(), type, is_optional);
    }
    Println("}");
    Println();
  }
}

void AstSourcePrinter::PrintConstructor(const NodeDef& node,
                                        absl::string_view lang_name) {
  auto vars = WithVars({
      {"NodeType", (Symbol(lang_name) + node.name()).ToPascalCase()},
  });
  Print("$NodeType$::$NodeType$(");
  if (!node.aggregated_fields().empty()) {
    Println();
    auto indent = WithIndent(4);

    TabPrinter separator_printer{{
        .print_separator = [this] { Print(",\n"); },
    }};
    for (const FieldDef* field : node.aggregated_fields()) {
      auto vars = WithVars({
          {"cc_type", CcType(*field)},
          {"field_name", field->name().ToCcVarName()},
      });

      separator_printer.Print();
      Print("$cc_type$ $field_name$");
    }
  }
  Println(")");

  {
    auto indent = WithIndent(4);

    TabPrinter tab_printer{{
        .print_prefix =
            [&] {
              Print(": ");
              Indent();
            },
        .print_separator = [&] { Print(",\n"); },
        .print_postfix = [&] { Outdent(); },
    }};
    for (const NodeDef* ancestor : node.ancestors()) {
      tab_printer.Print();

      auto vars = WithVars({
          {"AncestorType",
           (Symbol(lang_name) + ancestor->name()).ToPascalCase()},
      });
      Print("$AncestorType$(");

      TabPrinter ancestor_tab_printer{{
          .print_separator = [&] { Print(", "); },
      }};
      for (const FieldDef* field : ancestor->aggregated_fields()) {
        ancestor_tab_printer.Print();

        auto vars = WithVars({
            {"field_name", field->name().ToCcVarName()},
        });
        Print("std::move($field_name$)");
      }

      Print(")");
    }

    for (const FieldDef& field : node.fields()) {
      auto vars = WithVars({
          {"field_name", field.name().ToCcVarName()},
      });

      tab_printer.Print();
      Print("$field_name$_(std::move($field_name$))");
    }
  }

  Println(" {}");
}

void AstSourcePrinter::PrintGetterBody(const std::string& cc_expr,
                                       const Type& type) {
  auto vars = WithVars({
      {"cc_expr", cc_expr},
  });

  switch (type.kind()) {
    case TypeKind::kBuiltin: {
      Println("return $cc_expr$;");
      break;
    }

    case TypeKind::kEnum: {
      Println("return $cc_expr$;");
      break;
    }

    case TypeKind::kClass: {
      Println("return $cc_expr$.get();");
      break;
    }

    case TypeKind::kVariant: {
      const auto& variant_type = static_cast<const VariantType&>(type);

      Println("switch ($cc_expr$.index()) {");
      {
        auto indent = WithIndent();

        for (size_t i = 0; i != variant_type.types().size(); ++i) {
          auto vars = WithVars({
              {"i", std::to_string(i)},
          });
          const ScalarType& type = *variant_type.types().at(i);

          Println("case $i$: {");
          {
            auto indent = WithIndent();
            PrintGetterBody(absl::StrFormat("std::get<%zu>(%s)", i, cc_expr),
                            type);
          }
          Println("}");
        }

        Println("default:");
        Println("  LOG(FATAL) << \"Unreachable code.\";");
      }
      Println("}");

      break;
    }

    case TypeKind::kList: {
      Println("return &$cc_expr$;");
      break;
    }
  }
}

void AstSourcePrinter::PrintGetterBody(const Symbol& field_name,
                                       const Type& type, bool is_optional) {
  if (is_optional) {
    auto vars = WithVars({
        {"field_name", field_name.ToCcVarName()},
    });

    Println("if (!$field_name$_.has_value()) {");
    Println("  return std::nullopt;");
    Println("} else {");
    {
      auto indent = WithIndent();
      auto value_cc_expr = absl::StrCat(field_name.ToCcVarName(), "_.value()");
      PrintGetterBody(value_cc_expr, type);
    }
    Println("}");

  } else {
    PrintGetterBody(absl::StrCat(field_name.ToCcVarName(), "_"), type);
  }
}

void AstSourcePrinter::PrintSetterBody(const Symbol& field_name,
                                       const Type& type, bool is_optional) {
  auto vars = WithVars({
      {"field_name", field_name.ToCcVarName()},
  });

  if (type.IsA<BuiltinType>()) {
    const auto& builtin_type = static_cast<const BuiltinType&>(type);
    switch (builtin_type.builtin_kind()) {
      case BuiltinTypeKind::kBool:
      case BuiltinTypeKind::kDouble:
        Println("$field_name$_ = $field_name$;");
        return;
      default:
        break;
    }
  }

  if (type.IsA<EnumType>()) {
    Println("$field_name$_ = $field_name$;");
    return;
  }

  Println("$field_name$_ = std::move($field_name$);");
}

std::string PrintAstSource(const AstDef& ast, absl::string_view cc_namespace,
                           absl::string_view ast_path) {
  std::string str;
  {
    google::protobuf::io::StringOutputStream os(&str);
    AstSourcePrinter printer(&os);
    printer.PrintAst(ast, cc_namespace, ast_path);
  }

  return str;
}

}  // namespace maldoca
