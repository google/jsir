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

#include "maldoca/astgen/ast_header_printer.h"

#include <string>

#include "absl/strings/string_view.h"
#include "maldoca/astgen/ast_def.h"
#include "maldoca/astgen/ast_gen_utils.h"
#include "maldoca/astgen/symbol.h"
#include "maldoca/astgen/type.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"

namespace maldoca {

void AstHeaderPrinter::PrintAst(const AstDef& ast,
                                absl::string_view cc_namespace,
                                absl::string_view ast_path) {
  auto header_path = GetAstHeaderPath(ast_path);

  PrintLicense();
  Println();

  PrintCodeGenerationWarning();
  Println();

  PrintEnterHeaderGuard(header_path);
  Println();

  Println("// IWYU pragma: begin_keep");
  Println("// NOLINTBEGIN(whitespace/line_length)");
  Println("// clang-format off");
  Println();

  Println("#include <optional>");
  Println("#include <string>");
  Println("#include <variant>");
  Println("#include <vector>");
  Println();

  PrintIncludeHeader("absl/status/statusor.h");
  PrintIncludeHeader("absl/strings/string_view.h");
  PrintIncludeHeader("nlohmann/json.hpp");
  Println();

  PrintEnterNamespace(cc_namespace);
  Println();

  for (const EnumDef& enum_def : ast.enum_defs()) {
    PrintEnum(enum_def, ast.lang_name());
    Println();
  }

  for (const NodeDef* node : ast.topological_sorted_nodes()) {
    PrintNode(*node, ast.lang_name());
    Println();
  }

  Println("// clang-format on");
  Println("// NOLINTEND(whitespace/line_length)");
  Println("// IWYU pragma: end_keep");
  Println();

  PrintExitNamespace(cc_namespace);
  Println();

  PrintExitHeaderGuard(header_path);
}

void AstHeaderPrinter::PrintEnum(const EnumDef& enum_def,
                                 absl::string_view lang_name) {
  auto vars = WithVars({
      {"EnumName", (Symbol(lang_name) + enum_def.name()).ToPascalCase()},
      {"enum_name", enum_def.name().ToSnakeCase()},
  });

  Println("enum class $EnumName$ {");
  {
    auto indent = WithIndent();
    for (const EnumMemberDef& member : enum_def.members()) {
      auto vars = WithVars({
          {"kMemberName", (Symbol("k") + member.name()).ToCamelCase()},
      });

      Println("$kMemberName$,");
    }
  }
  Println("};");
  Println();

  Println("absl::string_view $EnumName$ToString($EnumName$ $enum_name$);");
  Println(
      "absl::StatusOr<$EnumName$> StringTo$EnumName$(absl::string_view s);");
}

void AstHeaderPrinter::PrintNode(const NodeDef& node,
                                 absl::string_view lang_name) {
  auto vars = WithVars({
      {"NodeType", (Symbol(lang_name) + node.name()).ToPascalCase()},
      {"json_variable", kJsonValueVariableName},
      {"os_variable", kOsValueVariableName},
  });

  if (node.node_type_enum().has_value()) {
    PrintEnum(*node.node_type_enum().value(), lang_name);
    Println();
  }

  Print("class $NodeType$");
  if (!node.parents().empty()) {
    Print(" : ");
    TabPrinter separator_printer{{
        .print_separator = [&] { Print(", "); },
    }};
    for (const NodeDef* parent : node.parents()) {
      auto vars = WithVars({
          {"BaseType", (Symbol(lang_name) + parent->name()).ToPascalCase()},
      });

      separator_printer.Print();
      Print("public virtual $BaseType$");
    }
  }
  Println(" {");

  // Always print "public:" because the declaration of FromJson() always
  // exists.
  Println(" public:");
  {
    auto indent = WithIndent();

    // Constructor
    if (!node.aggregated_fields().empty()) {
      PrintConstructor(node, lang_name);
      Println();
    }

    // Destructor
    if (node.parents().empty() && !node.children().empty()) {
      Println("virtual ~$NodeType$() = default;");
      Println();
    }

    // Get type enum.
    if (node.node_type_enum().has_value()) {
      auto node_type_enum_name = node.node_type_enum().value()->name();
      auto vars = WithVars({
          {"NodeTypeEnum",
           (Symbol(lang_name) + node_type_enum_name).ToPascalCase()},
          {"node_type_enum", node_type_enum_name.ToCcVarName()},
      });

      Println("virtual $NodeTypeEnum$ $node_type_enum$() const = 0;");
      Println();

    } else if (node.children().empty()) {
      for (const NodeDef* ancestor : node.ancestors()) {
        if (!ancestor->node_type_enum().has_value()) {
          continue;
        }

        auto root_type_enum_name = ancestor->node_type_enum().value()->name();
        auto vars = WithVars({
            {"RootTypeEnum",
             (Symbol(lang_name) + root_type_enum_name).ToPascalCase()},
            {"root_type_enum", root_type_enum_name.ToCcVarName()},
            {"NodeTypeNoLang", Symbol(node.name()).ToPascalCase()},
        });

        Println("$RootTypeEnum$ $root_type_enum$() const override {");
        Println("  return $RootTypeEnum$::k$NodeTypeNoLang$;");
        Println("}");
        Println();
      }
    }

    // Serialize
    if (node.parents().empty()) {
      if (node.children().empty()) {
        // Non-virtual.
        Println("void Serialize(std::ostream& $os_variable$) const;");
        Println();
      } else {
        // Virtual base.
        // We define a pure virtual function here, and override it in leaf
        // types.
        Println(
            "virtual void Serialize(std::ostream& $os_variable$) "
            "const = 0;");
        Println();
      }
    } else {
      if (node.children().empty()) {
        // Leaf type.
        // We override the virtual function.
        Println(
            "void Serialize(std::ostream& $os_variable$) "
            "const override;");
        Println();
      } else {
        // Non-leaf type - skipped.
        // We only override in leaf types. Here it's still pure virtual.
      }
    }

    // FromJson
    Println(
        "static absl::StatusOr<std::unique_ptr<$NodeType$>> FromJson("
        "const nlohmann::json& $json_variable$);");
    Println();

    // Getters and setters.
    for (const FieldDef& field : node.fields()) {
      PrintGetterSetterDeclarations(field, lang_name);
      Println();
    }
  }

  Println(" protected:");
  {
    auto indent = WithIndent();

    // SerializeFields
    Println("// Internal function used by Serialize().");
    Println("// Sets the fields defined in this class.");
    Println("// Does not set fields defined in ancestors.");
    Println(
        "void SerializeFields(std::ostream& $os_variable$, "
        "bool &needs_comma) const;");

    // Get<FieldName>FromJson() functions.
    if (!node.fields().empty()) {
      Println();
      Println("// Internal functions used by FromJson().");
      Println("// Extracts a field from a JSON object.");
      for (const FieldDef& field : node.fields()) {
        PrintGetFromJson(field, lang_name);
      }
    }
  }

  // Print member variables.
  if (!node.fields().empty()) {
    Println();
    Println(" private:");
    {
      auto indent = WithIndent();
      for (const FieldDef& field : node.fields()) {
        PrintMemberVariable(field, lang_name);
      }
    }
  }

  Println("};");
}

void AstHeaderPrinter::PrintConstructor(const NodeDef& node,
                                        absl::string_view lang_name) {
  auto vars = WithVars({
      {"NodeType", (Symbol(lang_name) + node.name()).ToPascalCase()},
  });
  Print("explicit $NodeType$(");
  if (!node.aggregated_fields().empty()) {
    Println();
    {
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
  }
  Println(");");
}

void AstHeaderPrinter::PrintGetterSetterDeclarations(
    const FieldDef& field, absl::string_view lang_name) {
  std::string cc_getter_type = CcMutableGetterType(field);
  std::string cc_const_getter_type = CcConstGetterType(field);

  auto vars = WithVars({
      {"cc_getter_type", cc_getter_type},
      {"cc_const_getter_type", cc_const_getter_type},
      {"cc_type", CcType(field)},
      {"field_name", field.name().ToCcVarName()},
  });

  // If the mutable getter would return the same type as the const getter, skip
  // the mutable getter.
  if (cc_getter_type != cc_const_getter_type) {
    Println("$cc_getter_type$ $field_name$();");
  }
  Println("$cc_const_getter_type$ $field_name$() const;");
  Println("void set_$field_name$($cc_type$ $field_name$);");
}

void AstHeaderPrinter::PrintMemberVariable(const FieldDef& field,
                                           absl::string_view lang_name) {
  auto vars = WithVars({
      {"cc_type", CcType(field)},
      {"field_name", field.name().ToCcVarName()},
  });

  Println("$cc_type$ $field_name$_;");
}

void AstHeaderPrinter::PrintGetFromJson(const FieldDef& field,
                                        absl::string_view lang_name) {
  auto vars = WithVars({
      {"cc_type", CcType(field)},
      {"FieldName", field.name().ToPascalCase()},
      {"os_variable", kOsValueVariableName},
  });

  Println(
      "static absl::StatusOr<$cc_type$> "
      "Get$FieldName$(const nlohmann::json& $json_variable$);");
}

std::string PrintAstHeader(const AstDef& ast, absl::string_view cc_namespace,
                           absl::string_view ast_path) {
  std::string str;
  {
    google::protobuf::io::StringOutputStream os(&str);
    AstHeaderPrinter printer(&os);
    printer.PrintAst(ast, cc_namespace, ast_path);
  }

  return str;
}

}  // namespace maldoca
