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

#include "maldoca/astgen/ast_from_json_printer.h"

#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "maldoca/astgen/ast_def.h"
#include "maldoca/astgen/ast_def.pb.h"
#include "maldoca/astgen/ast_gen_utils.h"
#include "maldoca/astgen/symbol.h"
#include "maldoca/astgen/type.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"

namespace maldoca {
namespace {

static void GetCheckedClasses(const Type& type, bool is_part_of_variant,
                              absl::flat_hash_set<std::string>* node_names) {
  switch (type.kind()) {
    case TypeKind::kBuiltin:
    case TypeKind::kEnum:
      break;
    case TypeKind::kClass: {
      if (is_part_of_variant) {
        const auto& class_type = static_cast<const ClassType&>(type);
        node_names->insert(class_type.name().ToPascalCase());
      }
      break;
    }
    case TypeKind::kVariant: {
      const auto& variant_type = static_cast<const VariantType&>(type);
      for (const auto& element_type : variant_type.types()) {
        GetCheckedClasses(*element_type, /*is_part_of_variant=*/true,
                          node_names);
      }
      break;
    }
    case TypeKind::kList: {
      const auto& list_type = static_cast<const ListType&>(type);
      GetCheckedClasses(list_type.element_type(), is_part_of_variant,
                        node_names);
      break;
    }
  }
}

static absl::flat_hash_set<std::string> GetCheckedClasses(const AstDef& ast) {
  absl::flat_hash_set<std::string> checked_classes;
  for (const NodeDef* node : ast.topological_sorted_nodes()) {
    for (const FieldDef& field : node->fields()) {
      GetCheckedClasses(field.type(), /*is_part_of_variant=*/false,
                        &checked_classes);
    }
  }
  return checked_classes;
}

}  // namespace

void AstFromJsonPrinter::PrintAst(const AstDef& ast,
                                  absl::string_view cc_namespace,
                                  absl::string_view ast_path) {
  auto vars = WithVars({
      {"json_variable", kJsonValueVariableName},
  });

  auto header_path = GetAstHeaderPath(ast_path);

  PrintLicense();
  Println();

  PrintCodeGenerationWarning();
  Println();

  Println("// NOLINTBEGIN(whitespace/line_length)");
  Println("// clang-format off");
  Println("// IWYU pragma: begin_keep");
  Println();

  Println("#include <cstdint>");
  Println("#include <memory>");
  Println("#include <optional>");
  Println("#include <string>");
  Println("#include <utility>");
  Println("#include <variant>");
  Println("#include <vector>");
  Println();

  PrintIncludeHeaders({
      std::string(header_path),
      "absl/container/flat_hash_set.h",
      "absl/memory/memory.h",
      "absl/status/status.h",
      "absl/status/statusor.h",
      "absl/strings/str_cat.h",
      "absl/strings/string_view.h",
      "maldoca/astgen/ast_from_json_utils.h",
      "maldoca/base/status_macros.h",
      "nlohmann/json.hpp",
  });
  Println();

  PrintEnterNamespace(cc_namespace);
  Println();

  absl::flat_hash_set<std::string> checked_classes = GetCheckedClasses(ast);

  for (const NodeDef* node : ast.topological_sorted_nodes()) {
    PrintTitle((Symbol(ast.lang_name()) + node->name()).ToPascalCase());
    Println();

    if (checked_classes.contains(node->name())) {
      PrintTypeChecker(*node);
      Println();
    }

    for (const FieldDef& field : node->fields()) {
      PrintGetFieldFunction(node->name(), field, ast.lang_name());
      Println();
    }

    PrintFromJsonFunction(*node, ast.lang_name());
    Println();
  }

  Println("// clang-format on");
  Println("// NOLINTEND(whitespace/line_length)");
  Println("// IWYU pragma: end_keep");
  Println();

  PrintExitNamespace(cc_namespace);
}

void AstFromJsonPrinter::PrintTypeChecker(const NodeDef& node) {
  auto vars = WithVars({
      {"NodeType", std::string(node.name())},
      {"json_variable", kJsonValueVariableName},
  });

  Println("static bool Is$NodeType$(const nlohmann::json& $json_variable$) {");
  absl::Cleanup end_body = [&] { Println("}"); };
  {
    auto indent = WithIndent();

    Println("if (!$json_variable$.is_object()) {");
    Println("  return false;");
    Println("}");

    if (node.children().empty() && node.parents().empty()) {
      // This is not a virtual class.
      Println("return true;");
      return;
    }

    const std::string code = UnIndentedSource(R"cc(
      auto type_it = $json_variable$.find("type");
      if (type_it == $json_variable$.end()) {
        return false;
      }
      const nlohmann::json &type_json = type_it.value();
      if (!type_json.is_string()) {
        return false;
      }
      const std::string &type = type_json.get<std::string>();
    )cc");
    Println(code);

    if (!node.leaves().empty()) {
      Println(
          "static const auto *kTypes = new absl::flat_hash_set<std::string>{");
      {
        auto indent = WithIndent(4);
        for (const NodeDef* leaf : node.leaves()) {
          auto vars = WithVars({
              {"LeafType", leaf->name()},
          });
          Println("\"$LeafType$\",");
        }
      }
      Println("};");
      Println();

      Println("return kTypes->contains(type);");

    } else {
      CHECK_EQ(node.name(), node.type().value());
      Println("return type == \"$NodeType$\";");
    }
  }
}

void AstFromJsonPrinter::PrintGetFieldFunction(const std::string& node_name,
                                               const FieldDef& field,
                                               absl::string_view lang_name) {
  std::string get_field_function_name = [&] {
    switch (field.optionalness()) {
      case OPTIONALNESS_REQUIRED:
        return "GetRequiredField";
      case OPTIONALNESS_MAYBE_NULL:
        return "GetNullableField";
      case OPTIONALNESS_MAYBE_UNDEFINED:
        return "GetOptionalField";
      default:
        LOG(FATAL) << "Unreachable code.";
    }
  }();

  auto vars = WithVars({
      {"NodeType", (Symbol(lang_name) + node_name).ToPascalCase()},
      {"return_cc_type", CcType(field)},
      {"field_cc_type", field.type().CcType()},
      {"fieldName", field.name().ToCamelCase()},
      {"FieldName", field.name().ToPascalCase()},
      {"json_variable", kJsonValueVariableName},
      {"GetField", get_field_function_name},
  });

  Println("absl::StatusOr<$return_cc_type$>");
  Println(
      "$NodeType$::Get$FieldName$(const nlohmann::json& $json_variable$) {");
  {
    auto indent = WithIndent();
    Println("return $GetField$<$field_cc_type$>(");
    {
      auto indent = WithIndent(4);
      Println("$json_variable$,");
      Println("\"$fieldName$\",");
      PrintConverter(field.type(), lang_name);
      Println();
    }
    Println(");");
  }
  Println("}");
}

void AstFromJsonPrinter::PrintTypeCheckerName(const ScalarType& type) {
  switch (type.kind()) {
    case TypeKind::kList:
    case TypeKind::kVariant: {
      LOG(FATAL) << "Unreachable code.";
    }

    case TypeKind::kClass: {
      const auto& class_type = static_cast<const ClassType&>(type);
      Print("Is" + class_type.name().ToPascalCase());
      break;
    }

    case TypeKind::kEnum: {
      const auto& enum_type = static_cast<const EnumType&>(type);
      Print("Is" + enum_type.name().ToPascalCase());
      break;
    }

    case TypeKind::kBuiltin: {
      const auto& builtin_type = static_cast<const BuiltinType&>(type);
      switch (builtin_type.builtin_kind()) {
        case BuiltinTypeKind::kString:
          Print("IsString");
          break;
        case BuiltinTypeKind::kBool:
          Print("IsBool");
          break;
        case BuiltinTypeKind::kInt64:
          Print("IsInt64");
          break;
        case BuiltinTypeKind::kDouble:
          Print("IsDouble");
          break;
        default:
          LOG(FATAL) << "Unreachable code.";
      }
      break;
    }
  }
}

void AstFromJsonPrinter::PrintConverter(const Type& type,
                                        absl::string_view lang_name) {
  switch (type.kind()) {
    case TypeKind::kList: {
      const auto& list_type = static_cast<const ListType&>(type);
      PrintListConverter(list_type, lang_name);
      break;
    }

    case TypeKind::kVariant: {
      const auto& variant_type = static_cast<const VariantType&>(type);
      PrintVariantConverter(variant_type, lang_name);
      break;
    }

    case TypeKind::kClass: {
      const auto& class_type = static_cast<const ClassType&>(type);
      PrintClassConverter(class_type, lang_name);
      break;
    }

    case TypeKind::kEnum: {
      const auto& enum_type = static_cast<const EnumType&>(type);
      PrintEnumConverter(enum_type, lang_name);
      break;
    }

    case TypeKind::kBuiltin: {
      const auto& builtin_type = static_cast<const BuiltinType&>(type);
      PrintBuiltinConverter(builtin_type, lang_name);
      break;
    }
  }
}

void AstFromJsonPrinter::PrintBuiltinConverter(const BuiltinType& builtin_type,
                                               absl::string_view lang_name) {
  switch (builtin_type.builtin_kind()) {
    case BuiltinTypeKind::kString:
      Print("JsonToString");
      break;
    case BuiltinTypeKind::kBool:
      Print("JsonToBool");
      break;
    case BuiltinTypeKind::kInt64:
      Print("JsonToInt64");
      break;
    case BuiltinTypeKind::kDouble:
      Print("JsonToDouble");
      break;
    default:
      LOG(FATAL) << "Unreachable code.";
  }
}

void AstFromJsonPrinter::PrintEnumConverter(const EnumType& enum_type,
                                            absl::string_view lang_name) {
  auto vars = WithVars({
      {"EnumName", (Symbol(lang_name) + enum_type.name()).ToPascalCase()},
      {"cc_type", enum_type.CcType()},
  });
  Print("Enum<$cc_type$>(StringTo$EnumName$)");
}

void AstFromJsonPrinter::PrintClassConverter(const ClassType& class_type,
                                             absl::string_view lang_name) {
  Print(absl::StrCat(class_type.CcClassName(), "::FromJson"));
}

void AstFromJsonPrinter::PrintVariantConverter(const VariantType& variant_type,
                                               absl::string_view lang_name) {
  // Variant(
  //     VariantOption<double>{
  //         .predicate = IsDouble,
  //         .converter = JsonToDouble,
  //     },
  //     VariantOption<std::string>{
  //         .predicate = IsString,
  //         .converter = JsonToString,
  //     }
  // )

  Println("Variant(");
  {
    auto indent = WithIndent(4);
    TabPrinter tab_printer{{
        .print_separator = [this] { Print(",\n"); },
    }};

    for (const auto& type : variant_type.types()) {
      tab_printer.Print();

      Println(absl::StrCat("VariantOption<", type->CcType(), ">{"));
      {
        auto indent = WithIndent(4);

        Print(".predicate = ");
        PrintTypeCheckerName(*type);
        Println(",");

        Print(".converter = ");
        PrintConverter(*type, lang_name);
        Println(",");
      }
      Print("}");
    }
  }
  Print(")");
}

void AstFromJsonPrinter::PrintListConverter(const ListType& list_type,
                                            absl::string_view lang_name) {
  // List<std::optional<std::unique_ptr<MyClass>>>(
  //     Nullable<std::unique_ptr<MyClass>>(MyClass::FromJson)
  // )

  const NonListType& element_type = list_type.element_type();

  auto vars = WithVars({
      {"nullable_element_cc_type",
       element_type.CcType(list_type.element_maybe_null())},
      {"element_cc_type", element_type.CcType()},
  });

  Println("List<$nullable_element_cc_type$>(");
  {
    auto indent = WithIndent(4);
    if (list_type.element_maybe_null() == MaybeNull::kYes) {
      Println("Nullable<$element_cc_type$>(");
      {
        auto indent = WithIndent(4);
        PrintConverter(element_type, lang_name);
        Println();
      }
      Println(")");
    } else {
      PrintConverter(element_type, lang_name);
      Println();
    }
  }
  Print(")");
}

void AstFromJsonPrinter::PrintFromJsonFunction(const NodeDef& node,
                                               absl::string_view lang_name) {
  auto vars = WithVars({
      {"NodeType", (Symbol(lang_name) + node.name()).ToPascalCase()},
      {"json_variable", kJsonValueVariableName},
  });

  Println("absl::StatusOr<std::unique_ptr<$NodeType$>>");
  Println("$NodeType$::FromJson(const nlohmann::json& $json_variable$) {");
  {
    auto indent = WithIndent();

    const auto check_is_object = UnIndentedSource(R"cc(
      if (!$json_variable$.is_object()) {
        return absl::InvalidArgumentError("JSON is not an object.");
      }
    )cc");
    Println(check_is_object);
    Println();

    if (!node.children().empty()) {
      // This is a non-leaf type.
      // We get the `type` field and dispatch the corresponding FromJson()
      // function.

      Println(
          "MALDOCA_ASSIGN_OR_RETURN"
          "(std::string type, GetType($json_variable$));");
      Println();

      IfStmtPrinter if_stmt_printer(this);
      for (const NodeDef* descendent : node.descendants()) {
        auto vars = WithVars({
            {"DescendentType",
             (Symbol(lang_name) + descendent->name()).ToPascalCase()},
            {"DescendentTypeNoLangName", descendent->name()},
        });
        if_stmt_printer.PrintCase({
            [&] { Print("type == \"$DescendentTypeNoLangName$\""); },
            [&] {
              Println("return $DescendentType$::FromJson($json_variable$);");
            },
        });
      }
      Println();

      Print("return absl::InvalidArgumentError");
      Println(R"((absl::StrCat("Invalid type: ", type));)");

    } else {
      // This is a leaf type.
      // We get all the fields and call the constructor.

      struct NodeFieldPair {
        std::string node_name;
        Symbol field_name;
      };
      std::vector<NodeFieldPair> node_field_pairs;
      for (const NodeDef* ancestor : node.ancestors()) {
        for (const FieldDef& field : ancestor->fields()) {
          node_field_pairs.push_back({ancestor->name(), field.name()});
        }
      }
      for (const FieldDef& field : node.fields()) {
        node_field_pairs.push_back({node.name(), field.name()});
      }

      for (const NodeFieldPair& node_field_pair : node_field_pairs) {
        auto vars = WithVars({
            {"NodeType",
             (Symbol(lang_name) + node_field_pair.node_name).ToPascalCase()},
            {"field_name", node_field_pair.field_name.ToCcVarName()},
            {"FieldName", node_field_pair.field_name.ToPascalCase()},
        });
        Println(
            "MALDOCA_ASSIGN_OR_RETURN(auto $field_name$, "
            "$NodeType$::Get$FieldName$($json_variable$));");
      }

      Println();

      // Call the constructor.
      Print("return absl::make_unique<$NodeType$>(\n");
      {
        auto indent = WithIndent(4);
        TabPrinter tab_printer{{
            .print_separator = [this] { Print(",\n"); },
        }};
        for (const FieldDef* field : node.aggregated_fields()) {
          auto vars = WithVars({
              {"field_name", field->name().ToCcVarName()},
          });

          tab_printer.Print();
          Print("std::move($field_name$)");
        }
      }

      Println(");");
    }
  }
  Println("}");
}

std::string PrintAstFromJson(const AstDef& ast, absl::string_view cc_namespace,
                             absl::string_view ast_path) {
  std::string str;
  {
    google::protobuf::io::StringOutputStream os(&str);
    AstFromJsonPrinter printer(&os);
    printer.PrintAst(ast, cc_namespace, ast_path);
  }

  return str;
}

}  // namespace maldoca
