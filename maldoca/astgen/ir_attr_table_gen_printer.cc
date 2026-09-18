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

#include "maldoca/astgen/ir_attr_table_gen_printer.h"

#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "maldoca/astgen/ast_def.h"
#include "maldoca/astgen/ast_gen_utils.h"
#include "maldoca/astgen/symbol.h"
#include "maldoca/astgen/type.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"

namespace maldoca {

void IrAttrTableGenPrinter::PrintAst(const AstDef& ast,
                                     absl::string_view ir_path) {
  PrintLicense();
  Println();

  PrintCodeGenerationWarning();
  Println();

  const auto ir_name = absl::StrCat(ast.lang_name(), "ir");
  const auto td_path =
      absl::StrCat(ir_path, "/", ir_name, "_attrs.generated.td");

  PrintEnterHeaderGuard(td_path);
  Println();

  std::vector<std::string> imports = {
      "mlir/IR/AttrTypeBase.td",
      "mlir/IR/OpBase.td",
      absl::StrCat(ir_path, "/interfaces.td"),
      absl::StrCat(ir_path, "/", ast.lang_name(), "ir_dialect.td"),
  };
  for (const auto& import : imports) {
    Println(absl::StrCat("include \"", import, "\""));
  }
  Println();
  for (const auto* node : ast.topological_sorted_nodes()) {
    if (!node->should_generate_ir_attr()) {
      continue;
    }
    PrintNode(ast, *node);
  }

  PrintExitHeaderGuard(td_path);
}

void IrAttrTableGenPrinter::PrintNode(const AstDef& ast, const NodeDef& node) {
  auto ir_name = absl::StrCat(ast.lang_name(), "ir");
  auto IrName = Symbol(ir_name).ToPascalCase();
  auto AttrName = node.ir_attr_name(ast.lang_name());
  auto attr_mnemonic = node.ir_attr_mnemonic();

  auto AttrBaseName = AttrName.ToPascalCase();
  if (absl::EndsWith(AttrBaseName, "Attr")) {
    AttrBaseName = AttrBaseName.substr(0, AttrBaseName.size() - 4);
  }

  std::vector<std::string> traits;
  traits.reserve(node.additional_attr_traits().size());
  for (const auto& trait : node.additional_attr_traits()) {
    traits.push_back(trait);
  }

  auto vars = WithVars({
      {"AttrName", AttrName.ToPascalCase()},
      {"AttrBaseName", AttrBaseName},
      {"attr_mnemonic", attr_mnemonic.ToCcVarName()},
      {"IrName", IrName},
  });

  if (traits.empty()) {
    Println(
        "def $AttrName$ : AttrDef<$IrName$_Dialect, \"$AttrBaseName$\", []> {");
  } else {
    Print(
        "def $AttrName$ : AttrDef<\n"
        "    $IrName$_Dialect, \"$AttrBaseName$\", [\n");
    {
      auto indent = WithIndent(8);
      TabPrinter tab_printer{{
          .print_separator = [&] { Print(",\n"); },
      }};
      for (const auto& trait : traits) {
        tab_printer.Print();
        Print(trait);
      }
    }
    Println("\n    ]> {");
  }

  {
    auto indent = WithIndent();
    Println("let mnemonic = \"$attr_mnemonic$\";");

    std::vector<const FieldDef*> fields;
    for (const auto* field : node.aggregated_fields()) {
      if (field->in_ir()) {
        fields.push_back(field);
      }
    }

    bool has_loc = node.ir_attr_has_loc();
    bool has_fields = !fields.empty();

    if (has_loc || has_fields) {
      if (has_loc && !has_fields) {
        Println("let parameters = (ins");
        Println("  OptionalParameter<\"$IrName$TriviaAttr\">: $$loc");
        Println(");");
      } else if (has_loc && has_fields) {
        Println("let parameters = (ins");
        {
          auto indent2 = WithIndent();
          Println("OptionalParameter<\"$IrName$TriviaAttr\">: $$loc,");
          TabPrinter separator_printer{{
              .print_separator = [&] { Print(",\n"); },
          }};
          for (const auto* field : fields) {
            separator_printer.Print();
            PrintArgument(ast, *field);
          }
          Println();
        }
        Println(");");
      } else {
        Println("let parameters = (ins");
        {
          auto indent2 = WithIndent();
          TabPrinter separator_printer{{
              .print_separator = [&] { Print(",\n"); },
          }};
          for (const auto* field : fields) {
            separator_printer.Print();
            PrintArgument(ast, *field);
          }
          Println();
        }
        Println(");");
      }
      Println("let assemblyFormat = \"params\";");
    }
  }

  Println("}");
  Println();
}

void IrAttrTableGenPrinter::PrintArgument(const AstDef& ast,
                                          const FieldDef& field) {
  // The type, without quotes. This is either the name of an `AttrDef` record or
  // a C++ type; the two are spelled differently in TableGen, hence the flag.
  std::string type_name;
  bool is_attr_def = false;
  // List fields expand to a complete parameter spec rather than a bare type.
  std::string list_parameter;

  if (field.type().IsA<BuiltinType>()) {
    const auto& builtin = static_cast<const BuiltinType&>(field.type());
    switch (builtin.builtin_kind()) {
      case BuiltinTypeKind::kString:
        type_name = "::mlir::StringAttr";
        break;
      case BuiltinTypeKind::kDouble:
        type_name = "::mlir::FloatAttr";
        break;
      case BuiltinTypeKind::kInt64:
        if (field.optionalness() == OPTIONALNESS_MAYBE_UNDEFINED ||
            field.optionalness() == OPTIONALNESS_MAYBE_NULL) {
          type_name = "std::optional<int64_t>";
        } else {
          type_name = "int64_t";
        }
        break;
      case BuiltinTypeKind::kBool:
        type_name = "::mlir::BoolAttr";
        break;
    }
  } else if (field.type().IsA<ClassType>()) {
    const auto& class_type = static_cast<const ClassType&>(field.type());
    type_name = (Symbol(absl::StrCat(ast.lang_name(), "ir")) +
                 class_type.name() + "Attr")
                    .ToPascalCase();
    is_attr_def = true;
  } else if (field.type().IsA<VariantType>()) {
    type_name = "::mlir::Attribute";
  } else if (field.type().IsA<ListType>()) {
    const auto& list_type = static_cast<const ListType&>(field.type());
    std::string element_type = "int64_t";
    if (list_type.element_type().IsA<ClassType>()) {
      const auto& elem_class =
          static_cast<const ClassType&>(list_type.element_type());
      element_type = (Symbol(absl::StrCat(ast.lang_name(), "ir")) +
                      elem_class.name() + "Attr")
                         .ToPascalCase();
    }
    list_parameter =
        absl::StrCat("OptionalArrayRefParameter<\"", element_type, "\">");
  } else {
    type_name = "::mlir::Attribute";
  }

  std::string type_str;
  if (!list_parameter.empty()) {
    // `OptionalArrayRefParameter` is already optional, so
    // `ir_attr_optional_parameter` does not apply.
    type_str = list_parameter;
  } else if (field.ir_attr_optional_parameter()) {
    type_str = absl::StrCat("OptionalParameter<\"", type_name, "\">");
  } else if (is_attr_def) {
    type_str = type_name;
  } else {
    type_str = absl::StrCat("\"", type_name, "\"");
  }

  auto vars = WithVars({
      {"type", type_str},
      {"name", field.name().ToCcVarName()},
  });
  Print("$type$: $$$name$");
}

std::string PrintIrAttrTableGen(const AstDef& ast, absl::string_view ir_path) {
  std::string result;
  google::protobuf::io::StringOutputStream os(&result);
  IrAttrTableGenPrinter printer(&os);
  printer.PrintAst(ast, ir_path);
  return result;
}

}  // namespace maldoca
