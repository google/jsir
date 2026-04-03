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

#include "maldoca/astgen/ts_interface_printer.h"

#include <string>

#include "absl/strings/escaping.h"
#include "absl/strings/string_view.h"
#include "maldoca/astgen/ast_def.h"
#include "maldoca/astgen/ast_def.pb.h"
#include "maldoca/astgen/ast_gen_utils.h"
#include "maldoca/astgen/type.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"

namespace maldoca {

void TsInterfacePrinter::PrintAst(const AstDef& ast) {
  for (const EnumDef& enum_def : ast.enum_defs()) {
    PrintEnum(enum_def, ast.lang_name());
    Println();
  }

  for (const auto& name : ast.node_names()) {
    const NodeDef& node = *ast.nodes().at(name);
    PrintNode(node);
    Println();
  }
}

void TsInterfacePrinter::PrintEnum(const EnumDef& enum_def,
                                   absl::string_view lang_name) {
  auto vars = WithVars({
      {"EnumName", enum_def.name().ToPascalCase()},
  });

  Println("type $EnumName$ =");
  {
    auto indent = WithIndent(4);
    for (const EnumMemberDef& member : enum_def.members()) {
      auto vars = WithVars({
          {"string_value", absl::CEscape(member.string_value())},
      });

      Println("| \"$string_value$\"");
    }
  }
}

void TsInterfacePrinter::PrintNode(const NodeDef& node) {
  auto vars = WithVars({
      {"NodeType", node.name()},
  });
  Print("interface $NodeType$");

  if (!node.parents().empty()) {
    Print(" <: ");

    TabPrinter separator_printer{{
        .print_separator = [&] { Print(", "); },
    }};
    for (const NodeDef* parent : node.parents()) {
      separator_printer.Print();
      Print(parent->name());
    }
  }

  Println(" {");
  {
    auto indent = WithIndent();
    for (const FieldDef& field : node.fields()) {
      PrintFieldDef(field);
    }
  }
  Println("}");
}

void TsInterfacePrinter::PrintFieldDef(const FieldDef& field) {
  Print(field.name().ToCamelCase());

  if (field.optionalness() == OPTIONALNESS_MAYBE_UNDEFINED) {
    Print("?");
  }

  Print(": ");

  MaybeNull maybe_null = field.optionalness() == OPTIONALNESS_MAYBE_NULL
                             ? MaybeNull::kYes
                             : MaybeNull::kNo;
  Print(field.type().JsType(maybe_null));

  Println();
}

std::string PrintTsInterface(const AstDef& ast) {
  std::string ts_interface;
  {
    google::protobuf::io::StringOutputStream os(&ts_interface);
    TsInterfacePrinter printer(&os);
    printer.PrintAst(ast);
  }
  return ts_interface;
}

}  // namespace maldoca
