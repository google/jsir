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

#ifndef MALDOCA_ASTGEN_TS_INTERFACE_PRINTER_H_
#define MALDOCA_ASTGEN_TS_INTERFACE_PRINTER_H_

#include <string>

#include "absl/strings/string_view.h"
#include "maldoca/astgen/ast_def.h"
#include "maldoca/astgen/printer_base.h"
#include "google/protobuf/io/zero_copy_stream.h"

namespace maldoca {

// Printer of the TypeScript interface definition for the AST.
//
// Format:
//
// interface ObjectMember <: Node {
//   key: Expression;
//   computed: boolean;
//   decorators?: [ Decorator ];
// }
class TsInterfacePrinter : AstGenPrinterBase {
 public:
  explicit TsInterfacePrinter(google::protobuf::io::ZeroCopyOutputStream* os)
      : AstGenPrinterBase(os) {}

  // Prints the "ast_ts_interface.generated" file.
  //
  // See test cases in test/ for examples.
  void PrintAst(const AstDef& ast);

  // Prints an enum definition.
  //
  // See test cases in test/ for examples.
  void PrintEnum(const EnumDef& enum_def, absl::string_view lang_name);

  // Prints the class declaration for a node.
  //
  // See test cases in test/ for examples.
  void PrintNode(const NodeDef& node);

  // Prints the definition of a field.
  //
  // Format:
  //  <fieldName>: <js_type>
  //  <fieldName>?: <js_type>
  //
  // - fieldName: Printed as camelCase.
  // - js_type: See `Type::JsType()`.
  //
  // Example:
  //  right: Expression
  //  param?: Pattern
  void PrintFieldDef(const FieldDef& field);
};

std::string PrintTsInterface(const AstDef& ast);

}  // namespace maldoca

#endif  // MALDOCA_ASTGEN_TS_INTERFACE_PRINTER_H_
