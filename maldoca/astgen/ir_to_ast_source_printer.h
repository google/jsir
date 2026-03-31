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

#ifndef MALDOCA_ASTGEN_IR_TO_AST_SOURCE_PRINTER_H_
#define MALDOCA_ASTGEN_IR_TO_AST_SOURCE_PRINTER_H_

#include <string>

#include "absl/strings/string_view.h"
#include "maldoca/astgen/ast_def.h"
#include "maldoca/astgen/cc_printer_base.h"
#include "maldoca/astgen/type.h"
#include "google/protobuf/io/zero_copy_stream.h"

namespace maldoca {

class IrToAstSourcePrinter : public CcPrinterBase {
 public:
  explicit IrToAstSourcePrinter(google::protobuf::io::ZeroCopyOutputStream* os)
      : CcPrinterBase(os) {}

  void PrintAst(const AstDef& ast, absl::string_view cc_namespace,
                absl::string_view ast_path, absl::string_view ir_path);

  // Prints the Visit<Node>() function.
  void PrintNonLeafNode(const AstDef& ast, const NodeDef& node, FieldKind kind);

  void PrintLeafNode(const AstDef& ast, const NodeDef& node, FieldKind kind);

  void PrintField(const AstDef& ast, const NodeDef& node,
                  const FieldDef& field);

  void PrintRegion(const AstDef& ast, const NodeDef& node,
                   const FieldDef& field);

  void PrintConverter(const AstDef& ast, const Type& type,
                      absl::string_view lang_name, FieldKind kind,
                      MaybeNull maybe_null);

  void PrintBuiltinConverter(const BuiltinType& builtin_type, FieldKind kind);

  void PrintEnumConverter(const EnumType& enum_type,
                          absl::string_view lang_name);

  void PrintClassConverter(const ClassType& class_type,
                           absl::string_view lang_name, FieldKind kind);

  void PrintVariantConverter(const AstDef& ast, const VariantType& variant_type,
                             absl::string_view lang_name, FieldKind kind);

  void PrintListConverter(const AstDef& ast, const ListType& list_type,
                          absl::string_view lang_name, FieldKind kind);
};

std::string PrintIrToAstSource(const AstDef& ast,
                               absl::string_view cc_namespace,
                               absl::string_view ast_path,
                               absl::string_view ir_path);

}  // namespace maldoca

#endif  // MALDOCA_ASTGEN_IR_TO_AST_SOURCE_PRINTER_H_
