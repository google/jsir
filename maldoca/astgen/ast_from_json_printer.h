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

#ifndef MALDOCA_ASTGEN_AST_FROM_JSON_PRINTER_H_
#define MALDOCA_ASTGEN_AST_FROM_JSON_PRINTER_H_

#include <string>

#include "absl/strings/string_view.h"
#include "maldoca/astgen/ast_def.h"
#include "maldoca/astgen/cc_printer_base.h"
#include "maldoca/astgen/type.h"
#include "google/protobuf/io/zero_copy_stream.h"

namespace maldoca {

class AstFromJsonPrinter : public CcPrinterBase {
 public:
  explicit AstFromJsonPrinter(google::protobuf::io::ZeroCopyOutputStream* os)
      : CcPrinterBase(os) {}

  void PrintAst(const AstDef& ast, absl::string_view cc_namespace,
                absl::string_view ast_path);

 private:
  void PrintTypeChecker(const NodeDef& node);

  void PrintGetFieldFunction(const std::string& node_name,
                             const FieldDef& field,
                             absl::string_view lang_name);

  void PrintFromJsonFunction(const NodeDef& node, absl::string_view lang_name);

  void PrintTypeCheckerName(const ScalarType& type);

  void PrintConverter(const Type& type, absl::string_view lang_name);

  void PrintBuiltinConverter(const BuiltinType& builtin_type,
                             absl::string_view lang_name);

  void PrintEnumConverter(const EnumType& enum_type,
                          absl::string_view lang_name);

  void PrintClassConverter(const ClassType& class_type,
                           absl::string_view lang_name);

  void PrintVariantConverter(const VariantType& variant_type,
                             absl::string_view lang_name);

  void PrintListConverter(const ListType& list_type,
                          absl::string_view lang_name);
};

std::string PrintAstFromJson(const AstDef& ast, absl::string_view cc_namespace,
                             absl::string_view ast_path);

}  // namespace maldoca

#endif  // MALDOCA_ASTGEN_AST_FROM_JSON_PRINTER_H_
