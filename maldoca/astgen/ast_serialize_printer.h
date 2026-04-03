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

#ifndef MALDOCA_ASTGEN_AST_SERIALIZE_PRINTER_H_
#define MALDOCA_ASTGEN_AST_SERIALIZE_PRINTER_H_

#include <string>

#include "absl/strings/string_view.h"
#include "maldoca/astgen/ast_def.h"
#include "maldoca/astgen/cc_printer_base.h"
#include "maldoca/astgen/type.h"
#include "google/protobuf/io/zero_copy_stream.h"

namespace maldoca {

// Printer of the C++ Serialize() function for the AST.
class AstSerializePrinter : public CcPrinterBase {
 public:
  explicit AstSerializePrinter(google::protobuf::io::ZeroCopyOutputStream* os)
      : CcPrinterBase(os) {}

  void PrintAst(const AstDef& ast, absl::string_view cc_namespace,
                absl::string_view ast_path);

 private:
  // Print*Serialize()
  //
  // Prints either:
  // - An assignment "<lhs> = ConvertSerialize(<rhs>);", or
  // - A variable definition "nlohmann::json <lhs> = ConvertSerialize(<rhs>);"
  //
  // - lhs: If printing an assignment, an lvalue expression of type
  //        nlohmann::json; if printing a variable definition, the name of that
  //        variable.
  // - rhs: An expression of type `type.CcType()`.
  void PrintBuiltinSerialize(const BuiltinType& type, const std::string& lhs,
                             const std::string& rhs);

  void PrintEnumSerialize(const EnumType& type, const std::string& lhs,
                          const std::string& rhs, absl::string_view lang_name);

  void PrintClassSerialize(const ClassType& type, const std::string& lhs,
                           const std::string& rhs);

  void PrintVariantSerialize(const VariantType& variant_type,
                             const std::string& lhs, const std::string& rhs,
                             absl::string_view lang_name);

  void PrintListSerialize(const ListType& list_type, const std::string& lhs,
                          const std::string& rhs, absl::string_view lang_name);

  void PrintSerialize(const Type& type, const std::string& lhs,
                      const std::string& rhs, absl::string_view lang_name);

  void PrintNullableToJson(const Type& type, MaybeNull maybe_null,
                           const std::string& lhs, const std::string& rhs,
                           absl::string_view lang_name);

  void PrintSerializeFieldsFunction(const NodeDef& node,
                                    absl::string_view lang_name);

  void PrintSerializeFunction(const NodeDef& node, absl::string_view lang_name);

  void PrintSerializeFunctionOverload(const NodeDef& node,
                                      absl::string_view lang_name);
};

// Prints the "ast_to_json.generated.cc" source file.
//
// - cc_namespace: The C++ namespace for all the AST node classes.
//   Example: "maldoca::astgen".
//
// - ast_path: The directory for the AST code.
//   "ast.generated.h" is in that directory.
//   This is used to print the #include.
std::string PrintAstToJson(const AstDef& ast, absl::string_view cc_namespace,
                           absl::string_view ast_path);

}  // namespace maldoca

#endif  // MALDOCA_ASTGEN_AST_SERIALIZE_PRINTER_H_
