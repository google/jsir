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

#ifndef MALDOCA_ASTGEN_AST_HEADER_PRINTER_H_
#define MALDOCA_ASTGEN_AST_HEADER_PRINTER_H_

#include <string>

#include "absl/strings/string_view.h"
#include "maldoca/astgen/ast_def.h"
#include "maldoca/astgen/cc_printer_base.h"
#include "google/protobuf/io/zero_copy_stream.h"

namespace maldoca {

// Printer of the C++ header for the AST.
class AstHeaderPrinter : public CcPrinterBase {
 public:
  explicit AstHeaderPrinter(google::protobuf::io::ZeroCopyOutputStream* os)
      : CcPrinterBase(os) {}

  // Prints the "ast.generated.h" header file.
  //
  // - cc_namespace: The C++ namespace for all the AST node classes.
  //   Example: "maldoca::astgen".
  //
  // - ast_path: The directory for the AST code.
  //   "ast.generated.h" is in that directory.
  //   This is used to generate the header guard.
  //
  // See test cases in test/ for examples.
  void PrintAst(const AstDef& ast, absl::string_view cc_namespace,
                absl::string_view ast_path);

  // Prints the enum definition and the prototypes of string conversion
  // functions.
  //
  // Example:
  //  enum UnaryOperator {
  //    kMinus,
  //    ...
  //  };
  //
  //  absl::string_view UnaryOperatorToString(UnaryOperator unary_operator);
  //  absl::StatusOr<UnaryOperator> StringToUnaryOperator(absl::string_view s);
  void PrintEnum(const EnumDef& enum_def, absl::string_view lang_name);

  // Prints the class declaration for a node.
  //
  // See test cases in test/ for examples.
  void PrintNode(const NodeDef& node, absl::string_view lang_name);

  // Prints the constructor of a node class.
  //
  // Example:
  //  explicit Variable(std::string identifier)
  //      : Expression(), identifier_(std::move(identifier)) {}
  void PrintConstructor(const NodeDef& node, absl::string_view lang_name);

  // Prints the getter and setter declarations for a field.
  //
  // Format:
  //  <cc_mutable_getter_type> <field_name>();
  //  <cc_const_getter_type> <field_name>() const;
  //  void set_<field_name>(<cc_type> <field_name>);
  //
  // - cc_mutable_getter_type: See `Type::CcMutableGetterType()`.
  // - cc_const_getter_type: See `Type::CcConstGetterType()`.
  // - cc_type: See `Type::CcType()`.
  //
  // Example:
  //  Expression* right();
  //  const Expression* right() const;
  //  void set_right(std::unique_ptr<Expression> right);
  void PrintGetterSetterDeclarations(const FieldDef& field,
                                     absl::string_view lang_name);

  // Prints a member variable declaration.
  //
  // Format:
  //  <cc_type> <field_name>_;
  //
  // - cc_type: The C++ value type. See `Type::CcType()`.
  // - field_name_: We print the name in snake_case and add a '_'.
  //
  // Example:
  //  std::unique_ptr<Expression> right_;
  void PrintMemberVariable(const FieldDef& field, absl::string_view lang_name);

  // Format:
  //  static absl::StatusOr<<cc_type>>
  //  Get<FieldName>FromJson(const nlohmann::json& json);
  //
  // Example:
  //  static absl::StatusOr<std::unique_ptr<Expression>>
  //  GetRightFromJson(const nlohmann::json& json);
  void PrintGetFromJson(const FieldDef& field, absl::string_view lang_name);
};

// Prints the "ast.generated.h" header file.
//
// - cc_namespace: The C++ namespace for all the AST node classes.
//   Example: "maldoca::astgen".
//
// - ast_path: The directory for the AST code.
//   "ast.generated.h" is in that directory.
//   This is used to generate the header guard.
std::string PrintAstHeader(const AstDef& ast, absl::string_view cc_namespace,
                           absl::string_view ast_path);

}  // namespace maldoca

#endif  // MALDOCA_ASTGEN_AST_HEADER_PRINTER_H_
