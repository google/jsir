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

#ifndef MALDOCA_ASTGEN_AST_SOURCE_PRINTER_H_
#define MALDOCA_ASTGEN_AST_SOURCE_PRINTER_H_

#include <string>

#include "absl/strings/string_view.h"
#include "maldoca/astgen/ast_def.h"
#include "maldoca/astgen/cc_printer_base.h"
#include "maldoca/astgen/symbol.h"
#include "maldoca/astgen/type.h"
#include "google/protobuf/io/zero_copy_stream.h"

namespace maldoca {

// Printer of the C++ source for the AST.
class AstSourcePrinter : public CcPrinterBase {
 public:
  explicit AstSourcePrinter(google::protobuf::io::ZeroCopyOutputStream* os)
      : CcPrinterBase(os) {}

  // Prints the "ast.generated.cc" file, which includes the definitions of
  // getters and setters of all the AST node classes.
  //
  // - cc_namespace: A namespace separated by "::".
  //   This is used to print C++ namespaces.
  //
  // - ast_path: The directory for the AST code.
  //   "ast.generated.h" is in that directory.
  //   This is used to print the #include.
  void PrintAst(const AstDef& ast, absl::string_view cc_namespace,
                absl::string_view ast_path);

 private:
  void PrintConstructor(const NodeDef& node, absl::string_view lang_name);

  // Prints the string conversion functions.
  //
  // Example:
  //
  //  absl::string_view UnaryOperatorToString(UnaryOperator unary_operator) {
  //    ...
  //  }
  //
  //  absl::StatusOr<UnaryOperator> StringToUnaryOperator(absl::string_view s) {
  //    ...
  //  }
  void PrintEnum(const EnumDef& enum_def, absl::string_view lang_name);

  // Prints the getters and setters of one AST node class.
  void PrintNode(const NodeDef& node, absl::string_view lang_name);

  // Prints the C++ code that returns a value that's compatible with the types
  // `type.CcMutableGetterType()` and `type.CcConstGetterType()`.
  //
  // `cc_expr` is an lvalue expression of the type `type.CcType()`.
  void PrintGetterBody(const std::string& cc_expr, const Type& type);

  // Prints the C++ code that returns a value that's compatible with the types
  // `type.CcMutableGetterType(is_optional)` and
  // `type.CcConstGetterType(is_optional)`.
  //
  // `cc_expr` is an lvalue expression of the type `type.CcType()`.
  void PrintGetterBody(const Symbol& field_name, const Type& type,
                       bool is_optional);

  // Prints the C++ code that sets one field.
  //
  // `field_name` is an lvalue expression that has the type
  // `type.CcType(is_optional)`. We need to set the field `field_name_`.
  void PrintSetterBody(const Symbol& field_name, const Type& type,
                       bool is_optional);
};

std::string PrintAstSource(const AstDef& ast, absl::string_view cc_namespace,
                           absl::string_view ast_path);
}  // namespace maldoca

#endif  // MALDOCA_ASTGEN_AST_SOURCE_PRINTER_H_
