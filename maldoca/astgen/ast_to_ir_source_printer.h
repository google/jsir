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

#ifndef MALDOCA_ASTGEN_AST_TO_IR_SOURCE_PRINTER_H_
#define MALDOCA_ASTGEN_AST_TO_IR_SOURCE_PRINTER_H_

#include <string>

#include "absl/strings/string_view.h"
#include "maldoca/astgen/ast_def.h"
#include "maldoca/astgen/cc_printer_base.h"
#include "maldoca/astgen/symbol.h"
#include "maldoca/astgen/type.h"
#include "google/protobuf/io/zero_copy_stream.h"

namespace maldoca {

class AstToIrSourcePrinter : public CcPrinterBase {
 public:
  explicit AstToIrSourcePrinter(google::protobuf::io::ZeroCopyOutputStream* os)
      : CcPrinterBase(os) {}

  // Action: What to do with the converted IR value/attribute.
  //
  // - Def: Define a variable.
  // - Assign: Assign the value/attribute to an existing variable.
  // - Create: Just create the value/attribute and ignore it.
  //
  // See comments for Print*ToIr for more details.
  enum class Action {
    kDef,
    kAssign,
    kCreate,
  };

  // Whether a C++ expression refers to a "reference" or a "value".
  //
  // Consider the following AST node:
  // class CallExpression : ... {
  //  public:
  //   const Expression *func() const;
  //   const std::vector<std::unique_ptr<Expression>> *args() const;
  // };
  //
  // - The type of func() is "const Expression *".
  //   We consider this a "reference".
  //
  // - The type of args()[0] is "std::unique_ptr<Expression> &".
  //   We consider this a "value".
  //
  // However, in the ASTGen type system, we refer them both as
  // ClassType{"Expression"}. Therefore, we need this additional enum to make
  // the distinction.
  //
  // If a function takes a "reference" but we have a "value", we need to call
  // ".get()" to turn it into a "reference".
  enum RefOrVal {
    kRef,
    kVal,
  };

  void PrintAst(const AstDef& ast, absl::string_view cc_namespace,
                absl::string_view ast_path, absl::string_view ir_path);

  // Prints the Visit<OpName>() function.
  void PrintNonLeafNode(const AstDef& ast, const NodeDef& node, FieldKind kind);

  void PrintLeafNode(const AstDef& ast, const NodeDef& node, FieldKind kind);

  // ===========================================================================
  // Print*ToIr
  // ===========================================================================
  //
  // Prints the conversion of a C++ expression that represents a field from the
  // AST to the corresponding MLIR value/attribute. The result is later used to
  // build MLIR ops.
  // - rhs: The original C++ expression that represents a field from the AST.
  //
  // - lhs: The name of the variable to assign to or create, after the
  //        conversion.
  //
  // - action:
  //   - kDef:
  //     mlir::Value <lhs> = Convert(<rhs>);
  //   - kAssign:
  //     <lhs> = Convert(<rhs>);
  //   - kCreate:
  //     Convert(<rhs>);
  //
  // - type: The type of the AST field.
  //
  // - ref_or_val: See comments for RefOrVal.
  //
  // - kind: Kind of the field. See comments for FieldKind.
  //   If kind == FIELD_KIND_LVAL, then we need to append "Ref" to the op name.
  void PrintBuiltinToIr(const AstDef& ast, Action action,
                        const BuiltinType& type, const Symbol& lhs,
                        const std::string& rhs);

  void PrintClassToIr(const AstDef& ast, Action action, const ClassType& type,
                      FieldKind kind, const Symbol& lhs,
                      const std::string& rhs);

  void PrintClassToIr(const AstDef& ast, Action action, const ClassType& type,
                      RefOrVal ref_or_val, FieldKind kind, const Symbol& lhs,
                      const std::string& rhs);

  void PrintEnumToIr(const AstDef& ast, Action action, const EnumType& type,
                     const Symbol& lhs, const std::string& rhs);

  void PrintVariantToIr(const AstDef& ast, Action action,
                        const VariantType& type, RefOrVal ref_or_val,
                        FieldKind kind, const Symbol& lhs,
                        const std::string& rhs);

  void PrintListToIr(const AstDef& ast, Action action, const ListType& type,
                     FieldKind kind, const Symbol& lhs, const std::string& rhs);

  void PrintToIr(const AstDef& ast, Action action, const Type& type,
                 RefOrVal ref_or_val, FieldKind kind, const Symbol& lhs,
                 const std::string& rhs);

  void PrintNullableToIr(const AstDef& ast, Action action, const Type& type,
                         MaybeNull maybe_null, RefOrVal ref_or_val,
                         FieldKind kind, const Symbol& lhs,
                         const std::string& rhs);

  // Prints the code that converts an AST field to an MLIR value/attribute and
  // stores the result in a new variable.
  //
  // Format:
  //
  // <TdType> mlir_<field_name> = Visit<Type>(node-><field_name>());
  //
  // Example:
  //
  // mlir::Value mlir_object = VisitExpression(node->object());
  void PrintField(const AstDef& ast, const NodeDef& node,
                  const FieldDef& field);

  // Prints the code that converts an AST field to a region. The region has been
  // created and the code just populates blocks and ops in it.
  //
  // Format:
  //
  // mlir::Region &mlir_<field_name>_region = op.<field_name>();
  // AppendNewBlockAndPopulate(mlir_<field_name>_region, [&] {
  //   <Converts node->foo() into elements in the region.>
  // });
  //
  // Example:
  //
  // mlir::Region &mlir_body_region = op.body();
  // AppendNewBlockAndPopulate(mlir_body_region, [&] {
  //   for (const auto &element : *node->body()) {
  //     VisitStatement(element.get());
  //   }
  // });
  void PrintRegion(const AstDef& ast, const NodeDef& node,
                   const FieldDef& field);
};

// Prints the "ast_to<lang_name>ir.generated.cc" file.
//
// - cc_namespace: The namespace where all IR op classes live.
//
// - ast_path: The directory for the AST code.
//
//   "ast.generated.h" is in that directory.
//
//   This is used to print the #includes.
//
// - ir_path: The directory for the IR code.
//
//   The following files are in that directory:
//   - "<lang_name>ir_dialect.td"
//   - "<lang_name>ir_ops.generated.td"
//   - "interfaces.td"
//   - "conversion/ast_to_<lang_name>ir.h"
//   - "conversion/ast_to_<lang_name>ir.generated.cc"
//
//   This is used to print the #includes and header guards.
std::string PrintAstToIrSource(const AstDef& ast,
                               absl::string_view cc_namespace,
                               absl::string_view ast_path,
                               absl::string_view ir_path);

}  // namespace maldoca

#endif  // MALDOCA_ASTGEN_AST_TO_IR_SOURCE_PRINTER_H_
