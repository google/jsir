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

#ifndef MALDOCA_ASTGEN_IR_TABLE_GEN_PRINTER_H_
#define MALDOCA_ASTGEN_IR_TABLE_GEN_PRINTER_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "maldoca/astgen/ast_def.h"
#include "maldoca/astgen/cc_printer_base.h"
#include "maldoca/astgen/type.h"
#include "google/protobuf/io/zero_copy_stream.h"

namespace maldoca {

class IrTableGenPrinter : public CcPrinterBase {
 public:
  explicit IrTableGenPrinter(google::protobuf::io::ZeroCopyOutputStream* os)
      : CcPrinterBase(os) {}

  void PrintAst(const AstDef& ast, absl::string_view ir_path);

  // Example:
  //
  // def JsirWithStatementOp : Jsir_Op<
  //     "with_statement", [
  //         JsirStatementOpInterfaceTraits
  //     ]> {
  //   let arguments = (ins
  //     AnyType: $object
  //   );
  //
  //   let regions = (region
  //     AnyRegion: $body
  //   );
  // }
  void PrintNode(const AstDef& ast, const NodeDef& node, FieldKind kind);

  // Returns the arguments that hold a field in an op in MLIR ODS.
  //
  // Format:
  //
  // <TdType>: $<name>
  //
  // See Typd::TdType() for what the MLIR ODS type is for each Type.
  //
  // Example:
  //
  // AnyType: $object
  //
  // A field with a literal attribute (or a split) is held in several arguments,
  // e.g.
  //
  // OptionalAttr<JsirIdentifierAttr>: $literal_property,
  // Optional<AnyType>: $computed_property
  std::vector<std::string> GetArguments(const AstDef& ast, const NodeDef& node,
                                        const FieldDef& field,
                                        FieldKind op_kind);

  // Returns the regions that hold a field in an op in MLIR ODS.
  //
  // Format:
  //
  // <RegionType>: $<name>
  //
  // Example:
  //
  // StmtRegion: $body
  std::vector<std::string> GetRegions(const AstDef& ast, const NodeDef& node,
                                      const FieldDef& field, FieldKind op_kind);

  // If a field is held in several arguments/regions, returns a trait that
  // verifies that exactly one of them is set (at most one, if the field is
  // optional).
  //
  // Example:
  //
  // PredOpTrait<"exactly one of literal_property, computed_property is set",
  //     CPred<"(int(...) + int(...)) == 1">>
  std::optional<std::string> GetExclusiveSlotsTrait(const AstDef& ast,
                                                    const NodeDef& node,
                                                    const FieldDef& field,
                                                    FieldKind op_kind);
};

// Prints the "<lang_name>ir_ops.generated.td" TableGen file.
//
// - ir_path: The directory for the IR code.
//
//   The following files are in that directory:
//   - "<lang_name>ir_dialect.td"
//   - "<lang_name>ir_ops.generated.td"
//   - "interfaces.td"
//
//   This is used to print the includes and header guards.
std::string PrintIrTableGen(const AstDef& ast, absl::string_view ir_path);

}  // namespace maldoca

#endif  // MALDOCA_ASTGEN_IR_TABLE_GEN_PRINTER_H_
