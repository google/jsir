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

#include <string>

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

  // Prints an argument for an op in MLIR ODS.
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
  void PrintArgument(const AstDef& ast, const NodeDef& node,
                     const FieldDef& field);

  // Prints a region in an op in MLIR ODS.
  //
  // Format:
  //
  // AnyRegion: $<name>
  //
  // Example:
  //
  // AnyRegion: $body
  void PrintRegion(const AstDef& ast, const NodeDef& node,
                   const FieldDef& field);
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
