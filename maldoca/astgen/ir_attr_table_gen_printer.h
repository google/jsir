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

#ifndef MALDOCA_ASTGEN_IR_ATTR_TABLE_GEN_PRINTER_H_
#define MALDOCA_ASTGEN_IR_ATTR_TABLE_GEN_PRINTER_H_

#include <string>

#include "absl/strings/string_view.h"
#include "maldoca/astgen/ast_def.h"
#include "maldoca/astgen/cc_printer_base.h"
#include "google/protobuf/io/zero_copy_stream.h"

namespace maldoca {

class IrAttrTableGenPrinter : public CcPrinterBase {
 public:
  explicit IrAttrTableGenPrinter(google::protobuf::io::ZeroCopyOutputStream* os)
      : CcPrinterBase(os) {}

  void PrintAst(const AstDef& ast, absl::string_view ir_path);

  void PrintNode(const AstDef& ast, const NodeDef& node);

  void PrintArgument(const AstDef& ast, const FieldDef& field);
};

// Prints the "<lang_name>ir_attrs.generated.td" TableGen file.
std::string PrintIrAttrTableGen(const AstDef& ast, absl::string_view ir_path);

}  // namespace maldoca

#endif  // MALDOCA_ASTGEN_IR_ATTR_TABLE_GEN_PRINTER_H_
