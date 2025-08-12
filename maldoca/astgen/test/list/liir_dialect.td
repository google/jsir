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

#ifndef MALDOCA_ASTGEN_TEST_LIST_LIR_DIALECT_TD_
#define MALDOCA_ASTGEN_TEST_LIST_LIR_DIALECT_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def Liir_Dialect : Dialect {
  let name = "liir";
  let cppNamespace = "::maldoca";

  let description = [{
    The ListIR, a test IR that makes extensive use of list types. All ops and
    fields are directly mapped from the AST.
  }];

  let useDefaultTypePrinterParser = 1;
}

class Liir_Type<string name, list<Trait> traits = [],
                string baseCppClass = "::mlir::Type">
    : TypeDef<Liir_Dialect, name, traits, baseCppClass> {
  let mnemonic = ?;
}

class Liir_Op<string mnemonic, list<Trait> traits = []> :
    Op<Liir_Dialect, mnemonic, traits>;

#endif  // MALDOCA_ASTGEN_TEST_LIST_LIR_DIALECT_TD_
