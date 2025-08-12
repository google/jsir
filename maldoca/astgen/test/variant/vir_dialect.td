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

#ifndef MALDOCA_ASTGEN_TEST_VARIANT_VIR_DIALECT_TD_
#define MALDOCA_ASTGEN_TEST_VARIANT_VIR_DIALECT_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def Vir_Dialect : Dialect {
  let name = "vir";
  let cppNamespace = "::maldoca";

  let description = [{
    The VariantIR, a test IR that makes extensive use of variant types. All ops
    and fields are directly mapped from the AST.
  }];

  let useDefaultTypePrinterParser = 1;
}

class Vir_Type<string name, list<Trait> traits = [],
               string baseCppClass = "::mlir::Type">
    : TypeDef<Vir_Dialect, name, traits, baseCppClass> {
  let mnemonic = ?;
}

class Vir_Op<string mnemonic, list<Trait> traits = []> :
    Op<Vir_Dialect, mnemonic, traits>;

#endif  // MALDOCA_ASTGEN_TEST_VARIANT_VIR_DIALECT_TD_
