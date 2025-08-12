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

#ifndef MALDOCA_ASTGEN_TEST_ASSIGN_AIR_DIALECT_TD_
#define MALDOCA_ASTGEN_TEST_ASSIGN_AIR_DIALECT_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def Air_Dialect : Dialect {
  let name = "air";
  let cppNamespace = "::maldoca";

  let description = [{
    The AssignIR, a test IR that models assignments. All ops and fields are
    directly mapped from the AST.

    In AIR, we represent Identifiers with two ops: AirIdentifierOp for rvalue
    and AirIdentifierRefOp for lvalue.
  }];

  let useDefaultTypePrinterParser = 1;
}

class Air_Type<string name, list<Trait> traits = [],
               string baseCppClass = "::mlir::Type">
    : TypeDef<Air_Dialect, name, traits, baseCppClass> {
  let mnemonic = ?;
}

class Air_Op<string mnemonic, list<Trait> traits = []> :
    Op<Air_Dialect, mnemonic, traits>;

#endif  // MALDOCA_ASTGEN_TEST_ASSIGN_AIR_DIALECT_TD_
