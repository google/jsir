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

#ifndef MALDOCA_ASTGEN_TEST_LAMBDDA_LIR_DIALECT_TD_
#define MALDOCA_ASTGEN_TEST_LAMBDDA_LIR_DIALECT_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def Lair_Dialect : Dialect {
  let name = "lair";
  let cppNamespace = "::maldoca";

  let description = [{
    The LambdaIR, a test IR that models lambda calculus. All ops and fields are
    directly mapped from the AST.

    In LAIR, we showcase ASTGen's ability of putting expression-likes in
    regions.
  }];
  let useDefaultTypePrinterParser = 1;
}

class Lair_Type<string name, list<Trait> traits = [],
                string baseCppClass = "::mlir::Type">
    : TypeDef<Lair_Dialect, name, traits, baseCppClass> {
  let mnemonic = ?;
}

class Lair_Op<string mnemonic, list<Trait> traits = []> :
    Op<Lair_Dialect, mnemonic, traits>;

def ExprRegion : Region<CPred<"::maldoca::IsExprRegion($_self)">>;

#endif  // MALDOCA_ASTGEN_TEST_LAMBDDA_LIR_DIALECT_TD_
