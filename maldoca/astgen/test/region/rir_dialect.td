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

#ifndef MALDOCA_ASTGEN_TEST_REGION_RIR_DIALECT_TD_
#define MALDOCA_ASTGEN_TEST_REGION_RIR_DIALECT_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def Rir_Dialect : Dialect {
  let name = "rir";
  let cppNamespace = "::maldoca";

  let description = [{
    The RegionIR, a test IR that has fields embedded in regions. All ops and
    fields are directly mapped from the AST.
  }];

  let useDefaultTypePrinterParser = 1;
}

class Rir_Type<string name, list<Trait> traits = [],
               string baseCppClass = "::mlir::Type">
    : TypeDef<Rir_Dialect, name, traits, baseCppClass> {
  let mnemonic = ?;
}

class Rir_Op<string mnemonic, list<Trait> traits = []> :
    Op<Rir_Dialect, mnemonic, traits>;

def UnknownRegion : Region<CPred<"::maldoca::IsUnknownRegion($_self)">>;

def ExprRegion : Region<CPred<"::maldoca::IsExprRegion($_self)">>;

def ExprsRegion : Region<CPred<"::maldoca::IsExprsRegion($_self)">>;

def StmtRegion : Region<CPred<"::maldoca::IsStmtRegion($_self)">>;

def StmtsRegion : Region<CPred<"::maldoca::IsStmtsRegion($_self)">>;

def RegionIsEmpty : CPred<"$_self.empty()">;

class OptionalRegion<Region region> : Region<
    Or<[region.predicate, RegionIsEmpty]>,
    region.summary
>;

#endif  // MALDOCA_ASTGEN_TEST_REGION_RIR_DIALECT_TD_
