// Copyright 2026 Google LLC
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

#ifndef MALDOCA_ASTGEN_TEST_SPLIT_SIR_DIALECT_TD_
#define MALDOCA_ASTGEN_TEST_SPLIT_SIR_DIALECT_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def Sir_Dialect : Dialect {
  let name = "sir";
  let cppNamespace = "::maldoca";

  let description = [{
    The SplitIR, a test IR whose ops hold some AST fields in several
    arguments/regions (see FieldIrMappingPb in ast_def.proto).
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

class Sir_Type<string name, list<Trait> traits = [],
               string baseCppClass = "::mlir::Type">
    : TypeDef<Sir_Dialect, name, traits, baseCppClass> {
  let mnemonic = ?;
}

class Sir_Op<string mnemonic, list<Trait> traits = []> :
    Op<Sir_Dialect, mnemonic, traits>;

def ExprRegion : Region<CPred<"::maldoca::IsExprRegion($_self)">>;

def StmtRegion : Region<CPred<"::maldoca::IsStmtRegion($_self)">>;

def RegionIsEmpty : CPred<"$_self.empty()">;

class OptionalRegion<Region region> : Region<
    Or<[region.predicate, RegionIsEmpty]>,
    region.summary
>;

#endif  // MALDOCA_ASTGEN_TEST_SPLIT_SIR_DIALECT_TD_
