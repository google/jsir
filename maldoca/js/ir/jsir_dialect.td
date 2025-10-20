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

#ifndef MALDOCA_JS_IR_JSIR_DIALECT_TD_
#define MALDOCA_JS_IR_JSIR_DIALECT_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def Jsir_Dialect : Dialect {
  let name = "jsir";
  let cppNamespace = "::maldoca";

  let hasConstantMaterializer = 1;
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

// Base class for JSIR dialect types.
class Jsir_Type<string name, list<Trait> traits = [],
                string baseCppClass = "::mlir::Type">
    : TypeDef<Jsir_Dialect, name, traits, baseCppClass> {
  let mnemonic = ?;
}

// Base class for JSIR dialect operations.
// This operation inherits from the base `Op` class in OpBase.td, and provides:
// - The parent dialect of the operation.
// - The mnemonic for the operation, or the name without the dialect prefix.
// - A list of traits for the operation.
class Jsir_Op<string mnemonic, list<Trait> traits = []> :
    Op<Jsir_Dialect, mnemonic, traits>;

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

def Jshir_Dialect : Dialect {
  let name = "jshir";
  let cppNamespace = "::maldoca";
}

// Base class for JSHIR dialect operations.
// This operation inherits from the base `Op` class in OpBase.td, and provides:
// - The parent dialect of the operation.
// - The mnemonic for the operation, or the name without the dialect prefix.
// - A list of traits for the operation.
class Jshir_Op<string mnemonic, list<Trait> traits = []> :
    Op<Jshir_Dialect, mnemonic, traits>;

#endif  // MALDOCA_JS_IR_JSIR_DIALECT_TD_
