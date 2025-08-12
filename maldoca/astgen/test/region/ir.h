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

#ifndef MALDOCA_ASTGEN_TEST_REGION_IR_H_
#define MALDOCA_ASTGEN_TEST_REGION_IR_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

namespace maldoca {

// Checks that region contains a single block. There is no restriction on the
// block. This means that this region can be either of the four kinds below.
bool IsUnknownRegion(mlir::Region &region);

// Checks that the region contains a single block that terminates with
// JsirExprRegionEnd. This means that this region calculates a single
// expression.
bool IsExprRegion(mlir::Region &region);

// Checks that the region contains a single block that terminates with
// JsirExprsRegionEnd. This means that this region calculates a list of
// expressions.
bool IsExprsRegion(mlir::Region &region);

// Checks that the region contains a single block that's non empty. This means
// that this region contains a statement.
bool IsStmtRegion(mlir::Region &region);

// Checks that the region contains a single block. There is no restriction on
// the block. This means that this region contains a list of statements (the
// most unrestrictive case).
bool IsStmtsRegion(mlir::Region &region);

}  // namespace maldoca

// Include the auto-generated header file containing the declaration of the RIR
// dialect.
#include "maldoca/astgen/test/region/rir_dialect.h.inc"

// Include the auto-generated header file containing the declarations of the RIR
// interfaces.
#include "maldoca/astgen/test/region/interfaces.h.inc"

// Include the auto-generated header file containing the declarations of the RIR
// types.
#define GET_TYPEDEF_CLASSES
#include "maldoca/astgen/test/region/rir_types.h.inc"

// Include the auto-generated header file containing the declarations of the RIR
// operations.
#define GET_OP_CLASSES
#include "maldoca/astgen/test/region/rir_ops.h.inc"

#endif  // MALDOCA_ASTGEN_TEST_REGION_IR_H_
