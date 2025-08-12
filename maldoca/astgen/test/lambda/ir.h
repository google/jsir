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

#ifndef MALDOCA_ASTGEN_TEST_LAMBDA_IR_H_
#define MALDOCA_ASTGEN_TEST_LAMBDA_IR_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

namespace maldoca {

// Checks that the region contains a single block that terminates with
// LairExprRegionEnd. This means that this region calculates a single
// expression.
bool IsExprRegion(mlir::Region &region);

}  // namespace maldoca

// Include the auto-generated header file containing the declaration of the LAIR
// dialect.
#include "maldoca/astgen/test/lambda/lair_dialect.h.inc"

// Include the auto-generated header file containing the declarations of the
// LAIR interfaces.
#include "maldoca/astgen/test/lambda/interfaces.h.inc"

// Include the auto-generated header file containing the declarations of the
// LAIR types.
#define GET_TYPEDEF_CLASSES
#include "maldoca/astgen/test/lambda/lair_types.h.inc"

// Include the auto-generated header file containing the declarations of the
// LAIR operations.
#define GET_OP_CLASSES
#include "maldoca/astgen/test/lambda/lair_ops.h.inc"

#endif  // MALDOCA_ASTGEN_TEST_LAMBDA_IR_H_
