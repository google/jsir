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

#ifndef MALDOCA_ASTGEN_TEST_VARIANT_IR_H_
#define MALDOCA_ASTGEN_TEST_VARIANT_IR_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

// Include the auto-generated header file containing the declaration of the VIR
// dialect.
#include "maldoca/astgen/test/variant/vir_dialect.h.inc"

// Include the auto-generated header file containing the declarations of the VIR
// interfaces.
#include "maldoca/astgen/test/variant/interfaces.h.inc"

// Include the auto-generated header file containing the declarations of the VIR
// types.
#define GET_TYPEDEF_CLASSES
#include "maldoca/astgen/test/variant/vir_types.h.inc"

// Include the auto-generated header file containing the declarations of the VIR
// operations.
#define GET_OP_CLASSES
#include "maldoca/astgen/test/variant/vir_ops.generated.h.inc"

#endif  // MALDOCA_ASTGEN_TEST_VARIANT_IR_H_
