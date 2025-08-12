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

#include "maldoca/astgen/test/lambda/ir.h"

// IWYU pragma: begin_keep

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Region.h"

// IWYU pragma: end_keep

// =============================================================================
// Dialect Definition
// =============================================================================

#include "maldoca/astgen/test/lambda/lair_dialect.cc.inc"

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void maldoca::LairDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "maldoca/astgen/test/lambda/lair_types.cc.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "maldoca/astgen/test/lambda/lair_ops.cc.inc"
      >();
}

// =============================================================================
// Dialect Interface Definitions
// =============================================================================

#include "maldoca/astgen/test/lambda/interfaces.cc.inc"

// =============================================================================
// Dialect Type Definitions
// =============================================================================

#define GET_TYPEDEF_CLASSES
#include "maldoca/astgen/test/lambda/lair_types.cc.inc"

// =============================================================================
// Dialect Op Definitions
// =============================================================================

#define GET_OP_CLASSES
#include "maldoca/astgen/test/lambda/lair_ops.cc.inc"

// =============================================================================
// Utils
// =============================================================================

namespace maldoca {

bool IsExprRegion(mlir::Region &region) {
  // Region must have exactly one block.
  if (!llvm::hasSingleElement(region)) {
    return false;
  }

  mlir::Block &block = region.front();

  // Block must have at least one op (terminator).
  if (block.empty()) {
    return false;
  }

  auto *terminator = &block.back();
  return llvm::isa<LairExprRegionEndOp>(terminator);
}

}  // namespace maldoca
