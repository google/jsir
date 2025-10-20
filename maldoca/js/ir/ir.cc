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

#include "maldoca/js/ir/ir.h"

// IWYU pragma: begin_keep

#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Region.h"

// IWYU pragma: end_keep

// =============================================================================
// Dialect Definition
// =============================================================================

#include "maldoca/js/ir/jshir_dialect.cc.inc"
#include "maldoca/js/ir/jsir_dialect.cc.inc"

// Dialect initialization, the instance will be owned by the context. This is
// the point of registration of types and operations for the dialect.
void maldoca::JsirDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "maldoca/js/ir/jsir_types.cc.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "maldoca/js/ir/jsir_ops.cc.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "maldoca/js/ir/jsir_attrs.cc.inc"
      >();
}

void maldoca::JshirDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "maldoca/js/ir/jshir_ops.cc.inc"
      >();
}

// Hook to materialize a single constant operation from a given attribute value
// with the desired resultant type. This method should use the provided builder
// to create the operation without changing the insertion position. The
// generated operation is expected to be constant-like. On success, this hook
// should return the value generated to represent the constant value.
// Otherwise, it should return nullptr on failure.
mlir::Operation *maldoca::JsirDialect::materializeConstant(
    mlir::OpBuilder &builder, mlir::Attribute value, mlir::Type type,
    mlir::Location loc) {
  return llvm::TypeSwitch<mlir::Attribute, mlir::Operation *>(value)
      .Case([&](mlir::BoolAttr value) {
        return builder.create<JsirBooleanLiteralOp>(loc, value);
      })
      .Case([&](mlir::FloatAttr value) {
        return builder.create<JsirNumericLiteralOp>(loc, value,
                                                    /*extra=*/nullptr);
      })
      .Case([&](mlir::StringAttr value) {
        return builder.create<JsirStringLiteralOp>(loc, value,
                                                   /*extra=*/nullptr);
      })
      .Case([&](JsirBigIntLiteralAttr value) {
        return builder.create<JsirBigIntLiteralOp>(loc, value.getValue(),
                                                   value.getExtra());
      })
      .Default([](mlir::Attribute value) { return nullptr; });
}

// =============================================================================
// Dialect Interface Definitions
// =============================================================================

#include "maldoca/js/ir/attr_interfaces.cc.inc"
#include "maldoca/js/ir/interfaces.cc.inc"

// =============================================================================
// Enum Definitions
// =============================================================================

#include "maldoca/js/ir/jsir_enum_attrs.cc.inc"

// =============================================================================
// Dialect Attribute Definitions
// =============================================================================

#define GET_ATTRDEF_CLASSES
#include "maldoca/js/ir/jsir_attrs.cc.inc"

// =============================================================================
// Dialect Type Definitions
// =============================================================================

#define GET_TYPEDEF_CLASSES
#include "maldoca/js/ir/jsir_types.cc.inc"

// =============================================================================
// Utils
// =============================================================================

namespace maldoca {

bool IsUnknownRegion(mlir::Region &region) {
  // Region must have exactly one block.
  return llvm::hasSingleElement(region);
}

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
  return llvm::isa<JsirExprRegionEndOp>(terminator);
}

bool IsExprsRegion(mlir::Region &region) {
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
  return llvm::isa<JsirExprsRegionEndOp>(terminator);
}

bool IsStmtRegion(mlir::Region &region) {
  // Region must have exactly one block.
  if (!llvm::hasSingleElement(region)) {
    return false;
  }

  mlir::Block &block = region.front();

  // Block must have at least one op (the statement).
  return !block.empty();
}

bool IsStmtsRegion(mlir::Region &region) {
  // Region must have exactly one block.
  return llvm::hasSingleElement(region);
}


}  // namespace maldoca
