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

#ifndef MALDOCA_JS_IR_ANALYSES_DYNAMIC_CONSTANT_PROPAGATION_ATTRS_H_
#define MALDOCA_JS_IR_ANALYSES_DYNAMIC_CONSTANT_PROPAGATION_ATTRS_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#include "maldoca/js/ir/analyses/dynamic_constant_propagation/jsir_builtin_dialect.h.inc"
#include "maldoca/js/ir/ir.h"

#define GET_ATTRDEF_CLASSES
#include "maldoca/js/ir/analyses/dynamic_constant_propagation/attrs.h.inc"

#endif  // MALDOCA_JS_IR_ANALYSES_DYNAMIC_CONSTANT_PROPAGATION_ATTRS_H_
