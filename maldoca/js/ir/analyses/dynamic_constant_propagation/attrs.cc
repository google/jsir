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

#include "maldoca/js/ir/analyses/dynamic_constant_propagation/attrs.h"

#include "llvm/ADT/TypeSwitch.h"  // NOLINT(misc-include-cleaner)
#include "mlir/IR/DialectImplementation.h"  // NOLINT(misc-include-cleaner)

#include "maldoca/js/ir/analyses/dynamic_constant_propagation/jsir_builtin_dialect.cc.inc"

#define GET_ATTRDEF_CLASSES
#include "maldoca/js/ir/analyses/dynamic_constant_propagation/attrs.cc.inc"

void maldoca::JsirBuiltinDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "maldoca/js/ir/analyses/dynamic_constant_propagation/attrs.cc.inc"
      >();
}
