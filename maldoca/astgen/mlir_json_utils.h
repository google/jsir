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

// Conversion between JSON values (nlohmann::json) and MLIR attributes.
//
// The IR stores the builtin JSON type (BuiltinTypeKind::kJson, e.g. Babel's
// `extra` object) as builtin MLIR attributes:
//
//   JSON            MLIR
//   ----            ----
//   null            mlir::UnitAttr
//   true / false    mlir::BoolAttr
//   integer         mlir::IntegerAttr (i64)
//   float           mlir::FloatAttr (f64)
//   string          mlir::StringAttr
//   array           mlir::ArrayAttr
//   object          mlir::DictionaryAttr
//
// The conversion is lossless, with these exceptions:
// - mlir::DictionaryAttr sorts its keys, so the key order of a JSON object is
//   not kept.
// - mlir::DictionaryAttr does not allow an empty key. An object member with an
//   empty key is dropped.
// - An unsigned integer larger than INT64_MAX becomes a float.

#ifndef MALDOCA_ASTGEN_MLIR_JSON_UTILS_H_
#define MALDOCA_ASTGEN_MLIR_JSON_UTILS_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "absl/status/statusor.h"
#include "nlohmann/json.hpp"

namespace maldoca {

// Converts any JSON value to an MLIR attribute (see the table above).
mlir::Attribute JsonToMlirAttr(mlir::MLIRContext* context,
                               const nlohmann::json& json);

// Converts a JSON object to an mlir::DictionaryAttr. Returns nullptr if `json`
// is not an object.
mlir::DictionaryAttr JsonToMlirDictionaryAttr(mlir::MLIRContext* context,
                                              const nlohmann::json& json);

// Converts an MLIR attribute back to a JSON value (see the table above).
// Returns an error for an attribute that is not in the table.
absl::StatusOr<nlohmann::json> MlirAttrToJson(mlir::Attribute attr);

}  // namespace maldoca

#endif  // MALDOCA_ASTGEN_MLIR_JSON_UTILS_H_
