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

#include "maldoca/astgen/mlir_json_utils.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "nlohmann/json.hpp"

namespace maldoca {

mlir::Attribute JsonToMlirAttr(mlir::MLIRContext* context,
                               const nlohmann::json& json) {
  mlir::Builder builder(context);
  switch (json.type()) {
    case nlohmann::json::value_t::null:
    case nlohmann::json::value_t::discarded:
      return builder.getUnitAttr();
    case nlohmann::json::value_t::boolean:
      return builder.getBoolAttr(json.get<bool>());
    case nlohmann::json::value_t::number_integer:
      return builder.getI64IntegerAttr(json.get<int64_t>());
    case nlohmann::json::value_t::number_unsigned: {
      uint64_t value = json.get<uint64_t>();
      if (value <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return builder.getI64IntegerAttr(static_cast<int64_t>(value));
      }
      return builder.getF64FloatAttr(static_cast<double>(value));
    }
    case nlohmann::json::value_t::number_float:
      return builder.getF64FloatAttr(json.get<double>());
    case nlohmann::json::value_t::string:
      return builder.getStringAttr(json.get_ref<const std::string&>());
    case nlohmann::json::value_t::binary:
      // Not produced by the JSON parser.
      return builder.getUnitAttr();
    case nlohmann::json::value_t::array: {
      std::vector<mlir::Attribute> elements;
      elements.reserve(json.size());
      for (const nlohmann::json& element : json) {
        elements.push_back(JsonToMlirAttr(context, element));
      }
      return builder.getArrayAttr(elements);
    }
    case nlohmann::json::value_t::object:
      return JsonToMlirDictionaryAttr(context, json);
  }
}

mlir::DictionaryAttr JsonToMlirDictionaryAttr(mlir::MLIRContext* context,
                                              const nlohmann::json& json) {
  if (!json.is_object()) {
    return nullptr;
  }
  mlir::Builder builder(context);
  std::vector<mlir::NamedAttribute> members;
  members.reserve(json.size());
  for (const auto& [key, value] : json.items()) {
    if (key.empty()) {
      continue;
    }
    members.push_back(
        builder.getNamedAttr(key, JsonToMlirAttr(context, value)));
  }
  return builder.getDictionaryAttr(members);
}

absl::StatusOr<nlohmann::json> MlirAttrToJson(mlir::Attribute attr) {
  if (attr == nullptr || llvm::isa<mlir::UnitAttr>(attr)) {
    return nullptr;
  }
  // BoolAttr is an IntegerAttr of type i1, so check it first.
  if (auto bool_attr = llvm::dyn_cast<mlir::BoolAttr>(attr)) {
    return bool_attr.getValue();
  }
  if (auto int_attr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
    // `getSExtValue()` would assert (or truncate) for integers wider than 64
    // bits, and would make large unsigned values negative.
    if (int_attr.getType().isUnsignedInteger()) {
      if (std::optional<uint64_t> value = int_attr.getValue().tryZExtValue()) {
        return *value;
      }
      return absl::InvalidArgumentError(
          "Unsigned integer attribute does not fit in 64 bits.");
    }
    if (std::optional<int64_t> value = int_attr.getValue().trySExtValue()) {
      return *value;
    }
    return absl::InvalidArgumentError(
        "Integer attribute does not fit in 64 bits.");
  }
  if (auto float_attr = llvm::dyn_cast<mlir::FloatAttr>(attr)) {
    return float_attr.getValueAsDouble();
  }
  if (auto string_attr = llvm::dyn_cast<mlir::StringAttr>(attr)) {
    return string_attr.str();
  }
  if (auto array_attr = llvm::dyn_cast<mlir::ArrayAttr>(attr)) {
    nlohmann::json array = nlohmann::json::array();
    for (mlir::Attribute element : array_attr) {
      ABSL_ASSIGN_OR_RETURN(nlohmann::json json_element,
                            MlirAttrToJson(element));
      array.push_back(std::move(json_element));
    }
    return array;
  }
  if (auto dict_attr = llvm::dyn_cast<mlir::DictionaryAttr>(attr)) {
    nlohmann::json object = nlohmann::json::object();
    for (mlir::NamedAttribute member : dict_attr) {
      ABSL_ASSIGN_OR_RETURN(nlohmann::json json_value,
                            MlirAttrToJson(member.getValue()));
      object[member.getName().str()] = std::move(json_value);
    }
    return object;
  }
  return absl::InvalidArgumentError(
      "MLIR attribute cannot be converted to JSON.");
}

}  // namespace maldoca
