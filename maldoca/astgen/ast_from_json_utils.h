// Copyright 2025 Google LLC
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

#ifndef MALDOCA_ASTGEN_AST_FROM_JSON_UTILS_H_
#define MALDOCA_ASTGEN_AST_FROM_JSON_UTILS_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "nlohmann/json.hpp"
#include "maldoca/base/status_macros.h"

namespace maldoca {

template <typename T>
using Converter = std::function<absl::StatusOr<T>(const nlohmann::json&)>;

template <typename T>
absl::StatusOr<T> GetRequiredField(const nlohmann::json& json,
                                   absl::string_view field_name,
                                   Converter<T> converter) {
  auto field_it = json.find(field_name);
  if (field_it == json.end()) {
    return absl::InvalidArgumentError(
        absl::StrCat("`", field_name, "` is undefined."));
  }
  const nlohmann::json& json_field = field_it.value();
  if (json_field.is_null()) {
    return absl::InvalidArgumentError(
        absl::StrCat("`", field_name, "` is null."));
  }
  return converter(json_field);
}

template <typename T>
absl::StatusOr<std::optional<T>> GetOptionalField(const nlohmann::json& json,
                                                  absl::string_view field_name,
                                                  Converter<T> converter) {
  auto field_it = json.find(field_name);
  if (field_it == json.end()) {
    return std::nullopt;
  }
  const nlohmann::json& json_field = field_it.value();
  if (json_field.is_null()) {
    return absl::InvalidArgumentError(
        absl::StrCat("`", field_name, "` is null."));
  }
  return converter(json_field);
}

template <typename T>
absl::StatusOr<std::optional<T>> GetNullableField(const nlohmann::json& json,
                                                  absl::string_view field_name,
                                                  Converter<T> converter) {
  auto field_it = json.find(field_name);
  if (field_it == json.end()) {
    return absl::InvalidArgumentError(
        absl::StrCat("`", field_name, "` is undefined."));
  }
  const nlohmann::json& json_field = field_it.value();
  if (json_field.is_null()) {
    return std::nullopt;
  }
  return converter(json_field);
}

inline bool IsString(const nlohmann::json& json) { return json.is_string(); }

inline absl::StatusOr<std::string> JsonToString(const nlohmann::json& json) {
  if (json.is_null()) {
    return absl::InvalidArgumentError("json is null.");
  }
  if (!json.is_string()) {
    return absl::InvalidArgumentError("json is not a string.");
  }
  return json.get<std::string>();
}

inline bool IsBool(const nlohmann::json& json) { return json.is_boolean(); }

inline absl::StatusOr<bool> JsonToBool(const nlohmann::json& json) {
  if (json.is_null()) {
    return absl::InvalidArgumentError("json is null.");
  }
  if (!json.is_boolean()) {
    return absl::InvalidArgumentError("json is not a boolean.");
  }
  return json.get<bool>();
}

inline bool IsInt64(const nlohmann::json& json) {
  return json.is_number_integer();
}

inline absl::StatusOr<int64_t> JsonToInt64(const nlohmann::json& json) {
  if (json.is_null()) {
    return absl::InvalidArgumentError("json is null.");
  }
  if (!json.is_number_integer()) {
    return absl::InvalidArgumentError("json is not an integer.");
  }
  return json.get<int64_t>();
}

inline bool IsDouble(const nlohmann::json& json) { return json.is_number(); }

inline absl::StatusOr<double> JsonToDouble(const nlohmann::json& json) {
  if (json.is_null()) {
    return absl::InvalidArgumentError("json is null.");
  }
  if (!json.is_number()) {
    return absl::InvalidArgumentError("json is not a number.");
  }
  return json.get<double>();
}

template <typename T>
inline Converter<T> Enum(
    std::function<absl::StatusOr<T>(absl::string_view)> string_to_enum) {
  return [=](const nlohmann::json& json) -> absl::StatusOr<T> {
    if (json.is_null()) {
      return absl::InvalidArgumentError("json is null.");
    }
    if (!json.is_string()) {
      return absl::InvalidArgumentError("json is not a string.");
    }
    return string_to_enum(json.get<std::string>());
  };
}

template <typename T>
Converter<std::optional<T>> Nullable(Converter<T> converter) {
  return [=](const nlohmann::json& json) -> absl::StatusOr<std::optional<T>> {
    if (json.is_null()) {
      return std::nullopt;
    }
    return converter(json);
  };
}

template <typename T>
Converter<std::vector<T>> List(Converter<T> elem_converter) {
  return [=](const nlohmann::json& json) -> absl::StatusOr<std::vector<T>> {
    if (json.is_null()) {
      return absl::InvalidArgumentError("json is null.");
    }
    if (!json.is_array()) {
      return absl::InvalidArgumentError("json is not an array.");
    }
    std::vector<T> result;
    for (const nlohmann::json& element : json) {
      MALDOCA_ASSIGN_OR_RETURN(T value, elem_converter(element));
      result.push_back(std::move(value));
    }
    return result;
  };
}

template <typename T>
struct VariantOption {
  std::function<bool(const nlohmann::json&)> predicate;
  Converter<T> converter;
};

// JsonToVariant:
//   Takes VariantOption<T1> option1, ..., VariantOption<Tk> optionk;
//   Returns absl::StatusOr<std::variant<T1, T2, ..., Tk>>.
//
//   If option1.predicate(json), then return option1.converter(json);
//   ...
//   If optionk.predicate(json), then return optionk.converter(json).
//   Otherwise, returns absl::InvalidArgumentError("Invalid type: ").

namespace internal {
template <typename Variant>
absl::StatusOr<Variant> JsonToVariantRecursive(const nlohmann::json& json) {
  return absl::InvalidArgumentError("Invalid type: ");
}

template <typename Variant, typename T, typename... Rest>
absl::StatusOr<Variant> JsonToVariantRecursive(const nlohmann::json& json,
                                               VariantOption<T> option,
                                               VariantOption<Rest>... rest) {
  if (option.predicate(json)) {
    MALDOCA_ASSIGN_OR_RETURN(T value, option.converter(json));
    return Variant(std::move(value));
  }
  return JsonToVariantRecursive<Variant>(json, std::move(rest)...);
}
}  // namespace internal

template <typename... Ts>
Converter<std::variant<Ts...>> Variant(VariantOption<Ts>... options) {
  return
      [=](const nlohmann::json& json) -> absl::StatusOr<std::variant<Ts...>> {
        return internal::JsonToVariantRecursive<std::variant<Ts...>>(
            json, std::move(options)...);
      };
}

inline absl::StatusOr<std::string> GetType(const nlohmann::json& json) {
  auto type_it = json.find("type");
  if (type_it == json.end()) {
    return absl::InvalidArgumentError("`type` is undefined.");
  }
  const nlohmann::json& json_type = type_it.value();
  if (json_type.is_null()) {
    return absl::InvalidArgumentError("json_type is null.");
  }
  if (!json_type.is_string()) {
    return absl::InvalidArgumentError("`json_type` expected to be string.");
  }
  return json_type.get<std::string>();
}

}  // namespace maldoca

#endif  // MALDOCA_ASTGEN_AST_FROM_JSON_UTILS_H_
