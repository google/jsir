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

#ifndef MALDOCA_ASTGEN_SYMBOL_H_
#define MALDOCA_ASTGEN_SYMBOL_H_

#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"

namespace maldoca {

// Models a list of words, and supports printing in snake_case, PascalCase, and
// camelCase.
//
// Also supports concatenation.
//
// For example, if a field is named "sourceType", then:
// - C++ variable name: source_type (printing snake_case)
// - Protobuf field name: source_type (printing snake_case)
// - C++ setter function name: get_source_type (concatenation)
// - JavaScript field name: sourceType (printing camelCase)
// - JSPB getter/setter: {get,set}SourceType (concatenation, printing camelCase)
class Symbol {
 public:
  // Input can be either snake_case, PascalCase, or camelCase.
  explicit Symbol(absl::string_view str = "");

  // Concatenation.
  // E.g. "one_two" + "three_four" => "one_two_three_four"
  Symbol& operator+=(const Symbol& other);
  Symbol& operator+=(Symbol&& other);

  Symbol& operator+=(absl::string_view other);

  template <typename T>
  Symbol operator+(T&& other) const {
    Symbol words = *this;
    words += other;
    return words;
  }

  // "snake_case"
  std::string ToSnakeCase() const;

  // Same as snake_case, but adds a '_' if collides with a reserved keyword.
  //
  // E.g. Symbol("static").ToCcVarName() => "static_"
  std::string ToCcVarName() const;

  // "PascalCase"
  std::string ToPascalCase() const;

  // "getPascalCase", but adds a '_' if the field name collides with a reserved
  // keyword.
  //
  // E.g. Symbol("static").ToMlirGetter() => "getStatic_"
  std::string ToMlirGetter() const;

  // "camelCase"
  std::string ToCamelCase() const;

 private:
  bool IsReservedKeyword() const;

  // Always store in lower case.
  std::vector<std::string> words_;
};

}  // namespace maldoca

#endif  // MALDOCA_ASTGEN_SYMBOL_H_
