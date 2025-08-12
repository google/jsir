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

#include "maldoca/astgen/symbol.h"

#include <iterator>
#include <string>

#include "absl/container/flat_hash_set.h"

namespace maldoca {

Symbol::Symbol(absl::string_view str) {
  // Transforms the string to lower case words.
  bool should_create_new_word = true;
  for (char ch : str) {
    if (absl::ascii_isupper(ch)) {
      words_.push_back("");
      words_.back().push_back(absl::ascii_tolower(ch));
      should_create_new_word = false;

    } else if (ch == '_') {
      should_create_new_word = true;

    } else {
      if (should_create_new_word) {
        words_.push_back("");
      }
      words_.back().push_back(ch);
      should_create_new_word = false;
    }
  }

  // An unfortunate patchwork: If the input ends with '_', this '_' is included
  // in the last word.
  //
  // Example: Let's say we want to define a field with the name "operator", we
  // have to define the MLIR field in the TableGen file as "operator_". Then,
  // the MLIR getter would become "getOperator_()" or "getOperator_Attr()".
  //
  // You might ask: given that the MLIR getter is already prefixed with "get",
  // why do we still need the "_"? Well, MLIR still generates the builder
  // argument name unchanged.
  //
  // Now, we need a way to turn "operator" into "getOperator_Attr".
  // This is done by:
  //   (Symbol("get") + Symbol("operator").ToCcVarName() + "attr").ToCamelCase()
  if (!str.empty() && str.back() == '_') {
    words_.back().push_back('_');
  }
}

Symbol &Symbol::operator+=(const Symbol &other) {
  words_.insert(words_.end(), other.words_.begin(), other.words_.end());
  return *this;
}

Symbol &Symbol::operator+=(Symbol &&other) {
  words_.insert(words_.end(), std::make_move_iterator(other.words_.begin()),
                std::make_move_iterator(other.words_.end()));
  return *this;
}

Symbol &Symbol::operator+=(absl::string_view other) {
  return operator+=(Symbol(other));
}

std::string Symbol::ToSnakeCase() const { return absl::StrJoin(words_, "_"); }

bool Symbol::IsReservedKeyword() const {
  // https://en.cppreference.com/w/cpp/keyword
  static const auto *kReservedKeywords = new absl::flat_hash_set<std::string>{
      "alignas",
      "alignof",
      "and",
      "and_eq",
      "asm",
      "atomic_cancel",
      "atomic_commit",
      "atomic_noexcept",
      "auto",
      "bitand",
      "bitor",
      "bool",
      "break",
      "case",
      "catch",
      "char",
      "char8_t",
      "char16_t",
      "char32_t",
      "class",
      "compl",
      "concept",
      "const",
      "consteval",
      "constexpr",
      "constinit",
      "const_cast",
      "continue",
      "co_await",
      "co_return",
      "co_yield",
      "decltype",
      "default",
      "delete",
      "do",
      "double",
      "dynamic_cast",
      "else",
      "enum",
      "explicit",
      "export",
      "extern",
      "false",
      "float",
      "for",
      "friend",
      "goto",
      "if",
      "inline",
      "int",
      "long",
      "mutable",
      "namespace",
      "new",
      "noexcept",
      "not",
      "not_eq",
      "nullptr",
      "operator",
      "or",
      "or_eq",
      "private",
      "protected",
      "public",
      "reflexpr",
      "register",
      "reinterpret_cast",
      "requires",
      "return",
      "short",
      "signed",
      "sizeof",
      "static",
      "static_assert",
      "static_cast",
      "struct",
      "switch",
      "synchronized",
      "template",
      "this",
      "thread_local",
      "throw",
      "true",
      "try",
      "typedef",
      "typeid",
      "typename",
      "union",
      "unsigned",
      "using",
      "virtual",
      "void",
      "volatile",
      "wchar_t",
      "while",
      "xor",
      "xor_eq",

      // Since https://reviews.llvm.org/D141742, "properties" cannot be an
      // argument name in an MLIR op.
      "properties",
  };

  std::string snake_case = ToSnakeCase();
  return kReservedKeywords->contains(snake_case);
}

std::string Symbol::ToCcVarName() const {
  std::string result = ToSnakeCase();
  if (IsReservedKeyword()) {
    result.push_back('_');
  }
  return result;
}

std::string Symbol::ToMlirGetter() const {
  std::string result = absl::StrCat("get", ToPascalCase());
  if (IsReservedKeyword()) {
    result.push_back('_');
  }
  return result;
}

std::string Symbol::ToPascalCase() const {
  std::string result;
  for (const auto &word : words_) {
    result.push_back(absl::ascii_toupper(word[0]));
    result.append(word.substr(1));
  }
  return result;
}

std::string Symbol::ToCamelCase() const {
  std::string result;
  for (const auto &word : words_) {
    if (result.empty()) {
      result = word;
    } else {
      result.push_back(absl::ascii_toupper(word[0]));
      result.append(word.substr(1));
    }
  }
  return result;
}

}  // namespace maldoca
