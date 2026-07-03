# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Port of maldoca/astgen/symbol.{h,cc} to Python."""

from __future__ import annotations

# https://en.cppreference.com/w/cpp/keyword
#
# This is reused verbatim from symbol.cc even though the generated code is
# now Python: the reserved words are still avoided because generated
# identifiers are emitted into both C++ and MLIR TableGen output.
_RESERVED_KEYWORDS = frozenset({
    "alignas", "alignof", "and", "and_eq", "asm", "atomic_cancel",
    "atomic_commit", "atomic_noexcept", "auto", "bitand", "bitor", "bool",
    "break", "case", "catch", "char", "char8_t", "char16_t", "char32_t",
    "class", "compl", "concept", "const", "consteval", "constexpr",
    "constinit", "const_cast", "continue", "co_await", "co_return",
    "co_yield", "decltype", "default", "delete", "do", "double",
    "dynamic_cast", "else", "enum", "explicit", "export", "extern", "false",
    "float", "for", "friend", "goto", "if", "inline", "int", "long",
    "mutable", "namespace", "new", "noexcept", "not", "not_eq", "nullptr",
    "operator", "or", "or_eq", "private", "protected", "public", "reflexpr",
    "register", "reinterpret_cast", "requires", "return", "short", "signed",
    "sizeof", "static", "static_assert", "static_cast", "struct", "switch",
    "synchronized", "template", "this", "thread_local", "throw", "true",
    "try", "typedef", "typeid", "typename", "union", "unsigned", "using",
    "virtual", "void", "volatile", "wchar_t", "while", "xor", "xor_eq",
    # Since https://reviews.llvm.org/D141742, "properties" cannot be an
    # argument name in an MLIR op.
    "properties",
})


class Symbol:
  """Models a list of words, and supports printing in snake_case, PascalCase,

  and camelCase.

  Also supports concatenation.

  For example, if a field is named "sourceType", then:
  - C++ variable name: source_type (printing snake_case)
  - Protobuf field name: source_type (printing snake_case)
  - C++ setter function name: get_source_type (concatenation)
  - JavaScript field name: sourceType (printing camelCase)
  - JSPB getter/setter: {get,set}SourceType (concatenation, printing
    camelCase)
  """

  # Always store in lower case.
  _words: list[str]

  def __init__(self, s: str = ""):
    # Input can be either snake_case, PascalCase, or camelCase.
    words: list[str] = []
    should_create_new_word = True
    for ch in s:
      if ch.isascii() and ch.isupper():
        words.append(ch.lower())
        should_create_new_word = False
      elif ch == "_":
        should_create_new_word = True
      else:
        if should_create_new_word:
          words.append("")
        words[-1] += ch
        should_create_new_word = False

    # An unfortunate patchwork: If the input ends with '_', this '_' is
    # included in the last word.
    #
    # Example: Let's say we want to define a field with the name "operator",
    # we have to define the MLIR field in the TableGen file as "operator_".
    # Then, the MLIR getter would become "getOperator_()" or
    # "getOperator_Attr()".
    #
    # You might ask: given that the MLIR getter is already prefixed with
    # "get", why do we still need the "_"? Well, MLIR still generates the
    # builder argument name unchanged.
    #
    # Now, we need a way to turn "operator" into "getOperator_Attr".
    # This is done by:
    #   Symbol("get") + Symbol("operator").to_cc_var_name() + "attr"
    #   ).to_camel_case()
    if s and s[-1] == "_":
      words[-1] += "_"

    self._words = words

  @classmethod
  def _from_words(cls, words: list[str]) -> "Symbol":
    symbol = cls.__new__(cls)
    symbol._words = words
    return symbol

  # Concatenation.
  # E.g. "one_two" + "three_four" => "one_two_three_four"
  def __add__(self, other: "Symbol | str") -> "Symbol":
    if isinstance(other, str):
      other = Symbol(other)
    return Symbol._from_words(self._words + other._words)

  def __iadd__(self, other: "Symbol | str") -> "Symbol":
    if isinstance(other, str):
      other = Symbol(other)
    self._words += other._words
    return self

  # "snake_case"
  def to_snake_case(self) -> str:
    return "_".join(self._words)

  # Same as snake_case, but adds a '_' if collides with a reserved keyword.
  #
  # E.g. Symbol("static").to_cc_var_name() => "static_"
  def to_cc_var_name(self) -> str:
    result = self.to_snake_case()
    if self._is_reserved_keyword():
      result += "_"
    return result

  # "PascalCase"
  def to_pascal_case(self) -> str:
    return "".join(word[0].upper() + word[1:] for word in self._words)

  # "getPascalCase", but adds a '_' if the field name collides with a
  # reserved keyword.
  #
  # E.g. Symbol("static").to_mlir_getter() => "getStatic_"
  def to_mlir_getter(self) -> str:
    result = "get" + self.to_pascal_case()
    if self._is_reserved_keyword():
      result += "_"
    return result

  # "camelCase"
  def to_camel_case(self) -> str:
    result = ""
    for word in self._words:
      if not result:
        result = word
      else:
        result += word[0].upper() + word[1:]
    return result

  def _is_reserved_keyword(self) -> bool:
    return self.to_snake_case() in _RESERVED_KEYWORDS

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, Symbol):
      return NotImplemented
    return self._words == other._words

  def __repr__(self) -> str:
    return f"Symbol({self.to_snake_case()!r})"
