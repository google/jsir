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
"""Port of maldoca/astgen/symbol_test.cc to Python."""

import unittest

from maldoca.astgen.symbol import Symbol


class SymbolTest(unittest.TestCase):

  def test_from_pascal_case(self):
    symbol = Symbol("GetLeftHandSide")
    self.assertEqual(symbol.to_pascal_case(), "GetLeftHandSide")
    self.assertEqual(symbol.to_camel_case(), "getLeftHandSide")
    self.assertEqual(symbol.to_snake_case(), "get_left_hand_side")

  def test_from_camel_case(self):
    symbol = Symbol("getLeftHandSide")
    self.assertEqual(symbol.to_pascal_case(), "GetLeftHandSide")
    self.assertEqual(symbol.to_camel_case(), "getLeftHandSide")
    self.assertEqual(symbol.to_snake_case(), "get_left_hand_side")

  def test_from_snake_case(self):
    symbol = Symbol("get_left_hand_side")
    self.assertEqual(symbol.to_pascal_case(), "GetLeftHandSide")
    self.assertEqual(symbol.to_camel_case(), "getLeftHandSide")
    self.assertEqual(symbol.to_snake_case(), "get_left_hand_side")

  def test_extra_underscores_are_ignored(self):
    symbol = Symbol("_get_left_hand_side")
    self.assertEqual(symbol.to_pascal_case(), "GetLeftHandSide")
    self.assertEqual(symbol.to_camel_case(), "getLeftHandSide")
    self.assertEqual(symbol.to_snake_case(), "get_left_hand_side")

    symbol = Symbol("get_left_hand_side_")
    self.assertEqual(symbol.to_pascal_case(), "GetLeftHandSide_")
    self.assertEqual(symbol.to_camel_case(), "getLeftHandSide_")
    self.assertEqual(symbol.to_snake_case(), "get_left_hand_side_")

    symbol = Symbol("get__left_hand_side")
    self.assertEqual(symbol.to_pascal_case(), "GetLeftHandSide")
    self.assertEqual(symbol.to_camel_case(), "getLeftHandSide")
    self.assertEqual(symbol.to_snake_case(), "get_left_hand_side")

  def test_concatenate_symbols(self):
    first = Symbol("get_left")
    second = Symbol("HandSide")
    symbol = first + second
    self.assertEqual(symbol.to_pascal_case(), "GetLeftHandSide")
    self.assertEqual(symbol.to_camel_case(), "getLeftHandSide")
    self.assertEqual(symbol.to_snake_case(), "get_left_hand_side")

  def test_concatenate_symbol_with_string(self):
    first = Symbol("get_left")
    second = "HandSide"
    symbol = first + second
    self.assertEqual(symbol.to_pascal_case(), "GetLeftHandSide")
    self.assertEqual(symbol.to_camel_case(), "getLeftHandSide")
    self.assertEqual(symbol.to_snake_case(), "get_left_hand_side")

  def test_avoid_cpp_keyword(self):
    symbol = Symbol("operator")
    self.assertEqual(symbol.to_cc_var_name(), "operator_")
    self.assertEqual(
        Symbol(symbol.to_cc_var_name()).to_pascal_case(), "Operator_"
    )
    self.assertEqual(
        (Symbol("get") + symbol.to_cc_var_name() + "attr").to_camel_case(),
        "getOperator_Attr",
    )
    self.assertEqual(
        (Symbol("get") + symbol.to_cc_var_name() + "attr").to_snake_case(),
        "get_operator__attr",
    )


if __name__ == "__main__":
  unittest.main()
