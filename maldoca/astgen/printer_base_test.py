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
"""Tests for maldoca/astgen/printer_base.py.

There's no C++ printer_base_test.cc to port from (the astgen C++ code
relies directly on google::protobuf::io::Printer, which has its own
upstream tests) -- these tests cover the from-scratch `Printer`
reimplementation directly, since every downstream printer depends on its
indentation and `$var$` substitution being exactly right.
"""

import unittest

from maldoca.astgen.printer_base import Printer


class PrinterTest(unittest.TestCase):

  def test_println_adds_newline(self):
    p = Printer()
    p.println("abc")
    p.println("def")
    self.assertEqual(p.content(), "abc\ndef\n")

  def test_blank_println_has_no_trailing_whitespace(self):
    p = Printer()
    p.println("abc")
    p.println()
    p.println("def")
    self.assertEqual(p.content(), "abc\n\ndef\n")

  def test_variable_substitution(self):
    p = Printer()
    p.println("Hello, $name$!", name="World")
    self.assertEqual(p.content(), "Hello, World!\n")

  def test_double_dollar_is_literal_dollar(self):
    p = Printer()
    p.println("cost: $$5")
    self.assertEqual(p.content(), "cost: $5\n")

  def test_unknown_variable_raises(self):
    p = Printer()
    with self.assertRaises(KeyError):
      p.println("$missing$")

  def test_indent_applies_to_each_line(self):
    p = Printer()
    p.println("outer")
    with p.with_indent():
      p.println("inner1")
      p.println("inner2")
    p.println("outer again")
    self.assertEqual(
        p.content(), "outer\n  inner1\n  inner2\nouter again\n"
    )

  def test_indent_applies_within_a_single_multiline_print(self):
    p = Printer()
    with p.with_indent():
      p.println("line1\nline2")
    self.assertEqual(p.content(), "  line1\n  line2\n")

  def test_blank_line_inside_indent_has_no_trailing_whitespace(self):
    p = Printer()
    with p.with_indent():
      p.println("a")
      p.println()
      p.println("b")
    self.assertEqual(p.content(), "  a\n\n  b\n")

  def test_nested_indent(self):
    p = Printer()
    p.println("a")
    with p.with_indent():
      p.println("b")
      with p.with_indent():
        p.println("c")
      p.println("d")
    p.println("e")
    self.assertEqual(p.content(), "a\n  b\n    c\n  d\ne\n")

  def test_with_vars_scopes_variable_lookup(self):
    p = Printer()
    with p.with_vars({"name": "World"}):
      p.println("Hello, $name$!")
    p.println("Bye, $$")
    self.assertEqual(p.content(), "Hello, World!\nBye, $\n")

  def test_with_vars_nesting_inner_overrides_outer(self):
    p = Printer()
    with p.with_vars({"name": "outer"}):
      p.println("1: $name$")
      with p.with_vars({"name": "inner"}):
        p.println("2: $name$")
      p.println("3: $name$")
    self.assertEqual(p.content(), "1: outer\n2: inner\n3: outer\n")

  def test_inline_variable_overrides_with_vars(self):
    p = Printer()
    with p.with_vars({"name": "outer"}):
      p.println("$name$", name="inline")
    self.assertEqual(p.content(), "inline\n")


if __name__ == "__main__":
  unittest.main()
