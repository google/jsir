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
"""Port of maldoca/astgen/type_test.cc to Python."""

import dataclasses
import unittest

from google.protobuf import text_format

from maldoca.astgen import ast_def_pb2
from maldoca.astgen import type_pb2
from maldoca.astgen.type import BuiltinType
from maldoca.astgen.type import ClassType
from maldoca.astgen.type import from_type_pb
from maldoca.astgen.type import ListType
from maldoca.astgen.type import NonListType
from maldoca.astgen.type import ScalarType
from maldoca.astgen.type import Type
from maldoca.astgen.type import VariantType


@dataclasses.dataclass
class TypeTestCase:
  type_pb: str
  js_type: str
  cc_type: str
  cc_getter_type: str
  cc_const_getter_type: str
  cc_lang_name: str = ""
  td_types: dict = dataclasses.field(default_factory=dict)
  cc_mlir_builder_type: dict = dataclasses.field(default_factory=dict)
  cc_mlir_getter_type: dict = dataclasses.field(default_factory=dict)


def _test_type_pb_to_type_and_print(test: unittest.TestCase, case: TypeTestCase):
  pb = type_pb2.TypePb()
  text_format.Parse(case.type_pb, pb)

  type_ = from_type_pb(pb, case.cc_lang_name)

  test.assertEqual(type_.js_type(), case.js_type)
  test.assertEqual(type_.cc_type(), case.cc_type)
  test.assertEqual(type_.cc_mutable_getter_type(), case.cc_getter_type)
  test.assertEqual(type_.cc_const_getter_type(), case.cc_const_getter_type)

  for field_kind, td_type in case.td_types.items():
    test.assertEqual(type_.td_type(field_kind), td_type)

  for field_kind, cc_mlir_builder_type in case.cc_mlir_builder_type.items():
    test.assertEqual(type_.cc_mlir_builder_type(field_kind), cc_mlir_builder_type)

  for field_kind, cc_mlir_getter_type in case.cc_mlir_getter_type.items():
    test.assertEqual(type_.cc_mlir_getter_type(field_kind), cc_mlir_getter_type)


class TypeTest(unittest.TestCase):

  def test_convert_builtin_type(self):
    _test_type_pb_to_type_and_print(
        self,
        TypeTestCase(
            type_pb="bool {}",
            js_type="boolean",
            cc_type="bool",
            cc_getter_type="bool",
            cc_const_getter_type="bool",
            td_types={ast_def_pb2.FIELD_KIND_ATTR: "BoolAttr"},
            cc_mlir_builder_type={
                ast_def_pb2.FIELD_KIND_ATTR: "mlir::BoolAttr"
            },
            cc_mlir_getter_type={
                ast_def_pb2.FIELD_KIND_ATTR: "mlir::BoolAttr"
            },
        ),
    )

    _test_type_pb_to_type_and_print(
        self,
        TypeTestCase(
            type_pb="int64 {}",
            js_type="/*int64*/number",
            cc_type="int64_t",
            cc_getter_type="int64_t",
            cc_const_getter_type="int64_t",
            td_types={ast_def_pb2.FIELD_KIND_ATTR: "I64Attr"},
            cc_mlir_builder_type={
                ast_def_pb2.FIELD_KIND_ATTR: "mlir::IntegerAttr"
            },
            cc_mlir_getter_type={
                ast_def_pb2.FIELD_KIND_ATTR: "mlir::IntegerAttr"
            },
        ),
    )

    _test_type_pb_to_type_and_print(
        self,
        TypeTestCase(
            type_pb="double {}",
            js_type="/*double*/number",
            cc_type="double",
            cc_getter_type="double",
            cc_const_getter_type="double",
            td_types={ast_def_pb2.FIELD_KIND_ATTR: "F64Attr"},
            cc_mlir_builder_type={
                ast_def_pb2.FIELD_KIND_ATTR: "mlir::FloatAttr"
            },
            cc_mlir_getter_type={
                ast_def_pb2.FIELD_KIND_ATTR: "mlir::FloatAttr"
            },
        ),
    )

    _test_type_pb_to_type_and_print(
        self,
        TypeTestCase(
            type_pb="string {}",
            js_type="string",
            cc_type="std::string",
            cc_getter_type="absl::string_view",
            cc_const_getter_type="absl::string_view",
            td_types={ast_def_pb2.FIELD_KIND_ATTR: "StrAttr"},
            cc_mlir_builder_type={
                ast_def_pb2.FIELD_KIND_ATTR: "mlir::StringAttr"
            },
            cc_mlir_getter_type={
                ast_def_pb2.FIELD_KIND_ATTR: "mlir::StringAttr"
            },
        ),
    )

  def test_convert_enum_type(self):
    _test_type_pb_to_type_and_print(
        self,
        TypeTestCase(
            type_pb='enum: "BinaryOperator"',
            js_type="BinaryOperator",
            cc_type="TestLangNameBinaryOperator",
            cc_getter_type="TestLangNameBinaryOperator",
            cc_const_getter_type="TestLangNameBinaryOperator",
            cc_lang_name="TestLangName",
            td_types={ast_def_pb2.FIELD_KIND_ATTR: "StrAttr"},
            cc_mlir_builder_type={
                ast_def_pb2.FIELD_KIND_ATTR: "mlir::StringAttr"
            },
            cc_mlir_getter_type={
                ast_def_pb2.FIELD_KIND_ATTR: "mlir::StringAttr"
            },
        ),
    )

  def test_convert_class_type(self):
    _test_type_pb_to_type_and_print(
        self,
        TypeTestCase(
            type_pb='class: "BinaryExpression"',
            js_type="BinaryExpression",
            cc_type="std::unique_ptr<TestLangNameBinaryExpression>",
            cc_getter_type="TestLangNameBinaryExpression*",
            cc_const_getter_type="const TestLangNameBinaryExpression*",
            cc_lang_name="TestLangName",
            td_types={ast_def_pb2.FIELD_KIND_RVAL: "AnyType"},
            cc_mlir_builder_type={ast_def_pb2.FIELD_KIND_RVAL: "mlir::Value"},
            cc_mlir_getter_type={ast_def_pb2.FIELD_KIND_RVAL: "mlir::Value"},
        ),
    )

  def test_convert_variant_type(self):
    _test_type_pb_to_type_and_print(
        self,
        TypeTestCase(
            type_pb="""
              variant {
                types { bool {} }
                types { string {} }
              }
            """,
            js_type="boolean | string",
            cc_type="std::variant<bool, std::string>",
            cc_getter_type="std::variant<bool, absl::string_view>",
            cc_const_getter_type="std::variant<bool, absl::string_view>",
            td_types={
                ast_def_pb2.FIELD_KIND_ATTR: "AnyAttrOf<[BoolAttr, StrAttr]>"
            },
            cc_mlir_builder_type={
                ast_def_pb2.FIELD_KIND_ATTR: "mlir::Attribute"
            },
            cc_mlir_getter_type={
                ast_def_pb2.FIELD_KIND_ATTR: "mlir::Attribute"
            },
        ),
    )

    _test_type_pb_to_type_and_print(
        self,
        TypeTestCase(
            type_pb="""
              variant {
                types { class: "Expression" }
                types { class: "Pattern" }
              }
            """,
            js_type="Expression | Pattern",
            cc_type=(
                "std::variant<std::unique_ptr<TestLangNameExpression>, "
                "std::unique_ptr<TestLangNamePattern>>"
            ),
            cc_getter_type=(
                "std::variant<TestLangNameExpression*, TestLangNamePattern*>"
            ),
            cc_const_getter_type=(
                "std::variant<const TestLangNameExpression*, "
                "const TestLangNamePattern*>"
            ),
            cc_lang_name="TestLangName",
            td_types={ast_def_pb2.FIELD_KIND_RVAL: "AnyType"},
            cc_mlir_builder_type={ast_def_pb2.FIELD_KIND_RVAL: "mlir::Value"},
            cc_mlir_getter_type={ast_def_pb2.FIELD_KIND_RVAL: "mlir::Value"},
        ),
    )

  def test_convert_list_type(self):
    _test_type_pb_to_type_and_print(
        self,
        TypeTestCase(
            type_pb="""
              list { element_type { class: "Expression" } }
            """,
            js_type="[ Expression ]",
            cc_type="std::vector<std::unique_ptr<TestLangNameExpression>>",
            cc_getter_type=(
                "std::vector<std::unique_ptr<TestLangNameExpression>>*"
            ),
            cc_const_getter_type=(
                "const std::vector<std::unique_ptr<TestLangNameExpression>>*"
            ),
            cc_lang_name="TestLangName",
            td_types={ast_def_pb2.FIELD_KIND_RVAL: "Variadic<AnyType>"},
            cc_mlir_builder_type={
                ast_def_pb2.FIELD_KIND_RVAL: "std::vector<mlir::Value>"
            },
            cc_mlir_getter_type={
                ast_def_pb2.FIELD_KIND_RVAL: "mlir::OperandRange"
            },
        ),
    )

    _test_type_pb_to_type_and_print(
        self,
        TypeTestCase(
            type_pb="""
              list {
                element_type { class: "Expression" }
                element_maybe_null: true
              }
            """,
            js_type="[ Expression | null ]",
            cc_type=(
                "std::vector<std::optional<std::unique_ptr<"
                "TestLangNameExpression>>>"
            ),
            cc_getter_type=(
                "std::vector<std::optional<std::unique_ptr<"
                "TestLangNameExpression>>>*"
            ),
            cc_const_getter_type=(
                "const std::vector<std::optional<std::unique_ptr<"
                "TestLangNameExpression>>>*"
            ),
            cc_lang_name="TestLangName",
            td_types={ast_def_pb2.FIELD_KIND_RVAL: "Variadic<AnyType>"},
            cc_mlir_builder_type={
                ast_def_pb2.FIELD_KIND_RVAL: "std::vector<mlir::Value>"
            },
            cc_mlir_getter_type={
                ast_def_pb2.FIELD_KIND_RVAL: "mlir::OperandRange"
            },
        ),
    )

    _test_type_pb_to_type_and_print(
        self,
        TypeTestCase(
            type_pb="""
              list {
                element_type {
                  variant {
                    types { class: "Expression" }
                    types { class: "Pattern" }
                  }
                }
                element_maybe_null: true
              }
            """,
            js_type="[ Expression | Pattern | null ]",
            cc_type=(
                "std::vector<std::optional<std::variant<std::unique_ptr<"
                "TestLangNameExpression>, std::unique_ptr<TestLangNamePattern"
                ">>>>"
            ),
            cc_getter_type=(
                "std::vector<std::optional<std::variant<std::unique_"
                "ptr<TestLangNameExpression>, "
                "std::unique_ptr<TestLangNamePattern>>>>*"
            ),
            cc_const_getter_type=(
                "const "
                "std::vector<std::optional<std::variant<std::"
                "unique_ptr<TestLangNameExpression>"
                ", std::unique_ptr<TestLangNamePattern>>>>*"
            ),
            cc_lang_name="TestLangName",
            td_types={ast_def_pb2.FIELD_KIND_RVAL: "Variadic<AnyType>"},
            cc_mlir_builder_type={
                ast_def_pb2.FIELD_KIND_RVAL: "std::vector<mlir::Value>"
            },
            cc_mlir_getter_type={
                ast_def_pb2.FIELD_KIND_RVAL: "mlir::OperandRange"
            },
        ),
    )

  def test_is_a_builtin_type(self):
    pb = type_pb2.TypePb()
    text_format.Parse("bool {}", pb)
    type_ = from_type_pb(pb, "TestLangName")

    self.assertIsInstance(type_, BuiltinType)

    self.assertIsInstance(type_, ScalarType)
    self.assertIsInstance(type_, NonListType)
    self.assertIsInstance(type_, Type)

    self.assertNotIsInstance(type_, ClassType)
    self.assertNotIsInstance(type_, VariantType)
    self.assertNotIsInstance(type_, ListType)

  def test_is_a_class_type(self):
    pb = type_pb2.TypePb()
    text_format.Parse('class: "Expression"', pb)
    type_ = from_type_pb(pb, "TestLangName")

    self.assertIsInstance(type_, ClassType)

    self.assertIsInstance(type_, ScalarType)
    self.assertIsInstance(type_, NonListType)
    self.assertIsInstance(type_, Type)

    self.assertNotIsInstance(type_, BuiltinType)
    self.assertNotIsInstance(type_, VariantType)
    self.assertNotIsInstance(type_, ListType)

  def test_is_a_variant_type(self):
    pb = type_pb2.TypePb()
    text_format.Parse(
        """
        variant {
          types { bool {} }
          types { string {} }
        }
        """,
        pb,
    )
    type_ = from_type_pb(pb, "TestLangName")

    self.assertIsInstance(type_, VariantType)

    self.assertIsInstance(type_, NonListType)
    self.assertIsInstance(type_, Type)

    self.assertNotIsInstance(type_, BuiltinType)
    self.assertNotIsInstance(type_, ClassType)
    self.assertNotIsInstance(type_, ScalarType)
    self.assertNotIsInstance(type_, ListType)

  def test_is_a_list_type(self):
    pb = type_pb2.TypePb()
    text_format.Parse("list { element_type { bool {} } }", pb)
    type_ = from_type_pb(pb, "TestLangName")

    self.assertIsInstance(type_, ListType)

    self.assertIsInstance(type_, Type)

    self.assertNotIsInstance(type_, BuiltinType)
    self.assertNotIsInstance(type_, ClassType)
    self.assertNotIsInstance(type_, ScalarType)
    self.assertNotIsInstance(type_, VariantType)
    self.assertNotIsInstance(type_, NonListType)

  def test_empty_type_is_invalid(self):
    pb = type_pb2.TypePb()
    text_format.Parse("", pb)
    with self.assertRaisesRegex(ValueError, "Invalid TypePb: KIND_NOT_SET."):
      from_type_pb(pb, "TestLangName")

    pb = type_pb2.TypePb()
    text_format.Parse("variant {}", pb)
    with self.assertRaisesRegex(ValueError, "Empty variant type."):
      from_type_pb(pb, "TestLangName")

    pb = type_pb2.TypePb()
    text_format.Parse("variant { types {} }", pb)
    with self.assertRaisesRegex(
        ValueError, "Invalid variant element type: KIND_NOT_SET."
    ):
      from_type_pb(pb, "TestLangName")

    pb = type_pb2.TypePb()
    text_format.Parse("list { element_type {} }", pb)
    with self.assertRaisesRegex(
        ValueError, "Invalid list element type: KIND_NOT_SET."
    ):
      from_type_pb(pb, "TestLangName")


if __name__ == "__main__":
  unittest.main()
