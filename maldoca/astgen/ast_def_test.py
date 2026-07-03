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
"""Tests for maldoca/astgen/ast_def.py.

`AstDef.from_proto()` (ported from `AstDef::FromProto()` in ast_def.cc) has
no dedicated C++ unit test in this repo -- it's only exercised indirectly by
the golden printer tests under maldoca/astgen/test/*. These tests cover the
graph algorithms (ancestors/descendants/leaves/aggregation, topological
sort, node_type_enum synthesis, ir_op_name/ir_op_mnemonic) and schema
validation directly, since they're intricate enough to warrant focused
coverage independent of the (much larger) golden printer tests ported later.
"""

import unittest

from google.protobuf import text_format

from maldoca.astgen import ast_def_pb2
from maldoca.astgen.ast_def import AstDef
from maldoca.astgen.symbol import Symbol
from maldoca.astgen.type import ClassType


def _parse(text: str) -> ast_def_pb2.AstDefPb:
  pb = ast_def_pb2.AstDefPb()
  text_format.Parse(text, pb)
  return pb


class AstDefGraphTest(unittest.TestCase):

  def test_ancestors_and_descendants(self):
    # CatDog <: Cat, Dog <: Animal (see comment in
    # _topological_sort_dependencies).
    ast_def = AstDef.from_proto(
        _parse("""
          lang_name: "la"
          nodes { name: "Animal" }
          nodes { name: "Cat" parents: "Animal" type: "Cat" }
          nodes { name: "Dog" parents: "Animal" type: "Dog" }
          nodes { name: "CatDog" parents: "Cat" parents: "Dog" type: "CatDog" }
        """)
    )

    cat_dog = ast_def.nodes["CatDog"]
    self.assertEqual(
        [n.name for n in cat_dog.ancestors], ["Animal", "Cat", "Dog"]
    )

    animal = ast_def.nodes["Animal"]
    # Note: `descendants` reuses the same DFS-postorder helper as
    # `ancestors`, but walking `children` edges instead of `parents` edges.
    # Because CatDog is reached first via the Cat branch, it's fully
    # expanded (and appended) before the Dog branch's traversal reaches it
    # again (where it's skipped as already-visited) -- hence CatDog sorts
    # before Cat and Dog here, even though it's the deepest descendant.
    self.assertEqual(
        [n.name for n in animal.descendants], ["CatDog", "Cat", "Dog"]
    )
    self.assertEqual([n.name for n in animal.leaves], ["CatDog"])
    self.assertEqual([n.name for n in animal.children], ["Cat", "Dog"])

  def test_aggregated_fields(self):
    ast_def = AstDef.from_proto(
        _parse("""
          lang_name: "la"
          nodes {
            name: "Base"
            fields {
              name: "x"
              type { bool {} }
              optionalness: OPTIONALNESS_REQUIRED
            }
          }
          nodes {
            name: "Derived"
            parents: "Base"
            type: "Derived"
            fields {
              name: "y"
              type { string {} }
              optionalness: OPTIONALNESS_REQUIRED
            }
          }
        """)
    )

    derived = ast_def.nodes["Derived"]
    self.assertEqual(
        [f.name.to_camel_case() for f in derived.aggregated_fields],
        ["x", "y"],
    )

  def test_node_type_enum_synthesized_for_root_with_children(self):
    ast_def = AstDef.from_proto(
        _parse("""
          lang_name: "la"
          nodes { name: "Animal" }
          nodes { name: "Cat" parents: "Animal" type: "Cat" }
          nodes { name: "Dog" parents: "Animal" type: "Dog" }
        """)
    )

    animal = ast_def.nodes["Animal"]
    self.assertIsNotNone(animal.node_type_enum)
    self.assertEqual(animal.node_type_enum.name, Symbol("AnimalType"))
    self.assertEqual(
        [m.name.to_pascal_case() for m in animal.node_type_enum.members],
        ["Cat", "Dog"],
    )

    # Leaves (and nodes without children) don't get a synthesized enum.
    self.assertIsNone(ast_def.nodes["Cat"].node_type_enum)

  def test_union_type_becomes_a_parent(self):
    ast_def = AstDef.from_proto(
        _parse("""
          lang_name: "la"
          nodes { name: "Cat" type: "Cat" }
          nodes { name: "Dog" type: "Dog" }
          union_types {
            name: "Animal"
            types: "Cat"
            types: "Dog"
          }
        """)
    )

    self.assertIn("Animal", ast_def.nodes)
    self.assertEqual(
        [p.name for p in ast_def.nodes["Cat"].parents], ["Animal"]
    )
    self.assertEqual(
        [n.name for n in ast_def.nodes["Animal"].leaves], ["Cat", "Dog"]
    )

  def test_topological_sorted_nodes_field_type_dependency(self):
    # `Wrapper` has a field of type `Inner`, so `Inner` must be sorted
    # before `Wrapper` even though there's no inheritance relationship.
    ast_def = AstDef.from_proto(
        _parse("""
          lang_name: "la"
          nodes {
            name: "Wrapper"
            type: "Wrapper"
            fields {
              name: "inner"
              type { class: "Inner" }
              optionalness: OPTIONALNESS_REQUIRED
              kind: FIELD_KIND_RVAL
            }
          }
          nodes { name: "Inner" type: "Inner" }
        """)
    )

    order = [n.name for n in ast_def.topological_sorted_nodes]
    self.assertLess(order.index("Inner"), order.index("Wrapper"))

  def test_class_type_field_resolves_node_def(self):
    ast_def = AstDef.from_proto(
        _parse("""
          lang_name: "la"
          nodes {
            name: "Wrapper"
            type: "Wrapper"
            fields {
              name: "inner"
              type { class: "Inner" }
              optionalness: OPTIONALNESS_REQUIRED
              kind: FIELD_KIND_RVAL
            }
          }
          nodes { name: "Inner" type: "Inner" }
        """)
    )

    field_type = ast_def.nodes["Wrapper"].fields[0].type
    self.assertIsInstance(field_type, ClassType)
    self.assertIs(field_type.node_def, ast_def.nodes["Inner"])


class NodeDefIrOpNameTest(unittest.TestCase):

  def test_leaf_rval_and_lval(self):
    ast_def = AstDef.from_proto(
        _parse("""
          lang_name: "la"
          nodes { name: "Identifier" type: "Identifier" }
        """)
    )
    identifier = ast_def.nodes["Identifier"]

    # Note: the IR dialect prefix is `lang_name + "ir"` concatenated as a
    # single fused string ("la" + "ir" = "lair"), not word-joined -- so it
    # parses as one Symbol word, matching the real "lair" dialect name seen
    # in maldoca/astgen/test/lambda/lair_dialect.td.
    self.assertEqual(
        identifier.ir_op_name("la", ast_def_pb2.FIELD_KIND_RVAL),
        Symbol("lairIdentifierOp"),
    )
    self.assertEqual(
        identifier.ir_op_name("la", ast_def_pb2.FIELD_KIND_LVAL),
        Symbol("lairIdentifierRefOp"),
    )
    self.assertEqual(
        identifier.ir_op_mnemonic(ast_def_pb2.FIELD_KIND_RVAL),
        Symbol("identifier"),
    )
    self.assertEqual(
        identifier.ir_op_mnemonic(ast_def_pb2.FIELD_KIND_LVAL),
        Symbol("identifierRef"),
    )

  def test_non_leaf_gets_interface_suffix(self):
    ast_def = AstDef.from_proto(
        _parse("""
          lang_name: "la"
          nodes { name: "Expression" }
          nodes { name: "Identifier" parents: "Expression" type: "Identifier" }
        """)
    )
    expression = ast_def.nodes["Expression"]
    self.assertEqual(
        expression.ir_op_name("la", ast_def_pb2.FIELD_KIND_RVAL),
        Symbol("lairExpressionOpInterface"),
    )

  def test_custom_ir_op_name_overrides_and_suppresses_descendants(self):
    ast_def = AstDef.from_proto(
        _parse("""
          lang_name: "la"
          nodes { name: "Expression" }
          nodes {
            name: "NumericLiteral"
            parents: "Expression"
            type: "NumericLiteral"
            ir_op_name: "mlir::arith::ConstantOp"
          }
        """)
    )
    numeric_literal = ast_def.nodes["NumericLiteral"]
    self.assertEqual(
        numeric_literal.ir_op_name("la", ast_def_pb2.FIELD_KIND_RVAL),
        Symbol("mlir::arith::ConstantOp"),
    )
    self.assertIsNone(
        numeric_literal.ir_op_mnemonic(ast_def_pb2.FIELD_KIND_RVAL)
    )

    # The ancestor falls back to None because a descendant has a custom
    # name.
    expression = ast_def.nodes["Expression"]
    self.assertIsNone(
        expression.ir_op_name("la", ast_def_pb2.FIELD_KIND_RVAL)
    )


class AstDefValidationTest(unittest.TestCase):

  def test_duplicate_node_name_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "Foo already exists!"):
      AstDef.from_proto(
          _parse("""
            lang_name: "la"
            nodes { name: "Foo" }
            nodes { name: "Foo" }
          """)
      )

  def test_missing_parent_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "Parent Bar doesn't exist!"):
      AstDef.from_proto(
          _parse("""
            lang_name: "la"
            nodes { name: "Foo" parents: "Bar" type: "Foo" }
          """)
      )

  def test_missing_union_member_is_rejected(self):
    with self.assertRaisesRegex(
        ValueError, "Union type Animal: member Cat doesn't exist!"
    ):
      AstDef.from_proto(
          _parse("""
            lang_name: "la"
            union_types { name: "Animal" types: "Cat" }
          """)
      )

  def test_non_camel_case_field_name_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "not in camelCase"):
      AstDef.from_proto(
          _parse("""
            lang_name: "la"
            nodes {
              name: "Foo"
              type: "Foo"
              fields {
                name: "NotCamelCase"
                type { bool {} }
                optionalness: OPTIONALNESS_REQUIRED
              }
            }
          """)
      )

  def test_non_pascal_case_enum_name_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "not in PascalCase"):
      AstDef.from_proto(
          _parse("""
            lang_name: "la"
            enums { name: "notPascalCase" }
          """)
      )


if __name__ == "__main__":
  unittest.main()
