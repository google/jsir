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
"""Port of maldoca/astgen/ts_interface_printer.{h,cc} to Python.

Printer of the TypeScript interface definition for the AST
("ast_ts_interface.generated"). Only used by tests / as documentation, not
by ast_gen_main.

Format:

interface ObjectMember <: Node {
  key: Expression;
  computed: boolean;
  decorators?: [ Decorator ];
}
"""

from __future__ import annotations

from maldoca.astgen import ast_def_pb2
from maldoca.astgen.ast_def import AstDef
from maldoca.astgen.ast_def import EnumDef
from maldoca.astgen.ast_def import FieldDef
from maldoca.astgen.ast_def import NodeDef
from maldoca.astgen.ast_gen_utils import TabPrinter
from maldoca.astgen.ast_gen_utils import TabPrinterOptions
from maldoca.astgen.printer_base import Printer
from maldoca.astgen.type import MaybeNull


def _cescape(s: str) -> str:
  out = []
  for ch in s:
    if ch == "\\":
      out.append("\\\\")
    elif ch == '"':
      out.append('\\"')
    elif ch == "\n":
      out.append("\\n")
    elif ch == "\t":
      out.append("\\t")
    elif ch == "\r":
      out.append("\\r")
    else:
      out.append(ch)
  return "".join(out)


class TsInterfacePrinter(Printer):
  """Printer of the TypeScript interface definition for the AST."""

  # Prints the "ast_ts_interface.generated" file.
  #
  # See test cases in test/ for examples.
  def print_ast(self, ast: AstDef) -> None:
    for enum_def in ast.enum_defs:
      self.print_enum(enum_def, ast.lang_name)
      self.println()

    # NOTE: Iterates `node_names` (original definition order), not
    # `topological_sorted_nodes` -- unlike every other printer.
    for name in ast.node_names:
      node = ast.nodes[name]
      self.print_node(node)
      self.println()

  # Prints an enum definition.
  #
  # See test cases in test/ for examples.
  def print_enum(self, enum_def: EnumDef, lang_name: str) -> None:
    del lang_name  # Unused; matches the (also-unused) C++ parameter.
    with self.with_vars({"EnumName": enum_def.name.to_pascal_case()}):
      self.println("type $EnumName$ =")
      with self.with_indent(4):
        for member in enum_def.members:
          with self.with_vars(
              {"string_value": _cescape(member.string_value)}
          ):
            self.println('| "$string_value$"')

  # Prints the class declaration for a node.
  #
  # See test cases in test/ for examples.
  def print_node(self, node: NodeDef) -> None:
    with self.with_vars({"NodeType": node.name}):
      self.print("interface $NodeType$")

      if node.parents:
        self.print(" <: ")

        with TabPrinter(
            TabPrinterOptions(print_separator=lambda: self.print(", "))
        ) as separator_printer:
          for parent in node.parents:
            separator_printer.print()
            self.print(parent.name)

      self.println(" {")
      with self.with_indent():
        for field in node.fields:
          self.print_field_def(field)
      self.println("}")

  # Prints the definition of a field.
  #
  # Format:
  #  <fieldName>: <js_type>
  #  <fieldName>?: <js_type>
  #
  # - fieldName: Printed as camelCase.
  # - js_type: See `Type.js_type()`.
  #
  # Example:
  #  right: Expression
  #  param?: Pattern
  def print_field_def(self, field: FieldDef) -> None:
    self.print(field.name.to_camel_case())

    if field.optionalness == ast_def_pb2.OPTIONALNESS_MAYBE_UNDEFINED:
      self.print("?")

    self.print(": ")

    maybe_null = (
        MaybeNull.YES
        if field.optionalness == ast_def_pb2.OPTIONALNESS_MAYBE_NULL
        else MaybeNull.NO
    )
    self.print(field.type.js_type_with_maybe_null(maybe_null))

    self.println()


def print_ts_interface(ast: AstDef) -> str:
  printer = TsInterfacePrinter()
  printer.print_ast(ast)
  return printer.content()
