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
"""Port of maldoca/astgen/ast_serialize_printer.{h,cc} to Python.

Printer of the C++ Serialize() function for the AST
("ast_to_json.generated.cc").
"""

from __future__ import annotations

from maldoca.astgen import ast_def_pb2
from maldoca.astgen.ast_def import AstDef
from maldoca.astgen.ast_def import NodeDef
from maldoca.astgen.ast_gen_utils import get_ast_header_path
from maldoca.astgen.ast_gen_utils import OS_VALUE_VARIABLE_NAME
from maldoca.astgen.cc_printer_base import CcPrinterBase
from maldoca.astgen.symbol import Symbol
from maldoca.astgen.type import BuiltinType
from maldoca.astgen.type import ClassType
from maldoca.astgen.type import EnumType
from maldoca.astgen.type import ListType
from maldoca.astgen.type import MaybeNull
from maldoca.astgen.type import Type
from maldoca.astgen.type import VariantType

_MAYBE_ADD_COMMA_FUNCTION = """\
void MaybeAddComma(std::ostream &$os_variable$, bool &needs_comma) {
  if (needs_comma) {
    $os_variable$ << ",";
  }
  needs_comma = true;
}
"""

# `element`/`element_json` are the names PrintListSerialize uses for its own
# loop/temporary variables; a field named identically would shadow them.
_LIST_RHS_ELEMENT = "element"
_LIST_LHS_ELEMENT = "element_json"


class AstSerializePrinter(CcPrinterBase):
  """Printer of the C++ Serialize() function for the AST."""

  def print_ast(self, ast: AstDef, cc_namespace: str, ast_path: str) -> None:
    with self.with_vars({"os_variable": OS_VALUE_VARIABLE_NAME}):
      header_path = get_ast_header_path(ast_path)

      self.print_license()
      self.println()

      self.print_code_generation_warning()
      self.println()

      self.println("// IWYU pragma: begin_keep")
      self.println("// NOLINTBEGIN(whitespace/line_length)")
      self.println("// clang-format off")
      self.println()

      self.println("#include <cmath>")
      self.println("#include <limits>")
      self.println("#include <ostream>")
      self.println("#include <string>")
      self.println("#include <utility>")
      self.println()

      self.print_include_headers([
          header_path,
          "absl/log/log.h",
          "absl/memory/memory.h",
          "absl/status/status.h",
          "absl/strings/string_view.h",
          "nlohmann/json.hpp",
      ])
      self.println()

      self.print_enter_namespace(cc_namespace)
      self.println()

      self.println(_MAYBE_ADD_COMMA_FUNCTION)

      for node in ast.topological_sorted_nodes:
        self.print_title((Symbol(ast.lang_name) + node.name).to_pascal_case())
        self.println()

        self.print_serialize_fields_function(node, ast.lang_name)
        self.println()

        if not node.children:
          self.print_serialize_function(node, ast.lang_name)
          self.println()

      self.println("// clang-format on")
      self.println("// NOLINTEND(whitespace/line_length)")
      self.println("// IWYU pragma: end_keep")
      self.println()

      self.print_exit_namespace(cc_namespace)

  # Print*Serialize()
  #
  # Prints either:
  # - An assignment "<lhs> = ConvertSerialize(<rhs>);", or
  # - A variable definition "nlohmann::json <lhs> = ConvertSerialize(<rhs>);"
  #
  # - lhs: If printing an assignment, an lvalue expression of type
  #        nlohmann::json; if printing a variable definition, the name of
  #        that variable.
  # - rhs: An expression of type `type.cc_type()`.
  def print_builtin_serialize(
      self, type_: BuiltinType, lhs: str, rhs: str
  ) -> None:
    del type_  # Unused; matches the (also-unused) C++ parameter.
    with self.with_vars({"lhs": lhs, "rhs": rhs}):
      if lhs:
        self.println(
            "$os_variable$ << $lhs$ << (nlohmann::json($rhs$)).dump();"
        )
      else:
        self.println("$os_variable$ << (nlohmann::json($rhs$)).dump();")

  def print_enum_serialize(
      self, type_: EnumType, lhs: str, rhs: str, lang_name: str
  ) -> None:
    with self.with_vars({
        "lhs": lhs,
        "rhs": rhs,
        "EnumName": (Symbol(lang_name) + type_.name).to_pascal_case(),
    }):
      if lhs:
        self.println(
            r'$os_variable$ << $lhs$ << "\"" << $EnumName$ToString($rhs$) <<'
            r' "\"";'
        )
      else:
        self.println(
            r'$os_variable$ << "\"" << $EnumName$ToString($rhs$) << "\"";'
        )

  def print_class_serialize(self, type_: ClassType, lhs: str, rhs: str) -> None:
    del type_  # Unused; matches the (also-unused) C++ parameter.
    with self.with_vars({"lhs": lhs, "rhs": rhs}):
      if lhs:
        self.println("$os_variable$ << $lhs$;")
      self.println("$rhs$->Serialize($os_variable$);")

  def print_variant_serialize(
      self, variant_type: VariantType, lhs: str, rhs: str, lang_name: str
  ) -> None:
    with self.with_vars({"lhs": lhs, "rhs": rhs}):
      self.println("switch ($rhs$.index()) {")
      with self.with_indent():
        for i, scalar_type in enumerate(variant_type.types):
          with self.with_vars({"i": str(i)}):
            self.println("case $i$: {")
            with self.with_indent():
              self.print_serialize(
                  scalar_type, lhs, f"std::get<{i}>({rhs})", lang_name
              )
              self.println("break;")
            self.println("}")

        self.println("default:")
        self.println('  LOG(FATAL) << "Unreachable code.";')
      self.println("}")

  def print_list_serialize(
      self, list_type: ListType, lhs: str, rhs: str, lang_name: str
  ) -> None:
    assert lhs != _LIST_RHS_ELEMENT
    assert rhs != _LIST_RHS_ELEMENT
    assert lhs != _LIST_LHS_ELEMENT
    assert rhs != _LIST_LHS_ELEMENT

    with self.with_vars({
        "lhs": lhs,
        "rhs": rhs,
        "rhs_element": _LIST_RHS_ELEMENT,
    }):
      if lhs:
        self.println('$os_variable$ << $lhs$ << "[";')
      else:
        self.println('$os_variable$ << "[";')
      self.println("{")
      with self.with_indent():
        self.println("bool needs_comma = false;")
        self.println("for (const auto& $rhs_element$ : $rhs$) {")
        with self.with_indent():
          self.println("MaybeAddComma($os_variable$, needs_comma);")
          self.print_nullable_to_json(
              list_type.element_type,
              list_type.element_maybe_null,
              "",
              _LIST_RHS_ELEMENT,
              lang_name,
          )
        self.println("}")
      self.println("}")
      self.println('$os_variable$ << "]";')

  def print_serialize(
      self, type_: Type, lhs: str, rhs: str, lang_name: str
  ) -> None:
    if isinstance(type_, BuiltinType):
      self.print_builtin_serialize(type_, lhs, rhs)
    elif isinstance(type_, EnumType):
      self.print_enum_serialize(type_, lhs, rhs, lang_name)
    elif isinstance(type_, ClassType):
      self.print_class_serialize(type_, lhs, rhs)
    elif isinstance(type_, VariantType):
      self.print_variant_serialize(type_, lhs, rhs, lang_name)
    elif isinstance(type_, ListType):
      self.print_list_serialize(type_, lhs, rhs, lang_name)

  def print_nullable_to_json(
      self,
      type_: Type,
      maybe_null: MaybeNull,
      lhs: str,
      rhs: str,
      lang_name: str,
  ) -> None:
    if maybe_null == MaybeNull.NO:
      self.print_serialize(type_, lhs, rhs, lang_name)
      return

    with self.with_vars({"lhs": lhs, "rhs": rhs}):
      self.println("if ($rhs$.has_value()) {")
      with self.with_indent():
        rhs_value = f"{rhs}.value()"
        self.print_serialize(type_, lhs, rhs_value, lang_name)
      self.println("} else {")
      with self.with_indent():
        if lhs:
          self.println('$os_variable$ << $lhs$ << "null";')
        else:
          self.println('$os_variable$ << "null";')
      self.println("}")

  def print_serialize_fields_function(
      self, node: NodeDef, lang_name: str
  ) -> None:
    with self.with_vars(
        {"NodeType": (Symbol(lang_name) + node.name).to_pascal_case()}
    ):
      self.println(
          "void $NodeType$::SerializeFields(std::ostream& $os_variable$, "
          "bool &needs_comma) const {"
      )
      with self.with_indent():
        for field in node.fields:
          # E.g. `"\"fieldName\":"` (a C++ string literal, printed as-is).
          lhs = f'"\\"{field.name.to_camel_case()}\\":"'
          # E.g. field_name_
          rhs = f"{field.name.to_cc_var_name()}_"

          if field.optionalness == ast_def_pb2.OPTIONALNESS_UNSPECIFIED:
            raise ValueError("Invalid Optionalness. Should be a bug.")

          elif field.optionalness == ast_def_pb2.OPTIONALNESS_REQUIRED:
            self.println("MaybeAddComma($os_variable$, needs_comma);")
            self.print_serialize(field.type, lhs, rhs, lang_name)

          elif field.optionalness == ast_def_pb2.OPTIONALNESS_MAYBE_UNDEFINED:
            with self.with_vars({"rhs": rhs}):
              # If <rhs> == std::nullopt, the assignment does not happen.
              self.println("if ($rhs$.has_value()) {")
              with self.with_indent():
                rhs_value = f"{rhs}.value()"
                self.println("MaybeAddComma($os_variable$, needs_comma);")
                self.print_serialize(field.type, lhs, rhs_value, lang_name)
              self.println("}")

          elif field.optionalness == ast_def_pb2.OPTIONALNESS_MAYBE_NULL:
            self.println("MaybeAddComma($os_variable$, needs_comma);")
            self.print_nullable_to_json(
                field.type, MaybeNull.YES, lhs, rhs, lang_name
            )
      self.println("}")

  def print_serialize_function(self, node: NodeDef, lang_name: str) -> None:
    with self.with_vars({
        "NodeType": (Symbol(lang_name) + node.name).to_pascal_case(),
        "NodeTypeNoLangName": node.name,
    }):
      self.println(
          "void $NodeType$::Serialize(std::ostream& $os_variable$) const {"
      )
      with self.with_indent():
        self.println('$os_variable$ << "{";')
        self.println("{")
        with self.with_indent():
          self.println("bool needs_comma = false;")

          # The "type" field.
          if node.parents or node.children:
            self.println("MaybeAddComma($os_variable$, needs_comma);")
            self.println(
                r'$os_variable$ << "\"type\":\"$NodeTypeNoLangName$\"";'
            )

          # Assign fields of ancestors of this node.
          for ancestor in node.ancestors:
            with self.with_vars({
                "AncestorType": (
                    Symbol(lang_name) + ancestor.name
                ).to_pascal_case()
            }):
              self.println(
                  "$AncestorType$::SerializeFields($os_variable$, "
                  "needs_comma);"
              )

          # Assign fields of the node itself.
          self.println(
              "$NodeType$::SerializeFields($os_variable$, needs_comma);"
          )
        self.println("}")

        self.println('$os_variable$ << "}";')
      self.println("}")


# Prints the "ast_to_json.generated.cc" source file.
#
# - cc_namespace: The C++ namespace for all the AST node classes.
#   Example: "maldoca::astgen".
#
# - ast_path: The directory for the AST code.
#   "ast.generated.h" is in that directory.
#   This is used to print the #include.
def print_ast_to_json(ast: AstDef, cc_namespace: str, ast_path: str) -> str:
  printer = AstSerializePrinter()
  printer.print_ast(ast, cc_namespace, ast_path)
  return printer.content()
