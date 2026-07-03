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
"""Port of maldoca/astgen/ast_source_printer.{h,cc} to Python.

Printer of the C++ source for the AST ("ast.generated.cc").
"""

from __future__ import annotations

from maldoca.astgen import ast_def_pb2
from maldoca.astgen.ast_def import AstDef
from maldoca.astgen.ast_def import EnumDef
from maldoca.astgen.ast_def import FieldDef
from maldoca.astgen.ast_def import NodeDef
from maldoca.astgen.ast_gen_utils import get_ast_header_path
from maldoca.astgen.ast_gen_utils import TabPrinter
from maldoca.astgen.ast_gen_utils import TabPrinterOptions
from maldoca.astgen.cc_printer_base import CcPrinterBase
from maldoca.astgen.symbol import Symbol
from maldoca.astgen.type import BuiltinType
from maldoca.astgen.type import BuiltinTypeKind
from maldoca.astgen.type import EnumType
from maldoca.astgen.type import Type
from maldoca.astgen.type import TypeKind
from maldoca.astgen.type import VariantType

_STRING_TO_ENUM_BODY = """\
auto it = kMap->find(s);
if (it == kMap->end()) {
  return absl::InvalidArgumentError(absl::StrCat("Invalid string for $EnumName$: ", s));
}
return it->second;"""


def _cescape(s: str) -> str:
  # Port of absl::CEscape() for the small set of characters that actually
  # show up in astgen enum string values (identifiers / JS operators).
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


class AstSourcePrinter(CcPrinterBase):
  """Printer of the C++ source for the AST."""

  # Prints the "ast.generated.cc" file, which includes the definitions of
  # getters and setters of all the AST node classes.
  #
  # - cc_namespace: A namespace separated by "::".
  #   This is used to print C++ namespaces.
  #
  # - ast_path: The directory for the AST code.
  #   "ast.generated.h" is in that directory.
  #   This is used to print the #include.
  def print_ast(self, ast: AstDef, cc_namespace: str, ast_path: str) -> None:
    header_path = get_ast_header_path(ast_path)

    self.print_license()
    self.println()

    self.print_code_generation_warning()
    self.println()

    self.print_include_header(header_path)
    self.println()

    self.println("// IWYU pragma: begin_keep")
    self.println("// NOLINTBEGIN(whitespace/line_length)")
    self.println("// clang-format off")
    self.println()

    self.println("#include <cstdint>")
    self.println("#include <memory>")
    self.println("#include <optional>")
    self.println("#include <string>")
    self.println("#include <utility>")
    self.println("#include <variant>")
    self.println("#include <vector>")
    self.println()

    self.print_include_header("absl/container/flat_hash_map.h")
    self.print_include_header("absl/memory/memory.h")
    self.print_include_header("absl/log/log.h")
    self.print_include_header("absl/status/status.h")
    self.print_include_header("absl/status/statusor.h")
    self.print_include_header("absl/strings/str_cat.h")
    self.print_include_header("absl/strings/string_view.h")
    self.print_include_header("nlohmann/json.hpp")
    self.println()

    self.print_enter_namespace(cc_namespace)
    self.println()

    for enum_def in ast.enum_defs:
      self.print_enum(enum_def, ast.lang_name)
      self.println()

    for node in ast.topological_sorted_nodes:
      self.print_node(node, ast.lang_name)

    self.println("// clang-format on")
    self.println("// NOLINTEND(whitespace/line_length)")
    self.println("// IWYU pragma: end_keep")
    self.println()

    self.print_exit_namespace(cc_namespace)

  # Prints the string conversion functions.
  #
  # Example:
  #
  #  absl::string_view UnaryOperatorToString(UnaryOperator unary_operator) {
  #    ...
  #  }
  #
  #  absl::StatusOr<UnaryOperator> StringToUnaryOperator(absl::string_view s) {
  #    ...
  #  }
  def print_enum(self, enum_def: EnumDef, lang_name: str) -> None:
    with self.with_vars({
        "EnumName": (Symbol(lang_name) + enum_def.name).to_pascal_case(),
        "enum_name": enum_def.name.to_snake_case(),
    }):
      self.println(
          "absl::string_view $EnumName$ToString($EnumName$ $enum_name$) {"
      )
      with self.with_indent():
        self.println("switch ($enum_name$) {")
        with self.with_indent():
          for member in enum_def.members:
            with self.with_vars({
                "kMemberName": (Symbol("k") + member.name).to_camel_case(),
                "string_value": _cescape(member.string_value),
            }):
              self.println("case $EnumName$::$kMemberName$:")
              self.println('  return "$string_value$";')
        self.println("}")
      self.println("}")
      self.println()

      self.println(
          "absl::StatusOr<$EnumName$> StringTo$EnumName$(absl::string_view"
          " s) {"
      )
      with self.with_indent():
        self.println(
            "static const auto *kMap = "
            "new absl::flat_hash_map<absl::string_view, $EnumName$> {"
        )
        with self.with_indent(4):
          for member in enum_def.members:
            with self.with_vars({
                "kMemberName": (Symbol("k") + member.name).to_camel_case(),
                "string_value": _cescape(member.string_value),
            }):
              self.println('{"$string_value$", $EnumName$::$kMemberName$},')
        self.println("};")
        self.println()

        self.println(_STRING_TO_ENUM_BODY)
      self.println("}")

  # Prints the getters and setters of one AST node class.
  def print_node(self, node: NodeDef, lang_name: str) -> None:
    self.print_title((Symbol(lang_name) + node.name).to_pascal_case())
    self.println()

    if node.node_type_enum is not None:
      self.print_enum(node.node_type_enum, lang_name)
      self.println()

    if node.aggregated_fields:
      self.print_constructor(node, lang_name)
      self.println()

    for field in node.fields:
      type_ = field.type
      is_optional = field.optionalness != ast_def_pb2.OPTIONALNESS_REQUIRED

      cc_getter_type = self.cc_mutable_getter_type(field)
      cc_const_getter_type = self.cc_const_getter_type(field)

      with self.with_vars({
          "NodeType": (Symbol(lang_name) + node.name).to_pascal_case(),
          "cc_getter_type": cc_getter_type,
          "cc_const_getter_type": cc_const_getter_type,
          "cc_type": self.cc_type(field),
          "field_name": field.name.to_cc_var_name(),
      }):
        # If both the mutable getter and const getter would have the same
        # return type, then we just skip the mutable getter and only keep
        # the const getter.
        if cc_getter_type != cc_const_getter_type:
          self.println("$cc_getter_type$ $NodeType$::$field_name$() {")
          with self.with_indent():
            self.print_getter_body(field.name, type_, is_optional)
          self.println("}")
          self.println()

        self.println(
            "$cc_const_getter_type$ $NodeType$::$field_name$() const {"
        )
        with self.with_indent():
          self.print_getter_body(field.name, type_, is_optional)
        self.println("}")
        self.println()

        self.println(
            "void $NodeType$::set_$field_name$($cc_type$ $field_name$) {"
        )
        with self.with_indent():
          self.print_setter_body(field.name, type_, is_optional)
        self.println("}")
        self.println()

  def print_constructor(self, node: NodeDef, lang_name: str) -> None:
    with self.with_vars(
        {"NodeType": (Symbol(lang_name) + node.name).to_pascal_case()}
    ):
      self.print("$NodeType$::$NodeType$(")
      if node.aggregated_fields:
        self.println()
        with self.with_indent(4):
          with TabPrinter(
              TabPrinterOptions(print_separator=lambda: self.print(",\n"))
          ) as separator_printer:
            for field in node.aggregated_fields:
              with self.with_vars({
                  "cc_type": self.cc_type(field),
                  "field_name": field.name.to_cc_var_name(),
              }):
                separator_printer.print()
                self.print("$cc_type$ $field_name$")
      self.println(")")

      with self.with_indent(4):

        def print_prefix() -> None:
          self.print(": ")
          self.indent()

        def print_postfix() -> None:
          self.outdent()

        with TabPrinter(
            TabPrinterOptions(
                print_prefix=print_prefix,
                print_separator=lambda: self.print(",\n"),
                print_postfix=print_postfix,
            )
        ) as tab_printer:
          for ancestor in node.ancestors:
            tab_printer.print()

            with self.with_vars({
                "AncestorType": (
                    Symbol(lang_name) + ancestor.name
                ).to_pascal_case()
            }):
              self.print("$AncestorType$(")

              with TabPrinter(
                  TabPrinterOptions(print_separator=lambda: self.print(", "))
              ) as ancestor_tab_printer:
                for field in ancestor.aggregated_fields:
                  ancestor_tab_printer.print()
                  with self.with_vars(
                      {"field_name": field.name.to_cc_var_name()}
                  ):
                    self.print("std::move($field_name$)")

              self.print(")")

          for field in node.fields:
            with self.with_vars(
                {"field_name": field.name.to_cc_var_name()}
            ):
              tab_printer.print()
              self.print("$field_name$_(std::move($field_name$))")

      self.println(" {}")

  # Prints the C++ code that returns a value that's compatible with the
  # types `type.cc_mutable_getter_type()` and `type.cc_const_getter_type()`.
  #
  # `cc_expr` is an lvalue expression of the type `type.cc_type()`.
  def print_getter_body_expr(self, cc_expr: str, type_: Type) -> None:
    with self.with_vars({"cc_expr": cc_expr}):
      if type_.kind in (TypeKind.BUILTIN, TypeKind.ENUM):
        self.println("return $cc_expr$;")

      elif type_.kind == TypeKind.CLASS:
        self.println("return $cc_expr$.get();")

      elif type_.kind == TypeKind.VARIANT:
        assert isinstance(type_, VariantType)
        self.println("switch ($cc_expr$.index()) {")
        with self.with_indent():
          for i, scalar_type in enumerate(type_.types):
            with self.with_vars({"i": str(i)}):
              self.println("case $i$: {")
              with self.with_indent():
                self.print_getter_body_expr(
                    f"std::get<{i}>({cc_expr})", scalar_type
                )
              self.println("}")

          self.println("default:")
          self.println('  LOG(FATAL) << "Unreachable code.";')
        self.println("}")

      elif type_.kind == TypeKind.LIST:
        self.println("return &$cc_expr$;")

  # Prints the C++ code that returns a value that's compatible with the
  # types `type.cc_mutable_getter_type(is_optional)` and
  # `type.cc_const_getter_type(is_optional)`.
  #
  # `cc_expr` is an lvalue expression of the type `type.cc_type()`.
  def print_getter_body(
      self, field_name: Symbol, type_: Type, is_optional: bool
  ) -> None:
    if is_optional:
      with self.with_vars({"field_name": field_name.to_cc_var_name()}):
        self.println("if (!$field_name$_.has_value()) {")
        self.println("  return std::nullopt;")
        self.println("} else {")
        with self.with_indent():
          value_cc_expr = f"{field_name.to_cc_var_name()}_.value()"
          self.print_getter_body_expr(value_cc_expr, type_)
        self.println("}")

    else:
      self.print_getter_body_expr(f"{field_name.to_cc_var_name()}_", type_)

  # Prints the C++ code that sets one field.
  #
  # `field_name` is an lvalue expression that has the type
  # `type.cc_type(is_optional)`. We need to set the field `field_name_`.
  def print_setter_body(
      self, field_name: Symbol, type_: Type, is_optional: bool
  ) -> None:
    del is_optional  # Unused; matches the (also-unused) C++ parameter.

    with self.with_vars({"field_name": field_name.to_cc_var_name()}):
      if isinstance(type_, BuiltinType) and type_.builtin_kind in (
          BuiltinTypeKind.BOOL,
          BuiltinTypeKind.DOUBLE,
      ):
        self.println("$field_name$_ = $field_name$;")
        return

      if isinstance(type_, EnumType):
        self.println("$field_name$_ = $field_name$;")
        return

      self.println("$field_name$_ = std::move($field_name$);")


def print_ast_source(ast: AstDef, cc_namespace: str, ast_path: str) -> str:
  printer = AstSourcePrinter()
  printer.print_ast(ast, cc_namespace, ast_path)
  return printer.content()
