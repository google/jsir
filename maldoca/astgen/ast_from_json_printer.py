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
"""Port of maldoca/astgen/ast_from_json_printer.{h,cc} to Python.

Prints "ast_from_json.generated.cc": the `FromJson()` static factory
functions that parse `nlohmann::json` back into AST node objects.
"""

from __future__ import annotations

from maldoca.astgen import ast_def_pb2
from maldoca.astgen.ast_def import AstDef
from maldoca.astgen.ast_def import FieldDef
from maldoca.astgen.ast_def import NodeDef
from maldoca.astgen.ast_gen_utils import get_ast_header_path
from maldoca.astgen.ast_gen_utils import IfStmtCase
from maldoca.astgen.ast_gen_utils import IfStmtPrinter
from maldoca.astgen.ast_gen_utils import JSON_VALUE_VARIABLE_NAME
from maldoca.astgen.ast_gen_utils import TabPrinter
from maldoca.astgen.ast_gen_utils import TabPrinterOptions
from maldoca.astgen.cc_printer_base import CcPrinterBase
from maldoca.astgen.symbol import Symbol
from maldoca.astgen.type import BuiltinType
from maldoca.astgen.type import BuiltinTypeKind
from maldoca.astgen.type import ClassType
from maldoca.astgen.type import EnumType
from maldoca.astgen.type import ListType
from maldoca.astgen.type import maybe_null_to_optionalness
from maldoca.astgen.type import MaybeNull
from maldoca.astgen.type import ScalarType
from maldoca.astgen.type import Type
from maldoca.astgen.type import VariantType

_TYPE_CHECKER_TYPE_LOOKUP = """\
auto type_it = $json_variable$.find("type");
if (type_it == $json_variable$.end()) {
  return false;
}
const nlohmann::json &type_json = type_it.value();
if (!type_json.is_string()) {
  return false;
}
const std::string &type = type_json.get<std::string>();"""

_CHECK_IS_OBJECT = """\
if (!$json_variable$.is_object()) {
  return absl::InvalidArgumentError("JSON is not an object.");
}"""


def _collect_checked_classes(
    type_: Type, is_part_of_variant: bool, node_names: set[str]
) -> None:
  if isinstance(type_, ClassType):
    if is_part_of_variant:
      node_names.add(type_.name.to_pascal_case())
  elif isinstance(type_, VariantType):
    for element_type in type_.types:
      _collect_checked_classes(element_type, True, node_names)
  elif isinstance(type_, ListType):
    _collect_checked_classes(type_.element_type, is_part_of_variant, node_names)
  # BuiltinType, EnumType: no checked classes.


def _get_checked_classes(ast: AstDef) -> set[str]:
  checked_classes: set[str] = set()
  for node in ast.topological_sorted_nodes:
    for field in node.fields:
      _collect_checked_classes(field.type, False, checked_classes)
  return checked_classes


class AstFromJsonPrinter(CcPrinterBase):
  """Printer of the `FromJson()` functions for the AST."""

  def print_ast(self, ast: AstDef, cc_namespace: str, ast_path: str) -> None:
    with self.with_vars({"json_variable": JSON_VALUE_VARIABLE_NAME}):
      header_path = get_ast_header_path(ast_path)

      self.print_license()
      self.println()

      self.print_code_generation_warning()
      self.println()

      self.println("// NOLINTBEGIN(whitespace/line_length)")
      self.println("// clang-format off")
      self.println("// IWYU pragma: begin_keep")
      self.println()

      self.println("#include <cstdint>")
      self.println("#include <memory>")
      self.println("#include <optional>")
      self.println("#include <string>")
      self.println("#include <utility>")
      self.println("#include <variant>")
      self.println("#include <vector>")
      self.println()

      self.print_include_headers([
          header_path,
          "absl/container/flat_hash_set.h",
          "absl/memory/memory.h",
          "absl/status/status.h",
          "absl/status/status_macros.h",
          "absl/status/statusor.h",
          "absl/strings/str_cat.h",
          "absl/strings/string_view.h",
          "maldoca/astgen/ast_from_json_utils.h",
          "nlohmann/json.hpp",
      ])
      self.println()

      self.print_enter_namespace(cc_namespace)
      self.println()

      checked_classes = _get_checked_classes(ast)

      for node in ast.topological_sorted_nodes:
        self.print_title((Symbol(ast.lang_name) + node.name).to_pascal_case())
        self.println()

        if node.name in checked_classes:
          self.print_type_checker(node)
          self.println()

        for field in node.fields:
          self.print_get_field_function(node.name, field, ast.lang_name)
          self.println()

        self.print_from_json_function(node, ast.lang_name)
        self.println()

      self.println("// clang-format on")
      self.println("// NOLINTEND(whitespace/line_length)")
      self.println("// IWYU pragma: end_keep")
      self.println()

      self.print_exit_namespace(cc_namespace)

  def print_type_checker(self, node: NodeDef) -> None:
    with self.with_vars({
        "NodeType": node.name,
        "json_variable": JSON_VALUE_VARIABLE_NAME,
    }):
      self.println(
          "static bool Is$NodeType$(const nlohmann::json& $json_variable$) {"
      )
      try:
        with self.with_indent():
          self.println("if (!$json_variable$.is_object()) {")
          self.println("  return false;")
          self.println("}")

          if not node.children and not node.parents:
            # This is not a virtual class.
            self.println("return true;")
            return

          self.println(_TYPE_CHECKER_TYPE_LOOKUP)

          if node.leaves:
            self.println(
                "static const auto *kTypes = new"
                " absl::flat_hash_set<std::string>{"
            )
            with self.with_indent(4):
              for leaf in node.leaves:
                with self.with_vars({"LeafType": leaf.name}):
                  self.println('"$LeafType$",')
            self.println("};")
            self.println()

            self.println("return kTypes->contains(type);")

          else:
            assert node.name == node.type
            self.println('return type == "$NodeType$";')
      finally:
        self.println("}")

  def print_get_field_function(
      self, node_name: str, field: FieldDef, lang_name: str
  ) -> None:
    if field.optionalness == ast_def_pb2.OPTIONALNESS_REQUIRED:
      get_field_function_name = "GetRequiredField"
    elif field.optionalness == ast_def_pb2.OPTIONALNESS_MAYBE_NULL:
      get_field_function_name = "GetNullableField"
    elif field.optionalness == ast_def_pb2.OPTIONALNESS_MAYBE_UNDEFINED:
      get_field_function_name = "GetOptionalField"
    else:
      raise ValueError("Unreachable code.")

    with self.with_vars({
        "NodeType": (Symbol(lang_name) + node_name).to_pascal_case(),
        "return_cc_type": self.cc_type(field),
        "field_cc_type": field.type.cc_type(),
        "fieldName": field.name.to_camel_case(),
        "FieldName": field.name.to_pascal_case(),
        "GetField": get_field_function_name,
    }):
      self.println("absl::StatusOr<$return_cc_type$>")
      self.println(
          "$NodeType$::Get$FieldName$(const nlohmann::json&"
          " $json_variable$) {"
      )
      with self.with_indent():
        self.println("return $GetField$<$field_cc_type$>(")
        with self.with_indent(4):
          self.println("$json_variable$,")
          self.println('"$fieldName$",')
          self.print_converter(field.type, lang_name)
          self.println()
        self.println(");")
      self.println("}")

  def print_type_checker_name(self, type_: ScalarType) -> None:
    if isinstance(type_, ClassType):
      self.print("Is" + type_.name.to_pascal_case())
    elif isinstance(type_, EnumType):
      self.print("Is" + type_.name.to_pascal_case())
    elif isinstance(type_, BuiltinType):
      self.print({
          BuiltinTypeKind.STRING: "IsString",
          BuiltinTypeKind.BOOL: "IsBool",
          BuiltinTypeKind.INT64: "IsInt64",
          BuiltinTypeKind.DOUBLE: "IsDouble",
      }[type_.builtin_kind])
    else:
      raise ValueError("Unreachable code.")

  def print_converter(self, type_: Type, lang_name: str) -> None:
    if isinstance(type_, ListType):
      self.print_list_converter(type_, lang_name)
    elif isinstance(type_, VariantType):
      self.print_variant_converter(type_, lang_name)
    elif isinstance(type_, ClassType):
      self.print_class_converter(type_, lang_name)
    elif isinstance(type_, EnumType):
      self.print_enum_converter(type_, lang_name)
    elif isinstance(type_, BuiltinType):
      self.print_builtin_converter(type_, lang_name)

  def print_builtin_converter(
      self, builtin_type: BuiltinType, lang_name: str
  ) -> None:
    del lang_name  # Unused; matches the (also-unused) C++ parameter.
    self.print({
        BuiltinTypeKind.STRING: "JsonToString",
        BuiltinTypeKind.BOOL: "JsonToBool",
        BuiltinTypeKind.INT64: "JsonToInt64",
        BuiltinTypeKind.DOUBLE: "JsonToDouble",
    }[builtin_type.builtin_kind])

  def print_enum_converter(self, enum_type: EnumType, lang_name: str) -> None:
    with self.with_vars({
        "EnumName": (Symbol(lang_name) + enum_type.name).to_pascal_case(),
        "cc_type": enum_type.cc_type(),
    }):
      self.print("Enum<$cc_type$>(StringTo$EnumName$)")

  def print_class_converter(
      self, class_type: ClassType, lang_name: str
  ) -> None:
    del lang_name  # Unused; matches the (also-unused) C++ parameter.
    self.print(f"{class_type.cc_class_name()}::FromJson")

  def print_variant_converter(
      self, variant_type: VariantType, lang_name: str
  ) -> None:
    # Variant(
    #     VariantOption<double>{
    #         .predicate = IsDouble,
    #         .converter = JsonToDouble,
    #     },
    #     VariantOption<std::string>{
    #         .predicate = IsString,
    #         .converter = JsonToString,
    #     }
    # )
    self.println("Variant(")
    with self.with_indent(4):
      with TabPrinter(
          TabPrinterOptions(print_separator=lambda: self.print(",\n"))
      ) as tab_printer:
        for scalar_type in variant_type.types:
          tab_printer.print()

          self.println(f"VariantOption<{scalar_type.cc_type()}>{{")
          with self.with_indent(4):
            self.print(".predicate = ")
            self.print_type_checker_name(scalar_type)
            self.println(",")

            self.print(".converter = ")
            self.print_converter(scalar_type, lang_name)
            self.println(",")
          self.print("}")
    self.print(")")

  def print_list_converter(
      self, list_type: ListType, lang_name: str
  ) -> None:
    # List<std::optional<std::unique_ptr<MyClass>>>(
    #     Nullable<std::unique_ptr<MyClass>>(MyClass::FromJson)
    # )
    element_type = list_type.element_type

    with self.with_vars({
        "nullable_element_cc_type": element_type.cc_type(
            maybe_null_to_optionalness(list_type.element_maybe_null)
        ),
        "element_cc_type": element_type.cc_type(),
    }):
      self.println("List<$nullable_element_cc_type$>(")
      with self.with_indent(4):
        if list_type.element_maybe_null == MaybeNull.YES:
          self.println("Nullable<$element_cc_type$>(")
          with self.with_indent(4):
            self.print_converter(element_type, lang_name)
            self.println()
          self.println(")")
        else:
          self.print_converter(element_type, lang_name)
          self.println()
    self.print(")")

  def print_from_json_function(self, node: NodeDef, lang_name: str) -> None:
    with self.with_vars({
        "NodeType": (Symbol(lang_name) + node.name).to_pascal_case(),
        "json_variable": JSON_VALUE_VARIABLE_NAME,
    }):
      self.println("absl::StatusOr<std::unique_ptr<$NodeType$>>")
      self.println(
          "$NodeType$::FromJson(const nlohmann::json& $json_variable$) {"
      )
      with self.with_indent():
        self.println(_CHECK_IS_OBJECT)
        self.println()

        if node.children:
          # This is a non-leaf type.
          # We get the `type` field and dispatch the corresponding
          # FromJson() function.
          self.println(
              "ABSL_ASSIGN_OR_RETURN(std::string type,"
              " GetType($json_variable$));"
          )
          self.println()

          if_stmt_printer = IfStmtPrinter(self)
          for descendant in node.descendants:

            def condition(descendant_name: str = descendant.name) -> None:
              with self.with_vars(
                  {"DescendentTypeNoLangName": descendant_name}
              ):
                self.print('type == "$DescendentTypeNoLangName$"')

            def body(
                descendant_type: str = (
                    Symbol(lang_name) + descendant.name
                ).to_pascal_case(),
            ) -> None:
              with self.with_vars({"DescendentType": descendant_type}):
                self.println(
                    "return $DescendentType$::FromJson($json_variable$);"
                )

            if_stmt_printer.print_case(
                IfStmtCase(condition=condition, body=body)
            )
          self.println()

          self.print("return absl::InvalidArgumentError")
          self.println('(absl::StrCat("Invalid type: ", type));')

        else:
          # This is a leaf type.
          # We get all the fields and call the constructor.
          node_field_pairs: list[tuple[str, Symbol]] = []
          for ancestor in node.ancestors:
            for field in ancestor.fields:
              node_field_pairs.append((ancestor.name, field.name))
          for field in node.fields:
            node_field_pairs.append((node.name, field.name))

          for pair_node_name, field_name in node_field_pairs:
            with self.with_vars({
                "NodeType": (
                    Symbol(lang_name) + pair_node_name
                ).to_pascal_case(),
                "field_name": field_name.to_cc_var_name(),
                "FieldName": field_name.to_pascal_case(),
            }):
              self.println(
                  "ABSL_ASSIGN_OR_RETURN(auto $field_name$, "
                  "$NodeType$::Get$FieldName$($json_variable$));"
              )

          self.println()

          # Call the constructor.
          self.print("return absl::make_unique<$NodeType$>(\n")
          with self.with_indent(4):
            with TabPrinter(
                TabPrinterOptions(print_separator=lambda: self.print(",\n"))
            ) as tab_printer:
              for field in node.aggregated_fields:
                with self.with_vars(
                    {"field_name": field.name.to_cc_var_name()}
                ):
                  tab_printer.print()
                  self.print("std::move($field_name$)")

          self.println(");")
      self.println("}")


def print_ast_from_json(ast: AstDef, cc_namespace: str, ast_path: str) -> str:
  printer = AstFromJsonPrinter()
  printer.print_ast(ast, cc_namespace, ast_path)
  return printer.content()
