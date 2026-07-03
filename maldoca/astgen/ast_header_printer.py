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
"""Port of maldoca/astgen/ast_header_printer.{h,cc} to Python.

Printer of the C++ header for the AST ("ast.generated.h").
"""

from __future__ import annotations

from maldoca.astgen.ast_def import AstDef
from maldoca.astgen.ast_def import EnumDef
from maldoca.astgen.ast_def import FieldDef
from maldoca.astgen.ast_def import NodeDef
from maldoca.astgen.ast_gen_utils import get_ast_header_path
from maldoca.astgen.ast_gen_utils import JSON_VALUE_VARIABLE_NAME
from maldoca.astgen.ast_gen_utils import OS_VALUE_VARIABLE_NAME
from maldoca.astgen.ast_gen_utils import TabPrinter
from maldoca.astgen.ast_gen_utils import TabPrinterOptions
from maldoca.astgen.cc_printer_base import CcPrinterBase
from maldoca.astgen.symbol import Symbol


class AstHeaderPrinter(CcPrinterBase):
  """Printer of the C++ header for the AST."""

  # Prints the "ast.generated.h" header file.
  #
  # - cc_namespace: The C++ namespace for all the AST node classes.
  #   Example: "maldoca::astgen".
  #
  # - ast_path: The directory for the AST code.
  #   "ast.generated.h" is in that directory.
  #   This is used to generate the header guard.
  #
  # See test cases in test/ for examples.
  def print_ast(self, ast: AstDef, cc_namespace: str, ast_path: str) -> None:
    header_path = get_ast_header_path(ast_path)

    self.print_license()
    self.println()

    self.print_code_generation_warning()
    self.println()

    self.print_enter_header_guard(header_path)
    self.println()

    self.println("// IWYU pragma: begin_keep")
    self.println("// NOLINTBEGIN(whitespace/line_length)")
    self.println("// clang-format off")
    self.println()

    self.println("#include <optional>")
    self.println("#include <string>")
    self.println("#include <variant>")
    self.println("#include <vector>")
    self.println()

    self.print_include_header("absl/status/statusor.h")
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
      self.println()

    self.println("// clang-format on")
    self.println("// NOLINTEND(whitespace/line_length)")
    self.println("// IWYU pragma: end_keep")
    self.println()

    self.print_exit_namespace(cc_namespace)
    self.println()

    self.print_exit_header_guard(header_path)

  # Prints the enum definition and the prototypes of string conversion
  # functions.
  #
  # Example:
  #  enum UnaryOperator {
  #    kMinus,
  #    ...
  #  };
  #
  #  absl::string_view UnaryOperatorToString(UnaryOperator unary_operator);
  #  absl::StatusOr<UnaryOperator> StringToUnaryOperator(absl::string_view s);
  def print_enum(self, enum_def: EnumDef, lang_name: str) -> None:
    with self.with_vars({
        "EnumName": (Symbol(lang_name) + enum_def.name).to_pascal_case(),
        "enum_name": enum_def.name.to_snake_case(),
    }):
      self.println("enum class $EnumName$ {")
      with self.with_indent():
        for member in enum_def.members:
          with self.with_vars(
              {"kMemberName": (Symbol("k") + member.name).to_camel_case()}
          ):
            self.println("$kMemberName$,")
      self.println("};")
      self.println()

      self.println(
          "absl::string_view $EnumName$ToString($EnumName$ $enum_name$);"
      )
      self.println(
          "absl::StatusOr<$EnumName$> StringTo$EnumName$(absl::string_view"
          " s);"
      )

  # Prints the class declaration for a node.
  #
  # See test cases in test/ for examples.
  def print_node(self, node: NodeDef, lang_name: str) -> None:
    with self.with_vars({
        "NodeType": (Symbol(lang_name) + node.name).to_pascal_case(),
        "json_variable": JSON_VALUE_VARIABLE_NAME,
        "os_variable": OS_VALUE_VARIABLE_NAME,
    }):
      if node.node_type_enum is not None:
        self.print_enum(node.node_type_enum, lang_name)
        self.println()

      self.print("class $NodeType$")
      if node.parents:
        self.print(" : ")
        with TabPrinter(
            TabPrinterOptions(print_separator=lambda: self.print(", "))
        ) as separator_printer:
          for parent in node.parents:
            with self.with_vars({
                "BaseType": (Symbol(lang_name) + parent.name).to_pascal_case()
            }):
              separator_printer.print()
              self.print("public virtual $BaseType$")
      self.println(" {")

      # Always print "public:" because the declaration of FromJson() always
      # exists.
      self.println(" public:")
      with self.with_indent():
        # Constructor
        if node.aggregated_fields:
          self.print_constructor(node, lang_name)
          self.println()

        # Destructor
        if not node.parents and node.children:
          self.println("virtual ~$NodeType$() = default;")
          self.println()

        # Get type enum.
        if node.node_type_enum is not None:
          node_type_enum_name = node.node_type_enum.name
          with self.with_vars({
              "NodeTypeEnum": (
                  Symbol(lang_name) + node_type_enum_name
              ).to_pascal_case(),
              "node_type_enum": node_type_enum_name.to_cc_var_name(),
          }):
            self.println(
                "virtual $NodeTypeEnum$ $node_type_enum$() const = 0;"
            )
            self.println()

        elif not node.children:
          for ancestor in node.ancestors:
            if ancestor.node_type_enum is None:
              continue

            root_type_enum_name = ancestor.node_type_enum.name
            with self.with_vars({
                "RootTypeEnum": (
                    Symbol(lang_name) + root_type_enum_name
                ).to_pascal_case(),
                "root_type_enum": root_type_enum_name.to_cc_var_name(),
                "NodeTypeNoLang": Symbol(node.name).to_pascal_case(),
            }):
              self.println(
                  "$RootTypeEnum$ $root_type_enum$() const override {"
              )
              self.println("  return $RootTypeEnum$::k$NodeTypeNoLang$;")
              self.println("}")
              self.println()

        # Serialize
        if not node.parents:
          if not node.children:
            # Non-virtual.
            self.println("void Serialize(std::ostream& $os_variable$) const;")
            self.println()
          else:
            # Virtual base.
            # We define a pure virtual function here, and override it in
            # leaf types.
            self.println(
                "virtual void Serialize(std::ostream& $os_variable$) const ="
                " 0;"
            )
            self.println()
        else:
          if not node.children:
            # Leaf type.
            # We override the virtual function.
            self.println(
                "void Serialize(std::ostream& $os_variable$) const"
                " override;"
            )
            self.println()
          # Non-leaf type - skipped.
          # We only override in leaf types. Here it's still pure virtual.

        # FromJson
        self.println(
            "static absl::StatusOr<std::unique_ptr<$NodeType$>> FromJson("
            "const nlohmann::json& $json_variable$);"
        )
        self.println()

        # Getters and setters.
        for field in node.fields:
          self.print_getter_setter_declarations(field, lang_name)
          self.println()

      self.println(" protected:")
      with self.with_indent():
        # SerializeFields
        self.println("// Internal function used by Serialize().")
        self.println("// Sets the fields defined in this class.")
        self.println("// Does not set fields defined in ancestors.")
        self.println(
            "void SerializeFields(std::ostream& $os_variable$, bool"
            " &needs_comma) const;"
        )

        # Get<FieldName>FromJson() functions.
        if node.fields:
          self.println()
          self.println("// Internal functions used by FromJson().")
          self.println("// Extracts a field from a JSON object.")
          for field in node.fields:
            self.print_get_from_json(field, lang_name)

      # Print member variables.
      if node.fields:
        self.println()
        self.println(" private:")
        with self.with_indent():
          for field in node.fields:
            self.print_member_variable(field, lang_name)

      self.println("};")

  # Prints the constructor of a node class.
  #
  # Example:
  #  explicit Variable(std::string identifier)
  #      : Expression(), identifier_(std::move(identifier)) {}
  def print_constructor(self, node: NodeDef, lang_name: str) -> None:
    with self.with_vars(
        {"NodeType": (Symbol(lang_name) + node.name).to_pascal_case()}
    ):
      self.print("explicit $NodeType$(")
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
      self.println(");")

  # Prints the getter and setter declarations for a field.
  #
  # Format:
  #  <cc_mutable_getter_type> <field_name>();
  #  <cc_const_getter_type> <field_name>() const;
  #  void set_<field_name>(<cc_type> <field_name>);
  #
  # - cc_mutable_getter_type: See `Type.cc_mutable_getter_type()`.
  # - cc_const_getter_type: See `Type.cc_const_getter_type()`.
  # - cc_type: See `Type.cc_type()`.
  #
  # Example:
  #  Expression* right();
  #  const Expression* right() const;
  #  void set_right(std::unique_ptr<Expression> right);
  def print_getter_setter_declarations(
      self, field: FieldDef, lang_name: str
  ) -> None:
    cc_getter_type = self.cc_mutable_getter_type(field)
    cc_const_getter_type = self.cc_const_getter_type(field)

    with self.with_vars({
        "cc_getter_type": cc_getter_type,
        "cc_const_getter_type": cc_const_getter_type,
        "cc_type": self.cc_type(field),
        "field_name": field.name.to_cc_var_name(),
    }):
      # If the mutable getter would return the same type as the const
      # getter, skip the mutable getter.
      if cc_getter_type != cc_const_getter_type:
        self.println("$cc_getter_type$ $field_name$();")
      self.println("$cc_const_getter_type$ $field_name$() const;")
      self.println("void set_$field_name$($cc_type$ $field_name$);")

  # Prints a member variable declaration.
  #
  # Format:
  #  <cc_type> <field_name>_;
  #
  # - cc_type: The C++ value type. See `Type.cc_type()`.
  # - field_name_: We print the name in snake_case and add a '_'.
  #
  # Example:
  #  std::unique_ptr<Expression> right_;
  def print_member_variable(self, field: FieldDef, lang_name: str) -> None:
    with self.with_vars({
        "cc_type": self.cc_type(field),
        "field_name": field.name.to_cc_var_name(),
    }):
      self.println("$cc_type$ $field_name$_;")

  # Format:
  #  static absl::StatusOr<<cc_type>>
  #  Get<FieldName>FromJson(const nlohmann::json& json);
  #
  # Example:
  #  static absl::StatusOr<std::unique_ptr<Expression>>
  #  GetRightFromJson(const nlohmann::json& json);
  def print_get_from_json(self, field: FieldDef, lang_name: str) -> None:
    with self.with_vars({
        "cc_type": self.cc_type(field),
        "FieldName": field.name.to_pascal_case(),
        "os_variable": OS_VALUE_VARIABLE_NAME,
    }):
      self.println(
          "static absl::StatusOr<$cc_type$> "
          "Get$FieldName$(const nlohmann::json& $json_variable$);"
      )


# Prints the "ast.generated.h" header file.
#
# - cc_namespace: The C++ namespace for all the AST node classes.
#   Example: "maldoca::astgen".
#
# - ast_path: The directory for the AST code.
#   "ast.generated.h" is in that directory.
#   This is used to generate the header guard.
def print_ast_header(ast: AstDef, cc_namespace: str, ast_path: str) -> str:
  printer = AstHeaderPrinter()
  printer.print_ast(ast, cc_namespace, ast_path)
  return printer.content()
