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
"""Port of maldoca/astgen/ir_to_ast_source_printer.{h,cc} to Python.

Prints "conversion/<lang>ir_to_ast.generated.cc": the C++ visitor code that
reconstructs AST node objects from MLIR ops/values/attributes (IR -> AST
raising). The mirror image of ast_to_ir_source_printer.py.
"""

from __future__ import annotations

from maldoca.astgen import ast_def_pb2
from maldoca.astgen.ast_def import AstDef
from maldoca.astgen.ast_def import FieldDef
from maldoca.astgen.ast_def import NodeDef
from maldoca.astgen.ast_gen_utils import field_is_argument
from maldoca.astgen.ast_gen_utils import field_is_region
from maldoca.astgen.ast_gen_utils import get_ast_header_path
from maldoca.astgen.ast_gen_utils import TabPrinter
from maldoca.astgen.ast_gen_utils import TabPrinterOptions
from maldoca.astgen.cc_printer_base import CcPrinterBase
from maldoca.astgen.symbol import Symbol
from maldoca.astgen.type import BuiltinType
from maldoca.astgen.type import BuiltinTypeKind
from maldoca.astgen.type import ClassType
from maldoca.astgen.type import EnumType
from maldoca.astgen.type import ListType
from maldoca.astgen.type import MaybeNull
from maldoca.astgen.type import Type
from maldoca.astgen.type import VariantType

FieldKind = ast_def_pb2.FieldKind


def _optionalness_to_maybe_null(optionalness: ast_def_pb2.Optionalness) -> MaybeNull:
  if optionalness in (
      ast_def_pb2.OPTIONALNESS_MAYBE_NULL,
      ast_def_pb2.OPTIONALNESS_MAYBE_UNDEFINED,
  ):
    return MaybeNull.YES
  return MaybeNull.NO


def _get_visitor(node: NodeDef, kind: FieldKind) -> Symbol:
  visitor = Symbol("Visit") + node.name
  if kind == ast_def_pb2.FIELD_KIND_ATTR:
    visitor += "Attr"
  if kind == ast_def_pb2.FIELD_KIND_LVAL:
    visitor += "Ref"
  return visitor


class IrToAstSourcePrinter(CcPrinterBase):
  """Printer of the IR -> AST raising visitor code."""

  def print_ast(
      self, ast: AstDef, cc_namespace: str, ast_path: str, ir_path: str
  ) -> None:
    ast_header_path = get_ast_header_path(ast_path)

    self.print_license()
    self.println()

    self.print_code_generation_warning()
    self.println()

    self.println("// IWYU pragma: begin_keep")
    self.println("// NOLINTBEGIN(whitespace/line_length)")
    self.println("// clang-format off")
    self.println()

    self.print_include_header(
        f"{ir_path}/conversion/{ast.lang_name}ir_to_ast.h"
    )
    self.println()

    self.println("#include <memory>")
    self.println("#include <optional>")
    self.println("#include <string>")
    self.println("#include <utility>")
    self.println("#include <variant>")
    self.println("#include <vector>")
    self.println()

    self.print_include_headers([
        "llvm/ADT/APFloat.h",
        "llvm/ADT/TypeSwitch.h",
        "llvm/Support/Casting.h",
        "mlir/IR/Attributes.h",
        "mlir/IR/Block.h",
        "mlir/IR/Builders.h",
        "mlir/IR/BuiltinAttributes.h",
        "mlir/IR/BuiltinTypes.h",
        "mlir/IR/Operation.h",
        "mlir/IR/Region.h",
        "mlir/IR/Value.h",
        "absl/cleanup/cleanup.h",
        "absl/log/check.h",
        "absl/log/log.h",
        "absl/status/status.h",
        "absl/status/status_macros.h",
        "absl/status/statusor.h",
        "absl/strings/str_cat.h",
        "absl/types/optional.h",
        "absl/types/variant.h",
        "maldoca/astgen/ir_to_ast_util.h",
        ast_header_path,
        f"{ir_path}/ir.h",
    ])
    self.println()

    self.print_enter_namespace(cc_namespace)
    self.println()

    for node in ast.topological_sorted_nodes:
      if node.children:
        for kind in node.aggregated_kinds:
          self.print_non_leaf_node(ast, node, kind)

      if not node.should_generate_ir_op:
        continue

      for kind in node.aggregated_kinds:
        self.print_leaf_node(ast, node, kind)

    self.println("// clang-format on")
    self.println("// NOLINTEND(whitespace/line_length)")
    self.println("// IWYU pragma: end_keep")
    self.println()

    self.print_exit_namespace(cc_namespace)

  # Prints the Visit<Node>() function.
  def print_non_leaf_node(
      self, ast: AstDef, node: NodeDef, kind: FieldKind
  ) -> None:
    ir_op_name = node.ir_op_name(ast.lang_name, kind)
    if ir_op_name is not None:
      input_type = ir_op_name.to_pascal_case()
    elif kind == ast_def_pb2.FIELD_KIND_UNSPECIFIED:
      raise ValueError("Invalid FieldKind: FIELD_KIND_UNSPECIFIED.")
    elif kind == ast_def_pb2.FIELD_KIND_ATTR:
      input_type = "mlir::Attribute"
    elif kind in (
        ast_def_pb2.FIELD_KIND_LVAL,
        ast_def_pb2.FIELD_KIND_RVAL,
        ast_def_pb2.FIELD_KIND_STMT,
    ):
      input_type = "mlir::Operation*"
    else:
      raise ValueError(f"Invalid FieldKind: {kind}")

    ir_name = Symbol(f"{ast.lang_name}ir")
    visitor = _get_visitor(node, kind)

    with self.with_vars({
        "InputType": input_type,
        "BaseName": (
            "mlir::Attribute"
            if kind == ast_def_pb2.FIELD_KIND_ATTR
            else "mlir::Operation*"
        ),
        "Name": (Symbol(ast.lang_name) + node.name).to_pascal_case(),
        "name": "attr" if kind == ast_def_pb2.FIELD_KIND_ATTR else "op",
        "IrName": ir_name.to_pascal_case(),
        "Visitor": visitor.to_pascal_case(),
    }):
      self.println("absl::StatusOr<std::unique_ptr<$Name$>>")
      self.println("$IrName$ToAst::$Visitor$($InputType$ $name$) {")
      with self.with_indent():
        self.println(
            "using Ret = absl::StatusOr<std::unique_ptr<$Name$>>;"
        )
        self.println("return llvm::TypeSwitch<$BaseName$, Ret>($name$)")
        with self.with_indent():
          for leaf in node.leaves:
            leaf_op_name = leaf.ir_op_name(ast.lang_name, kind)
            assert leaf_op_name is not None
            with self.with_vars({
                "LeafOpName": leaf_op_name.to_pascal_case(),
                "LeafVisitor": _get_visitor(leaf, kind).to_pascal_case(),
            }):
              self.println(".Case([&]($LeafOpName$ $name$) {")
              self.println("  return $LeafVisitor$($name$);")
              self.println("})")

          self.println(".Default([&]($BaseName$ op) {")
          self.println(
              '  return absl::InvalidArgumentError("Unrecognized op");'
          )
          self.println("});")
      self.println("}")
      self.println()

  def print_leaf_node(
      self, ast: AstDef, node: NodeDef, kind: FieldKind
  ) -> None:
    ir_op_name = node.ir_op_name(ast.lang_name, kind)
    assert ir_op_name is not None
    ir_name = Symbol(f"{ast.lang_name}ir")

    visitor = Symbol("Visit") + node.name
    if kind == ast_def_pb2.FIELD_KIND_LVAL:
      visitor += "Ref"

    with self.with_vars({
        "OpName": ir_op_name.to_pascal_case(),
        "Name": (Symbol(ast.lang_name) + node.name).to_pascal_case(),
        "name": "attr" if kind == ast_def_pb2.FIELD_KIND_ATTR else "op",
        "IrName": ir_name.to_pascal_case(),
        "Visitor": visitor.to_pascal_case(),
    }):
      self.println("absl::StatusOr<std::unique_ptr<$Name$>>")
      self.println("$IrName$ToAst::$Visitor$($OpName$ $name$) {")
      with self.with_indent():
        for field in node.aggregated_fields:
          if field_is_argument(field):
            self.print_field(ast, node, field)
          elif field_is_region(field):
            self.print_region(ast, node, field)

        # Call the constructor.
        self.print("return Create<$Name$>(\n")
        with self.with_indent(4):
          self.print("$name$")

          for field in node.aggregated_fields:
            if not field_is_argument(field) and not field_is_region(field):
              continue

            with self.with_vars(
                {"field_name": field.name.to_cc_var_name()}
            ):
              self.print(",\nstd::move($field_name$)")

        self.println(");")
      self.println("}")
      self.println()

  def print_field(self, ast: AstDef, node: NodeDef, field: FieldDef) -> None:
    del node  # Unused; matches the (also-unused) C++ parameter.
    maybe_null = _optionalness_to_maybe_null(field.optionalness)

    mlir_getter = field.name.to_mlir_getter()

    if field.kind == ast_def_pb2.FIELD_KIND_UNSPECIFIED:
      raise ValueError("Unspecified FieldKind.")
    elif field.kind == ast_def_pb2.FIELD_KIND_ATTR:
      rhs = f"op.{mlir_getter}Attr()"
    elif field.kind in (ast_def_pb2.FIELD_KIND_LVAL, ast_def_pb2.FIELD_KIND_RVAL):
      rhs = f"op.{mlir_getter}()"
    elif field.kind == ast_def_pb2.FIELD_KIND_STMT:
      raise ValueError("Unsupported FieldKind.")
    else:
      raise ValueError(f"Invalid FieldKind: {field.kind}")

    with self.with_vars({"lhs": field.name.to_cc_var_name(), "rhs": rhs}):
      self.println("ABSL_ASSIGN_OR_RETURN(")
      with self.with_indent(4):
        self.println("auto $lhs$,")
        self.print("Convert(\n")
        with self.with_indent(4):
          self.println("$rhs$,")
          self.print_converter(
              ast, field.type, ast.lang_name, field.kind, maybe_null
          )
          self.println()
        self.println(")")
      self.println(");")

  def print_region(self, ast: AstDef, node: NodeDef, field: FieldDef) -> None:
    del node  # Unused; matches the (also-unused) C++ parameter.
    maybe_null = _optionalness_to_maybe_null(field.optionalness)

    if field.kind == ast_def_pb2.FIELD_KIND_UNSPECIFIED:
      raise ValueError("Unspecified FieldKind.")
    elif field.kind == ast_def_pb2.FIELD_KIND_ATTR:
      raise ValueError(f"Unsupported FieldKind: {field.kind}")
    elif field.kind in (ast_def_pb2.FIELD_KIND_RVAL, ast_def_pb2.FIELD_KIND_LVAL):
      if isinstance(field.type, ListType):
        end_op = Symbol(f"{ast.lang_name}ir") + "ExprsRegionEndOp"
        converter_type = f"ExprsRegion<{end_op.to_pascal_case()}>"
      else:
        end_op = Symbol(f"{ast.lang_name}ir") + "ExprRegionEndOp"
        converter_type = f"ExprRegion<{end_op.to_pascal_case()}>"
    elif field.kind == ast_def_pb2.FIELD_KIND_STMT:
      converter_type = (
          "StmtsRegion" if isinstance(field.type, ListType) else "StmtRegion"
      )
    else:
      raise ValueError(f"Invalid FieldKind: {field.kind}")

    with self.with_vars({
        "ConverterType": converter_type,
        "lhs": field.name.to_cc_var_name(),
        "mlirGetter": field.name.to_mlir_getter(),
    }):
      self.println("ABSL_ASSIGN_OR_RETURN(")
      with self.with_indent(4):
        self.println("auto $lhs$,")
        self.print("Convert(\n")
        with self.with_indent(4):
          self.println("op.$mlirGetter$(),")

          if maybe_null == MaybeNull.YES:
            self.println("Nullable(")
            with self.with_indent(4):
              self.print("$ConverterType$(\n")
              with self.with_indent(4):
                self.print_converter(
                    ast, field.type, ast.lang_name, field.kind, MaybeNull.NO
                )
                self.println()
              self.println(")")
            self.println(")")
          else:
            self.print("$ConverterType$(\n")
            with self.with_indent(4):
              self.print_converter(
                  ast, field.type, ast.lang_name, field.kind, MaybeNull.NO
              )
              self.println()
            self.println(")")
        self.println(")")
      self.println(");")

  def print_converter(
      self, ast: AstDef, type_: Type, lang_name: str, kind: FieldKind,
      maybe_null: MaybeNull,
  ) -> None:
    if maybe_null == MaybeNull.YES:
      if kind in (ast_def_pb2.FIELD_KIND_LVAL, ast_def_pb2.FIELD_KIND_RVAL):
        none_op = Symbol(f"{ast.lang_name}ir") + "NoneOp"
        self.print(f"Nullable<{none_op.to_pascal_case()}>(\n")
      else:
        self.print("Nullable(\n")
      with self.with_indent(4):
        self.print_converter(ast, type_, lang_name, kind, MaybeNull.NO)
        self.println()
      self.print(")")
      return

    if isinstance(type_, ListType):
      self.print_list_converter(ast, type_, lang_name, kind)
    elif isinstance(type_, VariantType):
      self.print_variant_converter(ast, type_, lang_name, kind)
    elif isinstance(type_, ClassType):
      self.print_class_converter(type_, lang_name, kind)
    elif isinstance(type_, EnumType):
      self.print_enum_converter(type_, lang_name)
    elif isinstance(type_, BuiltinType):
      self.print_builtin_converter(type_, kind)

  def print_builtin_converter(
      self, builtin_type: BuiltinType, kind: FieldKind
  ) -> None:
    del kind  # Unused; matches the (also-unused) C++ parameter.
    if builtin_type.builtin_kind == BuiltinTypeKind.STRING:
      self.print("ToString()")
    elif builtin_type.builtin_kind == BuiltinTypeKind.BOOL:
      self.print("ToBool()")
    elif builtin_type.builtin_kind == BuiltinTypeKind.INT64:
      self.print("ToInt64()")
    elif builtin_type.builtin_kind == BuiltinTypeKind.DOUBLE:
      self.print("ToDouble()")

  def print_enum_converter(self, enum_type: EnumType, lang_name: str) -> None:
    enum_name = Symbol(lang_name) + enum_type.name
    with self.with_vars({
        "EnumName": enum_name.to_pascal_case(),
        "cc_type": enum_type.cc_type(),
    }):
      self.print("Enum<$cc_type$>(StringTo$EnumName$)")

  def print_class_converter(
      self, class_type: ClassType, lang_name: str, kind: FieldKind
  ) -> None:
    del lang_name  # Unused; matches the (also-unused) C++ parameter.
    visitor = Symbol("Visit") + class_type.name
    if kind == ast_def_pb2.FIELD_KIND_ATTR:
      visitor += "Attr"
    if kind == ast_def_pb2.FIELD_KIND_LVAL:
      visitor += "Ref"

    with self.with_vars({"Visitor": visitor.to_pascal_case()}):
      if kind == ast_def_pb2.FIELD_KIND_ATTR:
        self.print("ToAttrConverter($Visitor$)")
      else:
        self.print("ToOpConverter($Visitor$)")

  def print_variant_converter(
      self, ast: AstDef, variant_type: VariantType, lang_name: str,
      kind: FieldKind,
  ) -> None:
    if kind == ast_def_pb2.FIELD_KIND_ATTR:
      self.println("AttrVariant(")
    else:
      self.println("OpVariant(")
    with self.with_indent(4):
      with TabPrinter(
          TabPrinterOptions(print_separator=lambda: self.print(",\n"))
      ) as tab_printer:
        for scalar_type in variant_type.types:
          tab_printer.print()
          self.print_converter(ast, scalar_type, lang_name, kind, MaybeNull.NO)
      self.println()
    self.print(")")

  def print_list_converter(
      self, ast: AstDef, list_type: ListType, lang_name: str, kind: FieldKind
  ) -> None:
    self.println("List(")
    with self.with_indent(4):
      self.print_converter(
          ast, list_type.element_type, lang_name, kind,
          list_type.element_maybe_null,
      )
      self.println()
    self.print(")")


def print_ir_to_ast_source(
    ast: AstDef, cc_namespace: str, ast_path: str, ir_path: str
) -> str:
  printer = IrToAstSourcePrinter()
  printer.print_ast(ast, cc_namespace, ast_path, ir_path)
  return printer.content()
