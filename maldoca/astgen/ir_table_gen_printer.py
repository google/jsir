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
"""Port of maldoca/astgen/ir_table_gen_printer.{h,cc} to Python.

Prints the "<lang_name>ir_ops.generated.td" MLIR ODS (TableGen) file: one
`def <Op> : <Dialect>_Op<...>` per (node, FieldKind) pair, derived from each
node's fields (attrs vs values vs regions).
"""

from __future__ import annotations

from maldoca.astgen import ast_def_pb2
from maldoca.astgen.ast_def import AstDef
from maldoca.astgen.ast_def import FieldDef
from maldoca.astgen.ast_def import NodeDef
from maldoca.astgen.ast_gen_utils import field_is_argument
from maldoca.astgen.ast_gen_utils import field_is_region
from maldoca.astgen.ast_gen_utils import TabPrinter
from maldoca.astgen.ast_gen_utils import TabPrinterOptions
from maldoca.astgen.cc_printer_base import CcPrinterBase
from maldoca.astgen.symbol import Symbol
from maldoca.astgen.type import ListType

FieldKind = ast_def_pb2.FieldKind

_REGION_END_COMMENT = """\
// $ir$.*_region_end: An artificial op at the end of a region to collect
// expression-related values.
//
// Take $ir$.exprs_region_end as example:
// ======================================
//
// Consider the following function declaration:
// ```
// function foo(arg1, arg2 = defaultValue) {
//   ...
// }
// ```
//
// We lower it to the following IR (simplified):
// ```
// %0 = $ir$.identifier_ref {"foo"}
// $ir$.function_declaration(%0) (
//   // params
//   {
//     %1 = $ir$.identifier_ref {"a"}
//     %2 = $ir$.identifier_ref {"b"}
//     %3 = $ir$.identifier {"defaultValue"}
//     %4 = $ir$.assignment_pattern_ref(%2, %3)
//     $ir$.exprs_region_end(%1, %4)
//   },
//   // body
//   {
//     ...
//   }
// )
// ```
//
// We can see that:
//
// 1. We put the parameter-related ops in a region, instead of taking them as
//    normal arguments. In other words, we don't do this:
//
//    ```
//    %0 = $ir$.identifier_ref {"foo"}
//    %1 = $ir$.identifier_ref {"a"}
//    %2 = $ir$.identifier_ref {"b"}
//    %3 = $ir$.identifier {"defaultValue"}
//    %4 = $ir$.assignment_pattern_ref(%2, %3)
//    $ir$.function_declaration(%0, [%1, %4]) (
//      // body
//      {
//        ...
//      }
//    )
//    ```
//
//    The reason is that sometimes an argument might have a default value, and
//    the evaluation of that default value happens once for each function call
//    (i.e. it happens "within" the function). If we take the parameter as
//    normal argument, then %3 is only evaluated once - at function definition
//    time.
//
// 2. Even though the function has two parameters, we use 4 ops to represent
//    them. This is because some parameters are more complex and require more
//    than one op.
//
// 3. We use "$ir$.exprs_region_end" to list the "top-level" ops for the
//    parameters. In the example above, ops [%2, %3, %4] all represent the
//    parameter "b = defaultValue", but %4 is the top-level one. In other words,
//    %4 is the root of the tree [%2, %3, %4].
//
// 4. Strictly speaking, we don't really need "$ir$.exprs_region_end". The ops
//    within the "params" region form several trees, and we can figure out what
//    the roots are (a root is an op whose return value is not used by any other
//    op). So the use of "$ir$.exprs_region_end" is mostly for convenience."""

_EXPR_REGION_END = """\
def $Ir$ExprRegionEndOp : $Ir$_Op<"expr_region_end", [Terminator]> {
  let arguments = (ins
    AnyType: $$argument
  );
}"""

_EXPRS_REGION_END = """\
def $Ir$ExprsRegionEndOp : $Ir$_Op<"exprs_region_end", [Terminator]> {
  let arguments = (ins
    Variadic<AnyType>: $$arguments
  );
}"""

_MLIR_TRAIT_NAMES = {
    ast_def_pb2.MLIR_TRAIT_PURE: "Pure",
    ast_def_pb2.MLIR_TRAIT_ISOLATED_FROM_ABOVE: "IsolatedFromAbove",
}


class IrTableGenPrinter(CcPrinterBase):
  """Printer of the MLIR ODS (TableGen) file for the AST's IR ops."""

  def print_ast(self, ast: AstDef, ir_path: str) -> None:
    self.print_license()
    self.println()

    self.print_code_generation_warning()
    self.println()

    # E.g. lang_name == "js", then ir_name == "jsir".
    ir_name = f"{ast.lang_name}ir"

    # E.g. "<ir_path>/jsir_ops.generated.td".
    td_path = f"{ir_path}/{ir_name}_ops.generated.td"

    self.print_enter_header_guard(td_path)
    self.println()

    imports = [
        "mlir/Interfaces/ControlFlowInterfaces.td",
        "mlir/Interfaces/InferTypeOpInterface.td",
        "mlir/Interfaces/LoopLikeInterface.td",
        "mlir/Interfaces/SideEffectInterfaces.td",
        "mlir/IR/OpBase.td",
        "mlir/IR/SymbolInterfaces.td",
        f"{ir_path}/interfaces.td",
        f"{ir_path}/{ast.lang_name}ir_dialect.td",
        f"{ir_path}/{ast.lang_name}ir_types.td",
    ]
    for import_ in imports:
      self.println(f'include "{import_}"')
    self.println()

    has_expr_region = False
    has_exprs_region = False
    for node in ast.topological_sorted_nodes:
      for field in node.aggregated_fields:
        if not field.enclose_in_region:
          continue
        if field.kind not in (
            ast_def_pb2.FIELD_KIND_LVAL,
            ast_def_pb2.FIELD_KIND_RVAL,
        ):
          continue
        if isinstance(field.type, ListType):
          has_exprs_region = True
        else:
          has_expr_region = True

    if has_expr_region or has_exprs_region:
      ir = Symbol(f"{ast.lang_name}ir")

      with self.with_vars(
          {"ir": ir.to_snake_case(), "Ir": ir.to_pascal_case()}
      ):
        self.println(_REGION_END_COMMENT)

        if has_expr_region:
          self.println(_EXPR_REGION_END)
          self.println()

        if has_exprs_region:
          self.println(_EXPRS_REGION_END)
          self.println()

    for node in ast.topological_sorted_nodes:
      if not node.should_generate_ir_op:
        continue

      for kind in node.aggregated_kinds:
        self.print_node(ast, node, kind)

    self.print_exit_header_guard(td_path)

  # Example:
  #
  # def JsirWithStatementOp : Jsir_Op<
  #     "with_statement", [
  #         JsirStatementOpInterfaceTraits
  #     ]> {
  #   let arguments = (ins
  #     AnyType: $object
  #   );
  #
  #   let regions = (region
  #     AnyRegion: $body
  #   );
  # }
  def print_node(self, ast: AstDef, node: NodeDef, kind: FieldKind) -> None:
    op_name = node.ir_op_name(ast.lang_name, kind)
    assert op_name is not None
    op_mnemonic = node.ir_op_mnemonic(kind)
    assert op_mnemonic is not None

    with self.with_vars({
        "OpName": op_name.to_pascal_case(),
        "op_mnemonic": op_mnemonic.to_cc_var_name(),
        "IrName": Symbol(f"{ast.lang_name}ir").to_pascal_case(),
    }):
      traits: list[Symbol] = []
      for parent in node.parents:
        if kind not in parent.aggregated_kinds:
          continue
        parent_ir_op_name = parent.ir_op_name(ast.lang_name, kind)
        if parent_ir_op_name is None:
          continue
        traits.append(parent_ir_op_name + "Traits")

      # When there is more than one variadic operand, we must append the
      # AttrSizedOperandSegments trait. This is because MLIR internally
      # stores operands as a single array and without additional
      # information, it cannot attribute ranges of that array into the
      # corresponding variadic operands.
      #
      # MLIR doesn't allow universally adding AttrSizedOperandSegments -
      # only ops with more than one variadic operand are allowed.
      #
      # See: https://mlir.llvm.org/docs/OpDefinitions/#variadic-operands
      num_variadic_operands = 0
      for field in node.fields:
        if field.enclose_in_region:
          continue

        if field.kind == ast_def_pb2.FIELD_KIND_UNSPECIFIED:
          raise ValueError(
              f"{node.name}::{field.name.to_cc_var_name()}: FieldKind"
              " unspecified."
          )
        elif field.kind in (
            ast_def_pb2.FIELD_KIND_ATTR,
            ast_def_pb2.FIELD_KIND_STMT,
        ):
          pass
        elif field.kind in (
            ast_def_pb2.FIELD_KIND_LVAL,
            ast_def_pb2.FIELD_KIND_RVAL,
        ):
          if (
              isinstance(field.type, ListType)
              or field.optionalness == ast_def_pb2.OPTIONALNESS_MAYBE_NULL
              or field.optionalness
              == ast_def_pb2.OPTIONALNESS_MAYBE_UNDEFINED
          ):
            num_variadic_operands += 1
      if num_variadic_operands > 1:
        traits.append(Symbol("AttrSizedOperandSegments"))

      if any(field_is_region(f) for f in node.aggregated_fields):
        traits.append(Symbol("NoTerminator"))

      for mlir_trait in node.aggregated_additional_mlir_traits:
        if mlir_trait == ast_def_pb2.MLIR_TRAIT_INVALID:
          raise ValueError("Invalid MlirTrait.")
        traits.append(Symbol(_MLIR_TRAIT_NAMES[mlir_trait]))

      if not traits:
        self.println('def $OpName$ : $IrName$_Op<"$op_mnemonic$", []> {')
      else:
        # Example:
        # ```
        # def JsirBinaryExpressionOp : Jsir_Op<
        #     "binary_expression", [
        #         DeclareOpInterfaceMethods<JsirNodeOpInterface>,
        #         DeclareOpInterfaceMethods<JsirExpressionOpInterface>
        #     ]> {
        # ```
        self.print("def $OpName$ : $IrName$_Op<\n    \"$op_mnemonic$\", [\n")

        with self.with_indent(8):
          with TabPrinter(
              TabPrinterOptions(print_separator=lambda: self.print(",\n"))
          ) as tab_printer:
            for trait in traits:
              with self.with_vars({"Trait": trait.to_pascal_case()}):
                tab_printer.print()
                self.print("$Trait$")

        self.println("\n    ]> {")

      with self.with_indent():
        line_separator_printer = TabPrinter(
            TabPrinterOptions(print_separator=lambda: self.print("\n"))
        )
        if node.has_fold:
          line_separator_printer.print()
          self.println("let hasFolder = 1;")

        if any(field_is_argument(f) for f in node.aggregated_fields):
          line_separator_printer.print()

          self.println("let arguments = (ins")
          with self.with_indent():
            with TabPrinter(
                TabPrinterOptions(print_separator=lambda: self.print(",\n"))
            ) as separator_printer:
              for field in node.aggregated_fields:
                if not field_is_argument(field):
                  continue
                separator_printer.print()
                self.print_argument(ast, node, field)
          self.println()
          self.println(");")

        if any(field_is_region(f) for f in node.aggregated_fields):
          line_separator_printer.print()

          self.println("let regions = (region")
          with self.with_indent():
            with TabPrinter(
                TabPrinterOptions(print_separator=lambda: self.print(",\n"))
            ) as separator_printer:
              for field in node.aggregated_fields:
                if not field_is_region(field):
                  continue
                separator_printer.print()
                self.print_region(ast, node, field)
          self.println()
          self.println(");")

        # Only expressions have results.
        if kind in (
            ast_def_pb2.FIELD_KIND_LVAL,
            ast_def_pb2.FIELD_KIND_RVAL,
        ):
          line_separator_printer.print()

          self.println("let results = (outs")
          self.println("  $IrName$AnyType")
          self.println(");")

      self.println("}")
      self.println()

  # Prints an argument for an op in MLIR ODS.
  #
  # Format:
  #
  # <TdType>: $<name>
  #
  # See Type.td_type() for what the MLIR ODS type is for each Type.
  #
  # Example:
  #
  # AnyType: $object
  def print_argument(
      self, ast: AstDef, node: NodeDef, field: FieldDef
  ) -> None:
    del ast, node  # Unused; matches the (also-unused) C++ parameters.
    with self.with_vars({
        "type": field.type.td_type(field.kind, field.optionalness),
        "name": field.name.to_cc_var_name(),
    }):
      self.print("$type$: $$$name$")

  # Prints a region in an op in MLIR ODS.
  #
  # Format:
  #
  # AnyRegion: $<name>
  #
  # Example:
  #
  # AnyRegion: $body
  def print_region(self, ast: AstDef, node: NodeDef, field: FieldDef) -> None:
    del ast, node  # Unused; matches the (also-unused) C++ parameters.

    if field.kind == ast_def_pb2.FIELD_KIND_UNSPECIFIED:
      raise ValueError("FieldKind is unspecified.")
    elif field.kind == ast_def_pb2.FIELD_KIND_ATTR:
      raise ValueError("Region of attributes not supported.")
    elif field.kind in (
        ast_def_pb2.FIELD_KIND_LVAL,
        ast_def_pb2.FIELD_KIND_RVAL,
    ):
      region_type = "ExprsRegion" if isinstance(
          field.type, ListType
      ) else "ExprRegion"
    elif field.kind == ast_def_pb2.FIELD_KIND_STMT:
      region_type = "StmtsRegion" if isinstance(
          field.type, ListType
      ) else "StmtRegion"
    else:
      raise ValueError(f"Invalid FieldKind: {field.kind}")

    if field.optionalness == ast_def_pb2.OPTIONALNESS_UNSPECIFIED:
      raise ValueError("Optionalness unspecified.")
    elif field.optionalness == ast_def_pb2.OPTIONALNESS_REQUIRED:
      pass
    elif field.optionalness in (
        ast_def_pb2.OPTIONALNESS_MAYBE_NULL,
        ast_def_pb2.OPTIONALNESS_MAYBE_UNDEFINED,
    ):
      region_type = f"OptionalRegion<{region_type}>"

    with self.with_vars({
        "name": field.name.to_cc_var_name(),
        "RegionType": region_type,
    }):
      self.print("$RegionType$: $$$name$")


# Prints the "<lang_name>ir_ops.generated.td" TableGen file.
#
# - ir_path: The directory for the IR code.
#
#   The following files are in that directory:
#   - "<lang_name>ir_dialect.td"
#   - "<lang_name>ir_ops.generated.td"
#   - "interfaces.td"
#
#   This is used to print the includes and header guards.
def print_ir_table_gen(ast: AstDef, ir_path: str) -> str:
  printer = IrTableGenPrinter()
  printer.print_ast(ast, ir_path)
  return printer.content()
