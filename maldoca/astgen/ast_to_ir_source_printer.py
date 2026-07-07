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
"""Port of maldoca/astgen/ast_to_ir_source_printer.{h,cc} to Python.

Prints "conversion/ast_to_<lang>ir.generated.cc": the C++ visitor code that
walks AST node objects and builds the corresponding MLIR ops/values/
attributes (AST -> IR lowering).
"""

from __future__ import annotations

import enum

from maldoca.astgen import ast_def_pb2
from maldoca.astgen.ast_def import AstDef
from maldoca.astgen.ast_def import FieldDef
from maldoca.astgen.ast_def import NodeDef
from maldoca.astgen.ast_gen_utils import field_is_argument
from maldoca.astgen.ast_gen_utils import field_is_region
from maldoca.astgen.ast_gen_utils import get_ast_header_path
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


class Action(enum.Enum):
  """What to do with the converted IR value/attribute.

  - DEF: Define a variable.
  - ASSIGN: Assign the value/attribute to an existing variable.
  - CREATE: Just create the value/attribute and ignore it.
  """

  DEF = "def"
  ASSIGN = "assign"
  CREATE = "create"


class RefOrVal(enum.Enum):
  """Whether a C++ expression refers to a "reference" or a "value".

  Consider the following AST node:
  class CallExpression : ... {
   public:
    const Expression *func() const;
    const std::vector<std::unique_ptr<Expression>> *args() const;
  };

  - The type of func() is "const Expression *". We consider this a
    "reference".
  - The type of args()[0] is "std::unique_ptr<Expression> &". We consider
    this a "value".

  However, in the ASTGen type system, we refer them both as
  ClassType{"Expression"}. Therefore, we need this additional enum to make
  the distinction.

  If a function takes a "reference" but we have a "value", we need to call
  ".get()" to turn it into a "reference".
  """

  REF = "ref"
  VAL = "val"


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


# Gets the name of the *RegionEndOp.
# - For an lval or rval (expression): <Ir>ExprRegionEndOp.
# - For a list of lvals or rvals (expressions): <Ir>ExprsRegionEndOp.
def _get_region_end_op(ast: AstDef, field: FieldDef) -> Symbol:
  ir_name = Symbol(f"{ast.lang_name}ir")

  if field.kind == ast_def_pb2.FIELD_KIND_UNSPECIFIED:
    raise ValueError("Unspecified FieldKind.")
  elif field.kind == ast_def_pb2.FIELD_KIND_ATTR:
    raise ValueError(f"Unsupported FieldKind: {field.kind}")
  elif field.kind in (ast_def_pb2.FIELD_KIND_RVAL, ast_def_pb2.FIELD_KIND_LVAL):
    if isinstance(field.type, ListType):
      return ir_name + "ExprsRegionEndOp"
    else:
      return ir_name + "ExprRegionEndOp"
  elif field.kind == ast_def_pb2.FIELD_KIND_STMT:
    return Symbol()
  raise ValueError(f"Invalid FieldKind: {field.kind}")


class AstToIrSourcePrinter(CcPrinterBase):
  """Printer of the AST -> IR lowering visitor code."""

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
        f"{ir_path}/conversion/ast_to_{ast.lang_name}ir.h"
    )
    self.println()

    self.println("#include <memory>")
    self.println("#include <utility>")
    self.println("#include <vector>")
    self.println()

    self.print_include_headers([
        "llvm/ADT/APFloat.h",
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
        "absl/types/optional.h",
        "absl/types/variant.h",
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

  # Prints the Visit<OpName>() function.
  def print_non_leaf_node(
      self, ast: AstDef, node: NodeDef, kind: FieldKind
  ) -> None:
    ir_op_name = node.ir_op_name(ast.lang_name, kind)
    if ir_op_name is not None:
      return_type = ir_op_name.to_pascal_case()
    elif kind == ast_def_pb2.FIELD_KIND_UNSPECIFIED:
      raise ValueError("Invalid FieldKind: FIELD_KIND_UNSPECIFIED.")
    elif kind == ast_def_pb2.FIELD_KIND_ATTR:
      return_type = "mlir::Attribute"
    elif kind in (ast_def_pb2.FIELD_KIND_LVAL, ast_def_pb2.FIELD_KIND_RVAL):
      return_type = "mlir::Value"
    elif kind == ast_def_pb2.FIELD_KIND_STMT:
      return_type = "mlir::Operation*"
    else:
      raise ValueError(f"Invalid FieldKind: {kind}")

    ir_name = Symbol(f"{ast.lang_name}ir")
    visitor = _get_visitor(node, kind)

    with self.with_vars({
        "Ret": return_type,
        "Name": (Symbol(ast.lang_name) + node.name).to_pascal_case(),
        "IrName": ir_name.to_pascal_case(),
        "Visitor": visitor.to_pascal_case(),
    }):
      self.println(
          "$Ret$ AstTo$IrName$::$Visitor$(mlir::OpBuilder &builder, const"
          " $Name$ *node) {"
      )
      with self.with_indent():
        for leaf in node.leaves:
          with self.with_vars({
              "LeafName": (
                  Symbol(ast.lang_name) + leaf.name
              ).to_pascal_case(),
              "leaf_name": Symbol(leaf.name).to_cc_var_name(),
              "LeafVisitor": _get_visitor(leaf, kind).to_pascal_case(),
          }):
            self.println(
                "if (auto *$leaf_name$ = dynamic_cast<const $LeafName$"
                " *>(node)) {"
            )
            self.println("  return $LeafVisitor$(builder, $leaf_name$);")
            self.println("}")

        self.println('LOG(FATAL) << "Unreachable code.";')
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

    creator = Symbol("Create")
    if kind in (
        ast_def_pb2.FIELD_KIND_UNSPECIFIED,
        ast_def_pb2.FIELD_KIND_ATTR,
    ):
      raise ValueError(f"Unsupported kind: {kind}")
    elif kind in (ast_def_pb2.FIELD_KIND_LVAL, ast_def_pb2.FIELD_KIND_RVAL):
      creator += "Expr"
    elif kind == ast_def_pb2.FIELD_KIND_STMT:
      creator += "Stmt"
    else:
      raise ValueError(f"Invalid FieldKind: {kind}")

    with self.with_vars({
        "OpName": ir_op_name.to_pascal_case(),
        "Name": (Symbol(ast.lang_name) + node.name).to_pascal_case(),
        "IrName": ir_name.to_pascal_case(),
        "Visitor": visitor.to_pascal_case(),
        "Creator": creator.to_pascal_case(),
    }):
      self.println(
          "$OpName$ AstTo$IrName$::$Visitor$(mlir::OpBuilder &builder, const"
          " $Name$ *node) {"
      )
      with self.with_indent():
        for field in node.aggregated_fields:
          if field_is_argument(field):
            self.print_field(ast, node, field)

        has_regions = any(
            field_is_region(f) for f in node.aggregated_fields
        )
        if has_regions:
          self.print("auto op = ")
        else:
          self.print("return ")

        self.print("$Creator$<$OpName$>(builder, node")
        with self.with_indent(4):
          for field in node.aggregated_fields:
            if not field_is_argument(field):
              continue

            mlir_field_name = Symbol("mlir") + field.name
            with self.with_vars(
                {"mlir_field_name": mlir_field_name.to_cc_var_name()}
            ):
              self.print(", $mlir_field_name$")
        self.println(");")

        if has_regions:
          for field in node.aggregated_fields:
            if field_is_region(field):
              self.print_region(ast, node, field)

          self.println("return op;")

      self.println("}")
      self.println()

  # Prints the code that converts an AST field to an MLIR value/attribute
  # and stores the result in a new variable.
  #
  # Format:
  #
  # <TdType> mlir_<field_name> = Visit<Type>(node-><field_name>());
  #
  # Example:
  #
  # mlir::Value mlir_object = VisitExpression(node->object());
  def print_field(self, ast: AstDef, node: NodeDef, field: FieldDef) -> None:
    del node  # Unused; matches the (also-unused) C++ parameter.
    maybe_null = _optionalness_to_maybe_null(field.optionalness)

    lhs = Symbol("mlir") + field.name
    rhs = f"node->{field.name.to_cc_var_name()}()"
    self.print_nullable_to_ir(
        ast, Action.DEF, field.type, maybe_null, RefOrVal.REF, field.kind,
        lhs, rhs,
    )

  # Prints the code that converts an AST field to a region. The region has
  # been created and the code just populates blocks and ops in it.
  #
  # Format:
  #
  # mlir::Region &mlir_<field_name>_region = op.<field_name>();
  # AppendNewBlockAndPopulate(mlir_<field_name>_region, [&] {
  #   <Converts node->foo() into elements in the region.>
  # });
  #
  # Example:
  #
  # mlir::Region &mlir_body_region = op.body();
  # AppendNewBlockAndPopulate(mlir_body_region, [&] {
  #   for (const auto &element : *node->body()) {
  #     VisitStatement(element.get());
  #   }
  # });
  def print_region(self, ast: AstDef, node: NodeDef, field: FieldDef) -> None:
    del node  # Unused; matches the (also-unused) C++ parameter.
    maybe_null = _optionalness_to_maybe_null(field.optionalness)

    lhs = Symbol("mlir") + field.name
    lhs_region = lhs + "region"
    rhs = f"node->{field.name.to_cc_var_name()}()"

    with self.with_vars({
        "lhs": lhs.to_cc_var_name(),
        "lhs_region": lhs_region.to_cc_var_name(),
        "mlirGetter": field.name.to_mlir_getter(),
        "rhs": rhs,
    }):

      def populate_region() -> None:
        self.println("mlir::Region &$lhs_region$ = op.$mlirGetter$();")
        self.println(
            "AppendNewBlockAndPopulate(builder, $lhs_region$, [&] {"
        )
        with self.with_indent():
          if field.kind == ast_def_pb2.FIELD_KIND_UNSPECIFIED:
            raise ValueError("Unspecified FieldKind.")
          elif field.kind == ast_def_pb2.FIELD_KIND_ATTR:
            raise ValueError(f"Unsupported FieldKind: {field.kind}")
          elif field.kind in (
              ast_def_pb2.FIELD_KIND_RVAL,
              ast_def_pb2.FIELD_KIND_LVAL,
          ):
            action = Action.DEF
          elif field.kind == ast_def_pb2.FIELD_KIND_STMT:
            action = Action.CREATE
          else:
            raise ValueError(f"Invalid FieldKind: {field.kind}")

          region_end_op = _get_region_end_op(ast, field)
          self.print_to_ir(
              ast, action, field.type, RefOrVal.REF, field.kind, lhs, rhs
          )

          with self.with_vars(
              {"RegionEndOp": region_end_op.to_pascal_case()}
          ):
            if action == Action.ASSIGN:
              raise ValueError("Unsupported Action: Assign.")
            elif action == Action.CREATE:
              pass
            elif action == Action.DEF:
              self.println(
                  "CreateStmt<$RegionEndOp$>(builder, nullptr, $lhs$);"
              )
        self.println("});")

      if maybe_null == MaybeNull.YES:
        self.println("if ($rhs$.has_value()) {")
        with self.with_indent():
          rhs = f"{rhs}.value()"
          with self.with_vars({"rhs": rhs}):
            populate_region()
        self.println("}")
      else:
        populate_region()

  # ===========================================================================
  # Print*ToIr
  # ===========================================================================
  #
  # Prints the conversion of a C++ expression that represents a field from
  # the AST to the corresponding MLIR value/attribute. The result is later
  # used to build MLIR ops.
  # - rhs: The original C++ expression that represents a field from the AST.
  #
  # - lhs: The name of the variable to assign to or create, after the
  #        conversion.
  #
  # - action:
  #   - DEF:
  #     mlir::Value <lhs> = Convert(<rhs>);
  #   - ASSIGN:
  #     <lhs> = Convert(<rhs>);
  #   - CREATE:
  #     Convert(<rhs>);
  #
  # - type_: The type of the AST field.
  #
  # - ref_or_val: See the RefOrVal docstring.
  #
  # - kind: Kind of the field. See the FieldKind docstring.
  #   If kind == FIELD_KIND_LVAL, then we need to append "Ref" to the op
  #   name.
  def print_builtin_to_ir(
      self, ast: AstDef, action: Action, type_: BuiltinType, lhs: Symbol,
      rhs: str,
  ) -> None:
    del ast  # Unused; matches the (also-unused) C++ parameter.
    with self.with_vars({
        "mlir_type": type_.cc_mlir_builder_type(
            ast_def_pb2.FIELD_KIND_ATTR
        ),
        "lhs": lhs.to_cc_var_name(),
        "rhs": rhs,
    }):
      if action == Action.DEF:
        self.print("$mlir_type$ ")
        self.print("$lhs$ = ")
      elif action == Action.ASSIGN:
        self.print("$lhs$ = ")
      elif action == Action.CREATE:
        pass

      if type_.builtin_kind == BuiltinTypeKind.BOOL:
        self.print("builder.getBoolAttr($rhs$)")
      elif type_.builtin_kind == BuiltinTypeKind.INT64:
        self.print("builder.getI64IntegerAttr($rhs$)")
      elif type_.builtin_kind == BuiltinTypeKind.STRING:
        self.print("builder.getStringAttr($rhs$)")
      elif type_.builtin_kind == BuiltinTypeKind.DOUBLE:
        self.print("builder.getF64FloatAttr($rhs$)")

      self.println(";")

  def print_class_to_ir_ref(
      self, ast: AstDef, action: Action, type_: ClassType, kind: FieldKind,
      lhs: Symbol, rhs: str,
  ) -> None:
    del ast  # Unused; matches the (also-unused) C++ parameter.
    with self.with_vars({
        "ClassName": type_.name.to_pascal_case(),
        "lhs": lhs.to_cc_var_name(),
        "rhs": rhs,
    }):
      if action == Action.DEF:
        with self.with_vars(
            {"cc_mlir_type": type_.cc_mlir_builder_type(kind)}
        ):
          self.print("$cc_mlir_type$ ")
        self.print("$lhs$ = ")
      elif action == Action.ASSIGN:
        self.print("$lhs$ = ")
      elif action == Action.CREATE:
        pass

      if kind in (
          ast_def_pb2.FIELD_KIND_UNSPECIFIED,
          ast_def_pb2.FIELD_KIND_ATTR,
      ):
        self.println("Visit$ClassName$Attr(builder, $rhs$);")
      elif kind in (ast_def_pb2.FIELD_KIND_RVAL, ast_def_pb2.FIELD_KIND_STMT):
        self.println("Visit$ClassName$(builder, $rhs$);")
      elif kind == ast_def_pb2.FIELD_KIND_LVAL:
        self.println("Visit$ClassName$Ref(builder, $rhs$);")

  def print_class_to_ir(
      self, ast: AstDef, action: Action, type_: ClassType,
      ref_or_val: RefOrVal, kind: FieldKind, lhs: Symbol, rhs: str,
  ) -> None:
    if ref_or_val == RefOrVal.REF:
      self.print_class_to_ir_ref(ast, action, type_, kind, lhs, rhs)
    elif ref_or_val == RefOrVal.VAL:
      self.print_class_to_ir_ref(
          ast, action, type_, kind, lhs, f"{rhs}.get()"
      )

  def print_enum_to_ir(
      self, ast: AstDef, action: Action, type_: EnumType, lhs: Symbol,
      rhs: str,
  ) -> None:
    enum_name = (Symbol(ast.lang_name) + type_.name).to_pascal_case()
    rhs_str = f"{enum_name}ToString({rhs})"

    string_type = BuiltinType(BuiltinTypeKind.STRING, ast.lang_name)
    self.print_builtin_to_ir(ast, action, string_type, lhs, rhs_str)

  def print_variant_to_ir(
      self, ast: AstDef, action: Action, type_: VariantType,
      ref_or_val: RefOrVal, kind: FieldKind, lhs: Symbol, rhs: str,
  ) -> None:
    with self.with_vars({"lhs": lhs.to_cc_var_name(), "rhs": rhs}):
      if action == Action.DEF:
        with self.with_vars(
            {"cc_mlir_type": type_.cc_mlir_builder_type(kind)}
        ):
          self.println("$cc_mlir_type$ $lhs$;")
        case_action = Action.ASSIGN
      elif action == Action.ASSIGN:
        case_action = Action.ASSIGN
      else:
        case_action = Action.CREATE

      self.println("switch ($rhs$.index()) {")
      with self.with_indent():
        for i, scalar_type in enumerate(type_.types):
          with self.with_vars({"i": str(i)}):
            self.println("case $i$: {")
            with self.with_indent():
              self.print_to_ir(
                  ast, case_action, scalar_type, ref_or_val, kind, lhs,
                  f"std::get<{i}>({rhs})",
              )
              self.println("break;")
            self.println("}")

        self.println("default:")
        self.println('  LOG(FATAL) << "Unreachable code.";')
      self.println("}")

  def print_list_to_ir(
      self, ast: AstDef, action: Action, type_: ListType, kind: FieldKind,
      lhs: Symbol, rhs: str,
  ) -> None:
    lhs_element = Symbol("mlir_element")
    rhs_element = "element"

    with self.with_vars({
        "lhs": lhs.to_cc_var_name(),
        "lhs_data": (lhs + "data").to_cc_var_name(),
        "rhs": rhs,
        "lhs_element": lhs_element.to_cc_var_name(),
        "rhs_element": rhs_element,
    }):
      if kind == ast_def_pb2.FIELD_KIND_UNSPECIFIED:
        raise ValueError("FieldKind unspecified.")

      elif kind == ast_def_pb2.FIELD_KIND_STMT:
        # Case: List of Statements.
        assert action == Action.CREATE, (
            "We never collect statement ops in a vector."
        )

        self.println("for (const auto &$rhs_element$ : *$rhs$) {")
        with self.with_indent():
          self.print_nullable_to_ir(
              ast, Action.CREATE, type_.element_type,
              type_.element_maybe_null, RefOrVal.VAL, kind, lhs_element,
              rhs_element,
          )
        self.println("}")

      elif kind == ast_def_pb2.FIELD_KIND_ATTR:
        # Case: List of Attributes.
        #
        # We first create and fill a std::vector<mlir::Attribute> and then
        # convert it into a mlir::ArrayAttr (what the builder takes).
        self.println("std::vector<mlir::Attribute> $lhs_data$;")
        self.println("for (const auto &$rhs_element$ : *$rhs$) {")
        with self.with_indent():
          self.print_nullable_to_ir(
              ast, Action.DEF, type_.element_type, type_.element_maybe_null,
              RefOrVal.VAL, kind, lhs_element, rhs_element,
          )
          self.println("$lhs_data$.push_back(std::move($lhs_element$));")
        self.println("}")

        if action == Action.DEF:
          self.println("auto $lhs$ = builder.getArrayAttr($lhs_data$);")
        elif action == Action.ASSIGN:
          self.println("$lhs$ = builder.getArrayAttr($lhs_data$);")
        elif action == Action.CREATE:
          raise ValueError("We never put attributes in a region.")

      elif kind in (ast_def_pb2.FIELD_KIND_LVAL, ast_def_pb2.FIELD_KIND_RVAL):
        # Case: List of Values.
        #
        # We create and fill a std::vector<mlir::Value> which can be
        # implicitly converted to a mlir::ValueRange (what the builder
        # takes).
        if action == Action.DEF:
          self.println("std::vector<mlir::Value> $lhs$;")
        elif action == Action.ASSIGN:
          pass
        elif action == Action.CREATE:
          raise ValueError("We must put expressions in a vector.")

        self.println("for (const auto &$rhs_element$ : *$rhs$) {")
        with self.with_indent():
          if type_.element_maybe_null == MaybeNull.NO:
            self.print_to_ir(
                ast, Action.DEF, type_.element_type, RefOrVal.VAL, kind,
                lhs_element, rhs_element,
            )
          else:
            # Unfortunately, in the std::vector<mlir::Value> we can't have
            # any nullptr. In order to represent optional, we need the
            # special <Lang>irNoneOp.
            self.println("mlir::Value $lhs_element$;")
            self.println("if ($rhs_element$.has_value()) {")
            with self.with_indent():
              self.print_to_ir(
                  ast, Action.ASSIGN, type_.element_type, RefOrVal.VAL, kind,
                  lhs_element, f"{rhs_element}.value()",
              )
            self.println("} else {")
            with self.with_indent():
              none_op = Symbol(f"{ast.lang_name}ir") + "NoneOp"
              with self.with_vars({"NoneOp": none_op.to_pascal_case()}):
                self.println(
                    "$lhs_element$ = CreateExpr<$NoneOp$>(builder, node);"
                )
            self.println("}")

          self.println("$lhs$.push_back(std::move($lhs_element$));")

        self.println("}")
      else:
        raise ValueError(f"Invalid FieldKind: {kind}")

  def print_to_ir(
      self, ast: AstDef, action: Action, type_: Type, ref_or_val: RefOrVal,
      kind: FieldKind, lhs: Symbol, rhs: str,
  ) -> None:
    if isinstance(type_, BuiltinType):
      self.print_builtin_to_ir(ast, action, type_, lhs, rhs)
    elif isinstance(type_, ClassType):
      self.print_class_to_ir(ast, action, type_, ref_or_val, kind, lhs, rhs)
    elif isinstance(type_, EnumType):
      self.print_enum_to_ir(ast, action, type_, lhs, rhs)
    elif isinstance(type_, VariantType):
      self.print_variant_to_ir(
          ast, action, type_, ref_or_val, kind, lhs, rhs
      )
    elif isinstance(type_, ListType):
      assert ref_or_val == RefOrVal.REF
      self.print_list_to_ir(ast, action, type_, kind, lhs, rhs)

  def print_nullable_to_ir(
      self, ast: AstDef, action: Action, type_: Type, maybe_null: MaybeNull,
      ref_or_val: RefOrVal, kind: FieldKind, lhs: Symbol, rhs: str,
  ) -> None:
    with self.with_vars({"lhs": lhs.to_cc_var_name(), "rhs": rhs}):
      if maybe_null == MaybeNull.YES:
        if action == Action.ASSIGN:
          non_null_action = Action.ASSIGN
        elif action == Action.CREATE:
          non_null_action = Action.CREATE
        else:
          with self.with_vars(
              {"mlir_type": type_.cc_mlir_builder_type(kind)}
          ):
            self.println("$mlir_type$ $lhs$;")
          non_null_action = Action.ASSIGN

        self.println("if ($rhs$.has_value()) {")
        with self.with_indent():
          new_rhs = f"{rhs}.value()"
          self.print_to_ir(
              ast, non_null_action, type_, ref_or_val, kind, lhs, new_rhs
          )
        self.println("}")
      else:
        self.print_to_ir(ast, action, type_, ref_or_val, kind, lhs, rhs)


# Prints the "ast_to<lang_name>ir.generated.cc" file.
#
# - cc_namespace: The namespace where all IR op classes live.
#
# - ast_path: The directory for the AST code.
#
#   "ast.generated.h" is in that directory.
#
#   This is used to print the #includes.
#
# - ir_path: The directory for the IR code.
#
#   The following files are in that directory:
#   - "<lang_name>ir_dialect.td"
#   - "<lang_name>ir_ops.generated.td"
#   - "interfaces.td"
#   - "conversion/ast_to_<lang_name>ir.h"
#   - "conversion/ast_to_<lang_name>ir.generated.cc"
#
#   This is used to print the #includes and header guards.
def print_ast_to_ir_source(
    ast: AstDef, cc_namespace: str, ast_path: str, ir_path: str
) -> str:
  printer = AstToIrSourcePrinter()
  printer.print_ast(ast, cc_namespace, ast_path, ir_path)
  return printer.content()
