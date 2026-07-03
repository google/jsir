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
"""Port of maldoca/astgen/ast_def.{h,cc} to Python.

This is the semantic schema model built from an `AstDefPb`: `AstDef` (whole
schema), `NodeDef` (an AST node class, with resolved parents/ancestors/
children/descendants/leaves, aggregated fields, aggregated kinds, and a
topological ordering), `FieldDef`, `EnumDef`.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

from maldoca.astgen import ast_def_pb2
from maldoca.astgen import type as astgen_type
from maldoca.astgen.symbol import Symbol

FieldKind = ast_def_pb2.FieldKind
Optionalness = ast_def_pb2.Optionalness
MlirTrait = ast_def_pb2.MlirTrait


@dataclasses.dataclass(frozen=True)
class EnumMemberDef:
  name: Symbol
  string_value: str

  @classmethod
  def from_pb(cls, member_pb: ast_def_pb2.EnumMemberDefPb) -> "EnumMemberDef":
    name = Symbol(member_pb.name)
    if name.to_pascal_case() != member_pb.name:
      raise ValueError(
          f"The enum member name '{member_pb.name}' is not in PascalCase."
      )
    return cls(name, member_pb.string_value)


@dataclasses.dataclass(frozen=True)
class EnumDef:
  name: Symbol
  members: list[EnumMemberDef]

  @classmethod
  def from_pb(cls, enum_pb: ast_def_pb2.EnumDefPb) -> "EnumDef":
    name = Symbol(enum_pb.name)
    if name.to_pascal_case() != enum_pb.name:
      raise ValueError(
          f"The enum type name '{enum_pb.name}' is not in PascalCase."
      )
    members = [EnumMemberDef.from_pb(member_pb) for member_pb in enum_pb.members]
    return cls(name, members)


@dataclasses.dataclass
class FieldDef:
  """Definition of a field in a class."""

  name: Symbol
  optionalness: Optionalness
  type: astgen_type.Type
  kind: FieldKind
  ignore_in_ir: bool
  enclose_in_region: bool

  @classmethod
  def from_pb(
      cls, field_pb: ast_def_pb2.FieldDefPb, lang_name: str
  ) -> "FieldDef":
    name = Symbol(field_pb.name)

    # Check that the name is in camelCase.
    if name.to_camel_case() != field_pb.name:
      raise ValueError(f"Field '{field_pb.name}' is not in camelCase.")

    type_ = astgen_type.from_type_pb(field_pb.type, lang_name)

    if field_pb.optionalness == ast_def_pb2.OPTIONALNESS_UNSPECIFIED:
      raise ValueError(
          f"Field '{field_pb.name}' has OPTIONALNESS_UNSPECIFIED. This "
          "should be a bug, as the default value is already "
          "OPTIONALNESS_REQUIRED."
      )

    return cls(
        name=name,
        optionalness=field_pb.optionalness,
        type=type_,
        kind=field_pb.kind,
        ignore_in_ir=field_pb.ignore_in_ir,
        enclose_in_region=field_pb.enclose_in_region,
    )


class NodeDef:
  """Definition of an AST node type. Corresponds to a C++ class.

  Only `AstDef.from_proto()` constructs (and fills in the graph-derived
  fields of) a `NodeDef`.
  """

  def __init__(self):
    self.name: str = ""

    # In the JavaScript object version of the AST, a special "type" string
    # represents the kind of the node. The existence of a concrete "type"
    # value suggests that this is a leaf type. See ast_def.proto.
    self.type: Optional[str] = None

    # Fields defined by this node. Not including fields in parents.
    self.fields: list[FieldDef] = []

    # The classes that this derives from.
    self.parents: list["NodeDef"] = []

    # Topologically sorted: base comes before derived. Use the original
    # definition order to break tie.
    self.ancestors: list["NodeDef"] = []

    # All fields, including those defined by ancestors.
    self.aggregated_fields: list[FieldDef] = []

    # Direct children of this class.
    self.children: list["NodeDef"] = []

    # All types that directly or indirectly inherit this class.
    self.descendants: list["NodeDef"] = []

    # All descendants that are leaf classes.
    self.leaves: list["NodeDef"] = []

    self.node_type_enum: Optional[EnumDef] = None

    # Whether an IR op should be automatically generated. If false, the op
    # is expected to be manually written.
    self.should_generate_ir_op: bool = False

    # The allowed FieldKinds for this node. Does not include those specified
    # in ancestors.
    self.kinds: list[FieldKind] = []

    # The allowed FieldKinds for this node, including those in ancestors.
    self.aggregated_kinds: list[FieldKind] = []

    # Deprecated: no longer used for dialect splitting (HIR merged into IR).
    self.has_control_flow: bool = False

    # [Optional] Custom MLIR op name. Was `NodeDef::ir_op_name_` in C++; kept
    # under a different Python name so it doesn't collide with the
    # `ir_op_name()` method below (C++ can overload on this, Python can't).
    self.custom_ir_op_name: Optional[str] = None

    self.has_fold: bool = False

    # Additional MLIR traits to add to the op definition in ODS.
    self.additional_mlir_traits: list[MlirTrait] = []

    # Additional MLIR traits to add to the op definition in ODS, including
    # those from ancestors.
    self.aggregated_additional_mlir_traits: list[MlirTrait] = []

  # The MLIR op name (C++ class name).
  #
  # <IrName>: <LangName>ir
  #
  # - Non-leaf type: "<IrName><ClassName>OpInterface"
  # - Leaf type:
  #   - RVal:        "<IrName><ClassName>Op"
  #   - LVal:        "<IrName><ClassName>RefOp"
  #
  # If a custom IR op name is specified (NodeDefPb.ir_op_name), returns
  # that instead.
  #
  # If a custom IR op name is specified for any of the descendants, returns
  # None.
  def ir_op_name(self, lang_name: str, kind: FieldKind) -> Optional[Symbol]:
    if self.custom_ir_op_name is not None:
      return Symbol(self.custom_ir_op_name)

    if any(d.custom_ir_op_name is not None for d in self.descendants):
      return None

    ir_name = lang_name + "ir"
    result = Symbol(ir_name)
    result += self.name

    if kind == ast_def_pb2.FIELD_KIND_UNSPECIFIED:
      raise ValueError("Invalid FieldKind.")
    elif kind == ast_def_pb2.FIELD_KIND_ATTR:
      result += "Attr"
    elif kind in (ast_def_pb2.FIELD_KIND_RVAL, ast_def_pb2.FIELD_KIND_STMT):
      result += "Op"
    elif kind == ast_def_pb2.FIELD_KIND_LVAL:
      result += "RefOp"
    else:
      raise ValueError(f"Invalid FieldKind: {kind}")

    if self.children:
      result += "Interface"

    return result

  # The stringified MLIR op name (without dialect name).
  #
  # - Non-leaf type: N/A
  # - Leaf type:
  #   - RVal:        "<class_name>"
  #   - LVal:        "<class_name>_ref"
  #
  # If a custom IR op name is specified, returns None.
  #
  # If a custom IR op name is specified for any of the descendants, returns
  # None.
  def ir_op_mnemonic(self, kind: FieldKind) -> Optional[Symbol]:
    if self.custom_ir_op_name is not None:
      return None

    if any(d.custom_ir_op_name is not None for d in self.descendants):
      return None

    if kind == ast_def_pb2.FIELD_KIND_UNSPECIFIED:
      raise ValueError("Invalid FieldKind.")
    elif kind == ast_def_pb2.FIELD_KIND_ATTR:
      raise ValueError(f"Unsupported FieldKind: {kind}")
    elif kind == ast_def_pb2.FIELD_KIND_LVAL:
      return Symbol(self.name) + "ref"
    elif kind in (ast_def_pb2.FIELD_KIND_RVAL, ast_def_pb2.FIELD_KIND_STMT):
      return Symbol(self.name)
    raise ValueError(f"Invalid FieldKind: {kind}")


def _topological_sort_dependencies_visit(
    node: NodeDef,
    get_dependencies,
    pre_order_visited: set[NodeDef],
    sorted_dependencies: list[NodeDef],
) -> None:
  # We run a DFS to perform topological sort.
  #
  # We maintain two sets:
  # - sorted_dependencies: The result list being constructed.
  # - pre_order_visited: nodes in the recursion stack.
  #
  # Each node is inserted to `pre_order_visited` pre-order; moved to
  # `sorted_dependencies` post-order. If a node is already in
  # `sorted_dependencies`, skip it (typical DFS); if a node is already in
  # `pre_order_visited`, the graph has a cycle.
  for dependency in get_dependencies(node):
    if dependency in pre_order_visited:
      raise AssertionError("Graph has cycle!")
    if dependency in sorted_dependencies:
      continue

    pre_order_visited.add(dependency)
    _topological_sort_dependencies_visit(
        dependency, get_dependencies, pre_order_visited, sorted_dependencies
    )
    pre_order_visited.discard(dependency)
    sorted_dependencies.append(dependency)


def _topological_sort_dependencies(node: NodeDef, get_dependencies) -> list[NodeDef]:
  """Topologically sorts all the (transitive) dependencies of `node`.

  For example: (A <: B means A depends on B)

  Input graph:
    CatDog <: Cat, Dog
    Cat <: Animal
    Dog <: Animal

  _topological_sort_dependencies(CatDog, ...) => [Animal, Cat, Dog]

  Note: We use the original order of dependencies to break tie. For example,
  Cat appears before Dog and this is preserved.
  """
  pre_order_visited: set[NodeDef] = set()
  sorted_dependencies: list[NodeDef] = []
  _topological_sort_dependencies_visit(
      node, get_dependencies, pre_order_visited, sorted_dependencies
  )
  return sorted_dependencies


def _get_type_dependencies(
    type_: astgen_type.Type, nodes: dict[str, NodeDef]
) -> list[NodeDef]:
  """Gets the NodeDef dependencies of a given field Type.

  In the generated C++ code, these nodes must be defined before the type is
  used.
  """
  if isinstance(type_, astgen_type.ClassType):
    pascal_name = type_.name.to_pascal_case()
    node = nodes.get(pascal_name)
    if node is None:
      raise AssertionError(f"{pascal_name} undefined.")
    return [node]
  elif isinstance(type_, astgen_type.ListType):
    return _get_type_dependencies(type_.element_type, nodes)
  elif isinstance(type_, astgen_type.VariantType):
    dependencies = []
    for t in type_.types:
      dependencies.extend(_get_type_dependencies(t, nodes))
    return dependencies
  else:
    # BuiltinType, EnumType: no dependencies.
    return []


def _resolve_class_type(
    type_: astgen_type.Type, topological_sorted_nodes: list[NodeDef]
) -> None:
  """For each ClassType, if it resolves to a NodeDef, stores a reference to it."""
  if isinstance(type_, astgen_type.ClassType):
    for node in topological_sorted_nodes:
      if node.name == type_.name.to_pascal_case():
        type_.node_def = node
        break
  elif isinstance(type_, astgen_type.ListType):
    _resolve_class_type(type_.element_type, topological_sorted_nodes)
  elif isinstance(type_, astgen_type.VariantType):
    for t in type_.types:
      _resolve_class_type(t, topological_sorted_nodes)
  # BuiltinType, EnumType: nothing to resolve.


class AstDef:
  """Definition of an AST, built from an AstDefPb."""

  def __init__(
      self,
      lang_name: str,
      enum_defs: list[EnumDef],
      node_names: list[str],
      nodes: dict[str, NodeDef],
      topological_sorted_nodes: list[NodeDef],
  ):
    self.lang_name = lang_name
    self.enum_defs = enum_defs
    # Names of the nodes in the original order.
    self.node_names = node_names
    # Node name => node definition.
    self.nodes = nodes
    # Nodes listed in topological order: dependencies (parent classes, field
    # types) are defined before each class.
    self.topological_sorted_nodes = topological_sorted_nodes

  @classmethod
  def from_proto(cls, pb: ast_def_pb2.AstDefPb) -> "AstDef":
    """Creates an AST definition from a proto. Also validates the proto."""
    enum_defs = [EnumDef.from_pb(enum_def_pb) for enum_def_pb in pb.enums]

    node_names: list[str] = []
    nodes: dict[str, NodeDef] = {}

    for node_pb in pb.nodes:
      if node_pb.name in nodes:
        raise ValueError(f"{node_pb.name} already exists!")

      node = NodeDef()
      node.name = node_pb.name

      if node_pb.HasField("type"):
        node.type = node_pb.type

      node.fields = [
          FieldDef.from_pb(field_pb, pb.lang_name)
          for field_pb in node_pb.fields
      ]

      node.has_control_flow = node_pb.has_control_flow

      if node_pb.HasField("ir_op_name"):
        node.custom_ir_op_name = node_pb.ir_op_name

      node.should_generate_ir_op = node_pb.should_generate_ir_op
      node.has_fold = node_pb.has_fold
      node.kinds = list(node_pb.kinds)
      node.additional_mlir_traits = list(node_pb.additional_mlir_traits)

      node_names.append(node_pb.name)
      nodes[node_pb.name] = node

    # Set parent pointers.
    for node_pb in pb.nodes:
      node = nodes[node_pb.name]
      for parent_name in node_pb.parents:
        parent = nodes.get(parent_name)
        if parent is None:
          raise ValueError(f"Parent {parent_name} doesn't exist!")
        node.parents.append(parent)

    # For union types, create a node to represent each one and add that node
    # as a parent of the specified types.
    for union_type_pb in pb.union_types:
      union_type_node = NodeDef()
      union_type_node.name = union_type_pb.name
      union_type_node.should_generate_ir_op = union_type_pb.should_generate_ir_op
      union_type_node.kinds = list(union_type_pb.kinds)

      if union_type_pb.name in nodes:
        raise ValueError(f"{union_type_pb.name} already exists!")

      node_names.append(union_type_pb.name)
      nodes[union_type_pb.name] = union_type_node

    for union_type_pb in pb.union_types:
      union_type_node = nodes[union_type_pb.name]
      for member_name in union_type_pb.types:
        child_node = nodes.get(member_name)
        if child_node is None:
          raise ValueError(
              f"Union type {union_type_pb.name}: member {member_name} "
              "doesn't exist!"
          )
        child_node.parents.append(union_type_node)

      for parent_name in union_type_pb.parents:
        parent = nodes.get(parent_name)
        if parent is None:
          raise ValueError(f"Parent {parent_name} doesn't exist!")
        union_type_node.parents.append(parent)

    # NOTE: In the code below, we traverse `node_names` instead of `nodes`.
    # `node_names` preserves the original order of definitions. This makes
    # sure that the algorithm is always deterministic.

    # Set ancestors.
    for name in node_names:
      node = nodes[name]
      node.ancestors = _topological_sort_dependencies(
          node, lambda n: n.parents
      )

    # Set aggregated_fields.
    for name in node_names:
      node = nodes[name]
      for ancestor in node.ancestors:
        node.aggregated_fields.extend(ancestor.fields)
      node.aggregated_fields.extend(node.fields)

    # Set children.
    for name in node_names:
      node = nodes[name]
      for parent in node.parents:
        parent.children.append(node)

    # Set descendants.
    for name in node_names:
      node = nodes[name]
      node.descendants = _topological_sort_dependencies(
          node, lambda n: n.children
      )

    # Set leaves.
    for name in node_names:
      node = nodes[name]
      for descendant in node.descendants:
        if descendant.children:
          continue
        node.leaves.append(descendant)

    # Set aggregated_kinds.
    for name in node_names:
      node = nodes[name]
      aggregated_kinds: set[FieldKind] = set()
      for ancestor in node.ancestors:
        aggregated_kinds.update(ancestor.kinds)
      aggregated_kinds.update(node.kinds)
      node.aggregated_kinds = sorted(aggregated_kinds)

    # Set aggregated_additional_mlir_traits.
    for name in node_names:
      node = nodes[name]
      aggregated_traits: set[MlirTrait] = set()
      for ancestor in node.ancestors:
        aggregated_traits.update(ancestor.additional_mlir_traits)
      aggregated_traits.update(node.additional_mlir_traits)
      node.aggregated_additional_mlir_traits = sorted(aggregated_traits)

    # Reorder the node definitions so that dependencies always come first.
    topological_sorted_nodes: list[NodeDef] = []
    preorder_visited_nodes: set[NodeDef] = set()
    for name in node_names:
      node = nodes[name]

      def get_dependencies(n: NodeDef) -> list[NodeDef]:
        dependencies = list(n.parents)
        for field in n.fields:
          dependencies.extend(_get_type_dependencies(field.type, nodes))
        return dependencies

      _topological_sort_dependencies_visit(
          node, get_dependencies, preorder_visited_nodes,
          topological_sorted_nodes
      )
      if node not in topological_sorted_nodes:
        topological_sorted_nodes.append(node)

    # For each root node, add an enum field to represent the leaf type.
    for node in topological_sorted_nodes:
      if node.parents:
        continue
      if not node.children:
        continue

      type_enum_members = [
          EnumMemberDef(Symbol(leaf.name), leaf.name) for leaf in node.leaves
      ]
      node.node_type_enum = EnumDef(
          Symbol(node.name) + "Type", type_enum_members
      )

    # For each ClassType, if it resolves to a NodeDef, store a reference to
    # it.
    for node in topological_sorted_nodes:
      for field in node.fields:
        _resolve_class_type(field.type, topological_sorted_nodes)

    return cls(pb.lang_name, enum_defs, node_names, nodes, topological_sorted_nodes)
