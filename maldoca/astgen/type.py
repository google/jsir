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
"""Port of maldoca/astgen/type.{h,cc} to Python.

The Type Hierarchy
-------------------

Type        ::= NonListType, ListType
NonListType ::= ScalarType, VariantType
ScalarType  ::= BuiltinType, ClassType
BuiltinType ::= BoolType, DoubleType, StringType

                                 Type
                                   |
                       +-----------+-----------+
                       |                       |
                  NonListType                  |
                       |                       |
            +----------+----------+            |
            |                     |            |
        ScalarType                |            |
            |                     |            |
     +------+-------+             |            |
     |              |             |            |
 BuiltinType    ClassType    VariantType    ListType

Instead of porting the LLVM-style `IsA<T>()` RTTI helper from type.h, callers
should just use Python's built-in `isinstance()`.
"""

from __future__ import annotations

import abc
import enum

from maldoca.astgen import ast_def_pb2
from maldoca.astgen import type_pb2
from maldoca.astgen.symbol import Symbol

FieldKind = ast_def_pb2.FieldKind
Optionalness = ast_def_pb2.Optionalness

_MAYBE_NULL_OPTIONALNESS = frozenset({
    ast_def_pb2.OPTIONALNESS_MAYBE_NULL,
    ast_def_pb2.OPTIONALNESS_MAYBE_UNDEFINED,
})


class MaybeNull(enum.Enum):
  NO = "no"
  YES = "yes"


def maybe_null_to_optionalness(maybe_null: MaybeNull) -> Optionalness:
  if maybe_null == MaybeNull.YES:
    return ast_def_pb2.OPTIONALNESS_MAYBE_NULL
  return ast_def_pb2.OPTIONALNESS_REQUIRED


class TypeKind(enum.Enum):
  BUILTIN = "builtin"
  ENUM = "enum"
  CLASS = "class"
  VARIANT = "variant"
  LIST = "list"


class CcGetterKind(enum.Enum):
  MUTABLE = "mutable"
  CONST = "const"


class BuiltinTypeKind(enum.Enum):
  BOOL = "bool"
  INT64 = "int64"
  DOUBLE = "double"
  STRING = "string"


class Type(abc.ABC):
  """Base class of the field-type hierarchy. See module docstring."""

  def __init__(self, kind: TypeKind, lang_name: str):
    self.kind = kind
    self.lang_name = lang_name

  # ===========================================================================
  # JsType()
  # ===========================================================================

  # Prints TypeScript type annotations.
  #
  # Types that are maybe_null are printed as variants.
  # E.g. "bool"          with maybe_null=YES ==> "bool | null".
  # E.g. "bool | string" with maybe_null=YES ==> "bool | string | null".
  @abc.abstractmethod
  def js_type(self) -> str:
    ...

  def js_type_with_maybe_null(self, maybe_null: MaybeNull) -> str:
    s = self.js_type()
    if maybe_null == MaybeNull.YES:
      return s + " | null"
    return s

  # ===========================================================================
  # CcType()
  # ===========================================================================

  # Prints the C++ type for storing the field.
  #
  # Types that are maybe_null or maybe_undefined are printed as
  # "std::optional".
  #
  # bool                              => bool
  # double                            => double
  # string                            => std::string
  # ClassType                         => std::unique_ptr<ClassType>
  # Class1 | Class2                   => std::variant<std::unique_ptr<Class1>,
  #                                       std::unique_ptr<Class2>>
  # [ClassType]                       => std::vector<std::unique_ptr<
  #                                       ClassType>>
  # ClassType with maybe_null/undef.  => std::optional<std::unique_ptr<
  #                                       ClassType>>
  @abc.abstractmethod
  def _cc_type(self) -> str:
    ...

  def cc_type(
      self, optionalness: Optionalness = ast_def_pb2.OPTIONALNESS_UNSPECIFIED
  ) -> str:
    s = self._cc_type()
    if optionalness in _MAYBE_NULL_OPTIONALNESS:
      return f"std::optional<{s}>"
    return s

  # ===========================================================================
  # CcGetterType()
  # ===========================================================================

  # Common function that handles both the mutable and const getter types.
  @abc.abstractmethod
  def _cc_getter_type(self, getter_kind: CcGetterKind) -> str:
    ...

  def cc_getter_type(
      self,
      getter_kind: CcGetterKind,
      optionalness: Optionalness = ast_def_pb2.OPTIONALNESS_UNSPECIFIED,
  ) -> str:
    s = self._cc_getter_type(getter_kind)
    if optionalness in _MAYBE_NULL_OPTIONALNESS:
      return f"std::optional<{s}>"
    return s

  # Prints the C++ return type for the (mutable) getter function.
  #
  # bool                              => bool
  # double                            => double
  # string                            => std::string
  # ClassType                         => ClassType*
  # Class1 | Class2                   => std::variant<Class1*, Class2*>
  # [ClassType]                       => std::vector<std::unique_ptr<
  #                                       ClassType>>*
  # ClassType with maybe_null/undef.  => std::optional<ClassType*>
  def cc_mutable_getter_type(
      self, optionalness: Optionalness = ast_def_pb2.OPTIONALNESS_UNSPECIFIED
  ) -> str:
    return self.cc_getter_type(CcGetterKind.MUTABLE, optionalness)

  # Prints the C++ return type for the const getter function.
  #
  # bool                              => bool
  # double                            => double
  # string                            => absl::string_view
  # ClassType                         => const ClassType*
  # Class1 | Class2                   => std::variant<const Class1*,
  #                                       const Class2*>
  # [ClassType]                       => const std::vector<std::unique_ptr<
  #                                       ClassType>>*
  # ClassType with maybe_null/undef.  => std::optional<const ClassType*>
  def cc_const_getter_type(
      self, optionalness: Optionalness = ast_def_pb2.OPTIONALNESS_UNSPECIFIED
  ) -> str:
    return self.cc_getter_type(CcGetterKind.CONST, optionalness)

  # ===========================================================================
  # CcMlirBuilderType() / CcMlirGetterType()
  # ===========================================================================

  # Prints the C++ type for MLIR builders. See the docstrings on
  # NonListType.cc_mlir_type() and ListType for the exact mapping.
  @abc.abstractmethod
  def cc_mlir_builder_type(self, kind: FieldKind) -> str:
    ...

  # Prints the C++ type for MLIR getters. Same as cc_mlir_builder_type()
  # except for ListType (mlir::OperandRange instead of
  # std::vector<mlir::Value>).
  @abc.abstractmethod
  def cc_mlir_getter_type(self, kind: FieldKind) -> str:
    ...

  # ===========================================================================
  # TdType()
  # ===========================================================================

  # Prints the MLIR TableGen type.
  @abc.abstractmethod
  def _td_type(self, kind: FieldKind) -> str:
    ...

  def td_type(
      self,
      kind: FieldKind,
      optionalness: Optionalness = ast_def_pb2.OPTIONALNESS_UNSPECIFIED,
  ) -> str:
    if optionalness not in _MAYBE_NULL_OPTIONALNESS:
      return self._td_type(kind)

    if kind == ast_def_pb2.FIELD_KIND_UNSPECIFIED:
      raise ValueError("Unspecified FieldKind.")
    elif kind == ast_def_pb2.FIELD_KIND_ATTR:
      return f"OptionalAttr<{self._td_type(kind)}>"
    elif kind in (ast_def_pb2.FIELD_KIND_LVAL, ast_def_pb2.FIELD_KIND_RVAL):
      return f"Optional<{self._td_type(kind)}>"
    elif kind == ast_def_pb2.FIELD_KIND_STMT:
      raise ValueError("Statement fields are not supported.")
    raise ValueError(f"Invalid FieldKind: {kind}")


class NonListType(Type):
  """NonListType ::= ScalarType, VariantType."""

  # For `NonListType`, `cc_mlir_builder_type` and `cc_mlir_getter_type` are
  # the same. See the docstrings on `Type.cc_mlir_builder_type` /
  # `Type.cc_mlir_getter_type`.
  @abc.abstractmethod
  def cc_mlir_type(self, kind: FieldKind) -> str:
    ...

  def cc_mlir_builder_type(self, kind: FieldKind) -> str:
    return self.cc_mlir_type(kind)

  def cc_mlir_getter_type(self, kind: FieldKind) -> str:
    return self.cc_mlir_type(kind)


class ListType(Type):
  """ListType { element_type: NonListType, element_maybe_null: bool }.

  We explicitly don't allow nested lists, so the element type of a list must
  be non-list.
  """

  def __init__(
      self,
      element_type: NonListType,
      element_maybe_null: MaybeNull,
      lang_name: str,
  ):
    super().__init__(TypeKind.LIST, lang_name)
    self.element_type = element_type
    self.element_maybe_null = element_maybe_null

  def js_type(self) -> str:
    inner = self.element_type.js_type_with_maybe_null(self.element_maybe_null)
    return f"[ {inner} ]"

  def _cc_type(self) -> str:
    optionalness = maybe_null_to_optionalness(self.element_maybe_null)
    return f"std::vector<{self.element_type.cc_type(optionalness)}>"

  def _cc_getter_type(self, getter_kind: CcGetterKind) -> str:
    if getter_kind == CcGetterKind.MUTABLE:
      return f"{self._cc_type()}*"
    else:
      return f"const {self._cc_type()}*"

  def cc_mlir_builder_type(self, kind: FieldKind) -> str:
    if kind == ast_def_pb2.FIELD_KIND_UNSPECIFIED:
      raise ValueError("Unspecified FieldKind.")
    elif kind == ast_def_pb2.FIELD_KIND_ATTR:
      return "mlir::ArrayAttr"
    elif kind in (ast_def_pb2.FIELD_KIND_LVAL, ast_def_pb2.FIELD_KIND_RVAL):
      return "std::vector<mlir::Value>"
    elif kind == ast_def_pb2.FIELD_KIND_STMT:
      raise ValueError("List of statements not supported.")
    raise ValueError(f"Invalid FieldKind: {kind}")

  def cc_mlir_getter_type(self, kind: FieldKind) -> str:
    if kind == ast_def_pb2.FIELD_KIND_UNSPECIFIED:
      raise ValueError("Unspecified FieldKind.")
    elif kind == ast_def_pb2.FIELD_KIND_ATTR:
      return "mlir::ArrayAttr"
    elif kind in (ast_def_pb2.FIELD_KIND_LVAL, ast_def_pb2.FIELD_KIND_RVAL):
      return "mlir::OperandRange"
    elif kind == ast_def_pb2.FIELD_KIND_STMT:
      raise ValueError("List of statements not supported.")
    raise ValueError(f"Invalid FieldKind: {kind}")

  def _td_type(self, kind: FieldKind) -> str:
    if kind == ast_def_pb2.FIELD_KIND_UNSPECIFIED:
      raise ValueError("Unspecified FieldKind.")
    elif kind == ast_def_pb2.FIELD_KIND_ATTR:
      element_optionalness = maybe_null_to_optionalness(
          self.element_maybe_null
      )
      element_td_type = self.element_type.td_type(kind, element_optionalness)
      return f'TypedArrayAttrBase<{element_td_type}, "">'
    elif kind in (ast_def_pb2.FIELD_KIND_LVAL, ast_def_pb2.FIELD_KIND_RVAL):
      # TODO(b/204592400) Variadic<Optional<AnyType>> is not supported.
      element_td_type = self.element_type.td_type(kind)
      return f"Variadic<{element_td_type}>"
    elif kind == ast_def_pb2.FIELD_KIND_STMT:
      raise ValueError("Statement fields are not supported.")
    raise ValueError(f"Invalid FieldKind: {kind}")


class ScalarType(NonListType):
  """Scalar type: non-variant and non-list."""


class VariantType(NonListType):
  """VariantType { types: [ScalarType] }.

  We explicitly limit the types a variant can hold to be scalar. In other
  words, we don't allow nested variants or lists in variants.
  """

  def __init__(self, types: list[ScalarType], lang_name: str):
    super().__init__(TypeKind.VARIANT, lang_name)
    self.types = types

  def js_type(self) -> str:
    return " | ".join(t.js_type() for t in self.types)

  def _cc_type(self) -> str:
    inner = ", ".join(t.cc_type() for t in self.types)
    return f"std::variant<{inner}>"

  def _cc_getter_type(self, getter_kind: CcGetterKind) -> str:
    inner = ", ".join(t.cc_getter_type(getter_kind) for t in self.types)
    return f"std::variant<{inner}>"

  def cc_mlir_type(self, kind: FieldKind) -> str:
    cc_mlir_types = {t.cc_mlir_type(kind) for t in self.types}

    if kind == ast_def_pb2.FIELD_KIND_UNSPECIFIED:
      raise ValueError("Unspecified FieldKind.")
    elif kind == ast_def_pb2.FIELD_KIND_ATTR:
      if len(cc_mlir_types) == 1:
        return next(iter(cc_mlir_types))
      return "mlir::Attribute"
    elif kind in (ast_def_pb2.FIELD_KIND_LVAL, ast_def_pb2.FIELD_KIND_RVAL):
      assert len(cc_mlir_types) == 1
      return next(iter(cc_mlir_types))
    elif kind == ast_def_pb2.FIELD_KIND_STMT:
      raise ValueError("Variant of statements not supported.")
    raise ValueError(f"Invalid FieldKind: {kind}")

  def _td_type(self, kind: FieldKind) -> str:
    type_kinds = {t.kind for t in self.types}

    def variant_attr_td_type() -> str:
      td_types = ", ".join(t.td_type(kind) for t in self.types)
      return f"AnyAttrOf<[{td_types}]>"

    # Variant of builtin types.
    if type_kinds == {TypeKind.BUILTIN}:
      return variant_attr_td_type()

    # Variant of class types.
    if type_kinds == {TypeKind.CLASS}:
      if kind == ast_def_pb2.FIELD_KIND_UNSPECIFIED:
        raise ValueError("Unspecified FieldKind.")
      elif kind == ast_def_pb2.FIELD_KIND_ATTR:
        return variant_attr_td_type()
      elif kind in (ast_def_pb2.FIELD_KIND_LVAL, ast_def_pb2.FIELD_KIND_RVAL):
        return "AnyType"
      elif kind == ast_def_pb2.FIELD_KIND_STMT:
        raise ValueError("Statement fields are not supported.")
      raise ValueError(f"Invalid FieldKind: {kind}")

    raise ValueError(
        "We only support variants of builtin types or variants of class"
        " types."
    )


class BuiltinType(ScalarType):
  """BuiltinType ::= BoolType, Int64Type, DoubleType, StringType."""

  def __init__(self, builtin_kind: BuiltinTypeKind, lang_name: str):
    super().__init__(TypeKind.BUILTIN, lang_name)
    self.builtin_kind = builtin_kind

  def js_type(self) -> str:
    return {
        BuiltinTypeKind.BOOL: "boolean",
        BuiltinTypeKind.INT64: "/*int64*/number",
        BuiltinTypeKind.DOUBLE: "/*double*/number",
        BuiltinTypeKind.STRING: "string",
    }[self.builtin_kind]

  def _cc_type(self) -> str:
    return {
        BuiltinTypeKind.BOOL: "bool",
        BuiltinTypeKind.INT64: "int64_t",
        BuiltinTypeKind.DOUBLE: "double",
        BuiltinTypeKind.STRING: "std::string",
    }[self.builtin_kind]

  def _cc_getter_type(self, getter_kind: CcGetterKind) -> str:
    if self.builtin_kind == BuiltinTypeKind.STRING:
      return "absl::string_view"
    return self._cc_type()

  def cc_mlir_type(self, kind: FieldKind) -> str:
    if kind != ast_def_pb2.FIELD_KIND_ATTR:
      raise ValueError(f"Invalid FieldKind: {kind}")
    return {
        BuiltinTypeKind.BOOL: "mlir::BoolAttr",
        BuiltinTypeKind.INT64: "mlir::IntegerAttr",
        BuiltinTypeKind.DOUBLE: "mlir::FloatAttr",
        BuiltinTypeKind.STRING: "mlir::StringAttr",
    }[self.builtin_kind]

  def _td_type(self, kind: FieldKind) -> str:
    assert kind == ast_def_pb2.FIELD_KIND_ATTR, (
        f"Invalid FieldKind for builtin type: {kind}"
    )
    return {
        BuiltinTypeKind.BOOL: "BoolAttr",
        BuiltinTypeKind.INT64: "I64Attr",
        BuiltinTypeKind.DOUBLE: "F64Attr",
        BuiltinTypeKind.STRING: "StrAttr",
    }[self.builtin_kind]


class EnumType(ScalarType):
  """Represents an enum type defined elsewhere."""

  def __init__(self, name: Symbol, lang_name: str):
    super().__init__(TypeKind.ENUM, lang_name)
    self.name = name

  def js_type(self) -> str:
    return self.name.to_pascal_case()

  def _cc_type(self) -> str:
    return (Symbol(self.lang_name) + self.name).to_pascal_case()

  def _cc_getter_type(self, getter_kind: CcGetterKind) -> str:
    return self._cc_type()

  def cc_mlir_type(self, kind: FieldKind) -> str:
    if kind != ast_def_pb2.FIELD_KIND_ATTR:
      raise ValueError(f"Invalid FieldKind: {kind}")
    return "mlir::StringAttr"

  def _td_type(self, kind: FieldKind) -> str:
    if kind != ast_def_pb2.FIELD_KIND_ATTR:
      raise ValueError(f"Invalid FieldKind for enum type: {kind}")
    # TODO(b/182441574): Properly support enums.
    return "StrAttr"


class ClassType(ScalarType):
  """ClassType { name: Symbol }. Represents an AST node type defined elsewhere."""

  def __init__(self, name: Symbol, lang_name: str):
    super().__init__(TypeKind.CLASS, lang_name)
    self.name = name
    # Set by AstDef once the full schema (and its NodeDefs) is resolved. May
    # remain None (e.g. in unit tests that construct a ClassType directly).
    self.node_def = None

  def js_type(self) -> str:
    return self.name.to_pascal_case()

  def cc_class_name(self) -> str:
    return (Symbol(self.lang_name) + self.name).to_pascal_case()

  def _cc_type(self) -> str:
    return f"std::unique_ptr<{self.cc_class_name()}>"

  def _cc_getter_type(self, getter_kind: CcGetterKind) -> str:
    if getter_kind == CcGetterKind.MUTABLE:
      return f"{self.cc_class_name()}*"
    else:
      return f"const {self.cc_class_name()}*"

  def cc_mlir_type(self, kind: FieldKind) -> str:
    if kind == ast_def_pb2.FIELD_KIND_UNSPECIFIED:
      raise ValueError("Unspecified FieldKind.")
    elif kind == ast_def_pb2.FIELD_KIND_ATTR:
      if self.node_def is not None:
        ir_op_name = self.node_def.ir_op_name(self.lang_name, kind)
        if ir_op_name is not None:
          return ir_op_name.to_pascal_case()
      ir_name = Symbol(f"{self.lang_name}ir")
      return (ir_name + self.name + "Attr").to_pascal_case()
    elif kind in (ast_def_pb2.FIELD_KIND_LVAL, ast_def_pb2.FIELD_KIND_RVAL):
      return "mlir::Value"
    elif kind == ast_def_pb2.FIELD_KIND_STMT:
      raise ValueError(f"Invalid FieldKind: {kind}")
    raise ValueError(f"Invalid FieldKind: {kind}")

  def _td_type(self, kind: FieldKind) -> str:
    if kind == ast_def_pb2.FIELD_KIND_UNSPECIFIED:
      raise ValueError("Unspecified FieldKind.")
    elif kind == ast_def_pb2.FIELD_KIND_ATTR:
      if self.node_def is not None:
        ir_op_name = self.node_def.ir_op_name(self.lang_name, kind)
        if ir_op_name is not None:
          return ir_op_name.to_pascal_case()
      return (
          Symbol(self.lang_name + "ir") + self.name + "Attr"
      ).to_pascal_case()
    elif kind in (ast_def_pb2.FIELD_KIND_LVAL, ast_def_pb2.FIELD_KIND_RVAL):
      return "AnyType"
    elif kind == ast_def_pb2.FIELD_KIND_STMT:
      raise ValueError("Statement fields are not supported.")
    raise ValueError(f"Invalid FieldKind: {kind}")


def _from_bool_type_pb(pb: type_pb2.BoolTypePb) -> BuiltinType:
  del pb  # Unused; BoolTypePb is an empty marker message.
  return BuiltinType(BuiltinTypeKind.BOOL, "")


def _from_int64_type_pb(pb: type_pb2.Int64TypePb) -> BuiltinType:
  del pb  # Unused; Int64TypePb is an empty marker message.
  return BuiltinType(BuiltinTypeKind.INT64, "")


def _from_double_type_pb(pb: type_pb2.DoubleTypePb) -> BuiltinType:
  del pb  # Unused; DoubleTypePb is an empty marker message.
  return BuiltinType(BuiltinTypeKind.DOUBLE, "")


def _from_string_type_pb(pb: type_pb2.StringTypePb) -> BuiltinType:
  del pb  # Unused; StringTypePb is an empty marker message.
  return BuiltinType(BuiltinTypeKind.STRING, "")


def _from_enum_type_pb(enum_name: str, lang_name: str) -> EnumType:
  return EnumType(Symbol(enum_name), lang_name)


def _from_class_type_pb(class_name: str, lang_name: str) -> ClassType:
  return ClassType(Symbol(class_name), lang_name)


def _from_scalar_type_pb(
    pb: type_pb2.ScalarTypePb, lang_name: str
) -> ScalarType:
  kind_case = pb.WhichOneof("kind")
  if kind_case is None:
    raise ValueError("Invalid variant element type: KIND_NOT_SET.")
  elif kind_case == "bool":
    return _from_bool_type_pb(pb.bool)
  elif kind_case == "int64":
    return _from_int64_type_pb(pb.int64)
  elif kind_case == "double":
    return _from_double_type_pb(pb.double)
  elif kind_case == "string":
    return _from_string_type_pb(pb.string)
  elif kind_case == "enum":
    return _from_enum_type_pb(pb.enum, lang_name)
  elif kind_case == "class":
    return _from_class_type_pb(getattr(pb, "class"), lang_name)
  raise AssertionError(f"Unexpected ScalarTypePb kind: {kind_case}")


def _from_variant_type_pb(
    pb: type_pb2.VariantTypePb, lang_name: str
) -> VariantType:
  types = [_from_scalar_type_pb(t, lang_name) for t in pb.types]

  if not types:
    raise ValueError("Empty variant type.")

  if len(types) == 1:
    raise ValueError("Variant with only one case.")

  return VariantType(types, lang_name)


def _from_non_list_type_pb(
    pb: type_pb2.NonListTypePb, lang_name: str
) -> NonListType:
  kind_case = pb.WhichOneof("kind")
  if kind_case is None:
    raise ValueError("Invalid list element type: KIND_NOT_SET.")
  elif kind_case == "bool":
    return _from_bool_type_pb(pb.bool)
  elif kind_case == "int64":
    return _from_int64_type_pb(pb.int64)
  elif kind_case == "double":
    return _from_double_type_pb(pb.double)
  elif kind_case == "string":
    return _from_string_type_pb(pb.string)
  elif kind_case == "enum":
    return _from_enum_type_pb(pb.enum, lang_name)
  elif kind_case == "class":
    return _from_class_type_pb(getattr(pb, "class"), lang_name)
  elif kind_case == "variant":
    return _from_variant_type_pb(pb.variant, lang_name)
  raise AssertionError(f"Unexpected NonListTypePb kind: {kind_case}")


def _from_list_type_pb(pb: type_pb2.ListTypePb, lang_name: str) -> ListType:
  element_type = _from_non_list_type_pb(pb.element_type, lang_name)
  element_maybe_null = (
      MaybeNull.YES if pb.element_maybe_null else MaybeNull.NO
  )
  return ListType(element_type, element_maybe_null, lang_name)


def from_type_pb(pb: type_pb2.TypePb, lang_name: str = "") -> Type:
  """Converts from TypePb to Type. Raises ValueError on an invalid TypePb."""
  kind_case = pb.WhichOneof("kind")
  if kind_case is None:
    raise ValueError("Invalid TypePb: KIND_NOT_SET.")
  elif kind_case == "bool":
    return _from_bool_type_pb(pb.bool)
  elif kind_case == "int64":
    return _from_int64_type_pb(pb.int64)
  elif kind_case == "double":
    return _from_double_type_pb(pb.double)
  elif kind_case == "string":
    return _from_string_type_pb(pb.string)
  elif kind_case == "enum":
    return _from_enum_type_pb(pb.enum, lang_name)
  elif kind_case == "class":
    return _from_class_type_pb(getattr(pb, "class"), lang_name)
  elif kind_case == "variant":
    return _from_variant_type_pb(pb.variant, lang_name)
  elif kind_case == "list":
    return _from_list_type_pb(pb.list, lang_name)
  raise AssertionError(f"Unexpected TypePb kind: {kind_case}")
