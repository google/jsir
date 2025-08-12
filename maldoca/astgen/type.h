// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MALDOCA_ASTGEN_TYPE_H_
#define MALDOCA_ASTGEN_TYPE_H_

#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "maldoca/astgen/ast_def.pb.h"
#include "maldoca/astgen/symbol.h"
#include "maldoca/astgen/type.pb.h"

namespace maldoca {

class NodeDef;

// The Type Hierarchy
//
// Type        ::= NonListType, ListType
// NonListType ::= ScalarType, VariantType
// ScalarType  ::= BuiltinType, ClassType
// BuiltinType ::= BoolType, DoubleType, StringType
//
//                                  Type
//                                    |
//                        +-----------+-----------+
//                        |                       |
//                   NonListType                  |
//                        |                       |
//             +----------+----------+            |
//             |                     |            |
//         ScalarType                |            |
//             |                     |            |
//      +------+-------+             |            |
//      |              |             |            |
//  BuiltinType    ClassType    VariantType    ListType

enum class MaybeNull {
  kNo,
  kYes,
};

enum class TypeKind {
  kBuiltin,
  kEnum,
  kClass,
  kVariant,
  kList,
};

class Type {
 public:
  virtual ~Type() = default;

  // Check if a Type is a specific type T.
  //
  // Usage:
  //
  // const Type &type = ...;
  // if (type.IsA<NonListType>()) {
  //   const auto &non_list_type = static_cast<const NonListType &>(type);
  //   ...
  // }
  //
  // This is mimicking the LLVM-style RTTI.
  // https://llvm.org/docs/ProgrammersManual.html
  //
  // Each class T that inherits Type needs to have a IsTheClassOf() static
  // method that checks if a Type is a T.
  template <typename T>
  bool IsA() const {
    static_assert(std::is_base_of_v<Type, T>, "T is not a subclass of Type.");
    return T::IsTheClassOf(*this);
  }

  static bool IsTheClassOf(const Type &type) { return true; }

  TypeKind kind() const { return kind_; }

  absl::string_view lang_name() const { return lang_name_; }

  // Prints TypeScript type annotations.
  //
  // Types that are maybe_null are printed as variants.
  // E.g. "bool"          with "maybe_null = true" ==> "bool | null".
  // E.g. "bool | string" with "maybe_null = true" ==> "bool | string | null".
  virtual std::string JsType() const = 0;
  std::string JsType(MaybeNull maybe_null) const;

  // Prints the C++ type for storing the field.
  //
  // Types that are maybe_null or maybe_undefined are printed as
  // "std::optional".
  //
  // bool
  //   => bool
  //
  // double
  //   => double
  //
  // string
  //   => std::string
  //
  // ClassType
  //   => std::unique_ptr<ClassType>
  //
  // Class1 | Class2
  //   => std::variant<std::unique_ptr<Class1>, std::unique_ptr<Class2>>
  //
  // [ClassType]
  //   => std::vector<std::unique_ptr<ClassType>>
  //
  // ClassType with maybe_null or maybe_undefined
  //   => std::optional<std::unique_ptr<ClassType>>
  virtual std::string CcType() const = 0;
  std::string CcType(MaybeNull maybe_null) const;
  std::string CcType(Optionalness optionalness) const;

  // Prints the C++ return type for the getter function.
  //
  // Types that are maybe_null or maybe_undefined are printed as
  // "std::optional".
  //
  // bool
  //   => bool
  //
  // double
  //   => double
  //
  // string
  //   => std::string
  //
  // ClassType
  //   => ClassType*
  //
  // Class1 | Class2
  //   => std::variant<Class1*, Class2*>
  //
  // [ClassType]
  //   => std::vector<std::unique_ptr<ClassType>>*
  //
  // ClassType with maybe_null or maybe_undefined
  //   => std::optional<ClassType*>
  std::string CcMutableGetterType() const;
  std::string CcMutableGetterType(MaybeNull maybe_null) const;
  std::string CcMutableGetterType(Optionalness optionalness) const;

  // Prints the C++ return type for the const getter function.
  //
  // bool
  //   => bool
  //
  // double
  //   => double
  //
  // string
  //   => absl::string_view
  //
  // ClassType
  //   => const ClassType*
  //
  // Class1 | Class2
  //   => std::variant<const Class1*, const Class2*>
  //
  // [ClassType]
  //   => const std::vector<std::unique_ptr<ClassType>>*
  //
  // ClassType with maybe_null or maybe_undefined
  //   => std::optional<const ClassType*>
  std::string CcConstGetterType(MaybeNull maybe_null) const;
  std::string CcConstGetterType(Optionalness optionalness) const;
  std::string CcConstGetterType() const;

  // Common functions that handle both CcMutableGetterType() and
  // CcConstGetterType().
  enum class CcGetterKind {
    kMutable,
    kConst,
  };
  virtual std::string CcGetterType(CcGetterKind getter_kind) const = 0;
  std::string CcGetterType(CcGetterKind getter_kind,
                           MaybeNull maybe_null) const;
  std::string CcGetterType(CcGetterKind getter_kind,
                           Optionalness optionalness) const;

  // Prints the C++ type for MLIR builders.
  //
  // - maybe_null/optionalness:
  //   Whether to qualify the type with optional (Type by itself is
  //   non-optional).
  //
  // - kind:
  //   Each field in an AST node has a kind. Different kinds lead to different
  //   ops in the IR. See detailed explanations in ast_def.proto, and concrete
  //   listings below.
  //
  // Builtin type: kind must be FIELD_KIND_ATTR.
  //   bool   => mlir::BoolAttr
  //   int64  => mlir::IntegerAttr
  //   double => mlir::FloatAttr
  //   string => mlir::StringAttr
  //
  // Builtin type with maybe_null or maybe_undefined:
  //   Same as above. MLIR attributes can be nullptr.
  //
  // ClassType with kind == FIELD_KIND_LVAL or FIELD_KIND_RVAL:
  //   => mlir::Value
  //
  // ClassType with kind == FIELD_KIND_ATTR:
  //   => IrNameClassTypeAttr
  //
  // ClassType with maybe_null or maybe_undefined
  //   => Same as above. MLIR values can be nullptr.
  //
  // Class1 | Class2: kind must be FIELD_KIND_LVAL or FIELD_KIND_RVAL.
  //   => mlir::Value
  //
  // Builtin1 | Builtin2: kind must be FIELD_KIND_ATTR.
  //   => mlir::Attribute
  //
  // Builtin1 | Builtin2 with maybe_null or maybe_undefined:
  //   Same as above. MLIR attributes can be nullptr.
  //
  // [ClassType] with kind == FIELD_KIND_LVAL or FIELD_KIND_RVAL:
  //   => std::vector<mlir::Value>
  //
  // [ClassType] with kind == FIELD_KIND_ATTR:
  //   => std::vector<IrNameClassTypeAttr>
  //
  // [Builtin]
  //   => mlir::ArrayAttr
  virtual std::string CcMlirBuilderType(FieldKind kind) const = 0;

  // Prints the C++ type for MLIR getters.
  //
  // - maybe_null/optionalness:
  //   Whether to qualify the type with optional (Type by itself is
  //   non-optional).
  //
  // - kind:
  //   Each field in an AST node has a kind. Different kinds lead to different
  //   ops in the IR. See detailed explanations in ast_def.proto, and concrete
  //   listings below.
  //
  // Builtin type: kind must be FIELD_KIND_ATTR.
  //   bool   => mlir::BoolAttr
  //   int64  => mlir::IntegerAttr
  //   double => mlir::FloatAttr
  //   string => mlir::StringAttr
  //
  // Builtin type with maybe_null or maybe_undefined:
  //   Same as above. MLIR attributes can be nullptr.
  //
  // ClassType with kind == FIELD_KIND_LVAL or FIELD_KIND_RVAL:
  //   => mlir::Value
  //
  // ClassType with kind == FIELD_KIND_ATTR:
  //   => IrNameClassTypeAttr
  //
  // ClassType with maybe_null or maybe_undefined
  //   => Same as above. MLIR values can be nullptr.
  //
  // Class1 | Class2: kind must be FIELD_KIND_LVAL or FIELD_KIND_RVAL.
  //   => mlir::Value
  //
  // Builtin1 | Builtin2: kind must be FIELD_KIND_ATTR.
  //   => mlir::Attribute
  //
  // Builtin1 | Builtin2 with maybe_null or maybe_undefined:
  //   Same as above. MLIR attributes can be nullptr.
  //
  // [ClassType] with kind == FIELD_KIND_LVAL or FIELD_KIND_RVAL:
  //   => mlir::OperandRange
  //
  // [ClassType] with kind == FIELD_KIND_ATTR:
  //   => mlir::OperandRange
  //
  // [Builtin]
  //   => mlir::ArrayAttr
  virtual std::string CcMlirGetterType(FieldKind kind) const = 0;

  // Prints the MLIR TableGen type.
  //
  // - maybe_null/optionalness:
  //   Whether to qualify the type with optional (Type by itself is
  //   non-optional).
  //
  // - kind:
  //   Each field in an AST node has a kind. Different kinds lead to different
  //   ops in the IR. See detailed explanations in ast_def.proto.
  //   Currently the only difference here is that for an attribute, we use
  //   OptionalAttr<...>; otherwise, we use Optional<...>.
  //
  // Builtin type: kind must be FIELD_KIND_ATTR.
  //   bool   => BoolAttr
  //   int64  => I64Attr
  //   double => F64Attr
  //   string => StrAttr
  //
  // Builtin type with maybe_null or maybe_undefined:
  //   OptionalAttr<...>
  //
  // ClassType: kind must be FIELD_KIND_LVAL or FIELD_KIND_RVAL.
  //   => AnyType
  //
  // Class1 | Class2: kind must be FIELD_KIND_LVAL or FIELD_KIND_RVAL.
  //   => AnyType
  //
  // Builtin1 | Builtin2: kind must be FIELD_KIND_ATTR.
  //   => AnyAttrOf<Builtin1, Builtin2>
  //
  // Builtin1 | Builtin2 with maybe_null or maybe_undefined:
  //   => OptionalAttr<AnyAttrOf<Builtin1, Builtin2>>
  //
  // ClassType with maybe_null or maybe_undefined
  //   => Optional<AnyType>
  //
  // [ClassType]
  //   => Variadic<AnyType>
  //
  // Currently, maybe_null and maybe_undefined are not supported for list types
  // and list element types.
  std::string TdType(MaybeNull maybe_null, FieldKind kind) const;
  std::string TdType(Optionalness optionalness, FieldKind kind) const;
  virtual std::string TdType(FieldKind kind) const = 0;

 protected:
  explicit Type(TypeKind kind, absl::string_view lang_name)
      : lang_name_(lang_name), kind_(kind) {}
  std::string lang_name_;

 private:
  const TypeKind kind_;
  friend class AstDef;
};

class NonListType : public Type {
 public:
  static bool IsTheClassOf(const Type &type) {
    return type.kind() == TypeKind::kBuiltin ||
           type.kind() == TypeKind::kEnum || type.kind() == TypeKind::kClass ||
           type.kind() == TypeKind::kVariant;
  }

  // For `NonListType`, `CcMlirGetterType` and `CcMlirBuilderType` are the same.
  // For the definitions of `CcMlirGetterType` and `CcMlirBuilderType`, see
  // comments for class `Type`.
  virtual std::string CcMlirType(FieldKind kind) const = 0;

  std::string CcMlirBuilderType(FieldKind kind) const final {
    return CcMlirType(kind);
  }

  std::string CcMlirGetterType(FieldKind kind) const final {
    return CcMlirType(kind);
  }

 protected:
  explicit NonListType(TypeKind kind, absl::string_view lang_name)
      : Type(kind, lang_name) {}
};

// ListType {
//   element_type: NonListType
//   element_maybe_null: bool
// }
//
// We explicitly don't allow nested lists, so the element type of a list must be
// non-list.
class ListType : public Type {
 public:
  explicit ListType(std::unique_ptr<NonListType> element_type,
                    MaybeNull element_maybe_null, absl::string_view lang_name)
      : Type(TypeKind::kList, lang_name),
        element_type_(std::move(element_type)),
        element_maybe_null_(element_maybe_null) {}

  static bool IsTheClassOf(const Type &type) {
    return type.kind() == TypeKind::kList;
  }

  const NonListType &element_type() const { return *element_type_; }
  NonListType &element_type() { return *element_type_; }

  MaybeNull element_maybe_null() const { return element_maybe_null_; }

  std::string JsType() const override;

  std::string CcType() const override;

  std::string CcGetterType(CcGetterKind getter_kind) const override;

  std::string CcMlirBuilderType(FieldKind kind) const override;

  std::string CcMlirGetterType(FieldKind kind) const override;

  std::string TdType(FieldKind kind) const override;

 private:
  std::unique_ptr<NonListType> element_type_;
  MaybeNull element_maybe_null_;
};

// Scalar type: non-variant and non-list.
class ScalarType : public NonListType {
 public:
  static bool IsTheClassOf(const Type &type) {
    return type.kind() == TypeKind::kBuiltin ||
           type.kind() == TypeKind::kEnum || type.kind() == TypeKind::kClass;
  }

 protected:
  explicit ScalarType(TypeKind kind, absl::string_view lang_name)
      : NonListType(kind, lang_name) {}
};

// VariantType {
//   types: [ScalarType]
// }
//
// We explicitly limit the types a variant can hold to be scalar. In other
// words, we don't allow nested variants or lists in variants.
class VariantType : public NonListType {
 public:
  explicit VariantType(std::vector<std::unique_ptr<ScalarType>> types,
                       absl::string_view lang_name)
      : NonListType(TypeKind::kVariant, lang_name), types_(std::move(types)) {}

  static bool IsTheClassOf(const Type &type) {
    return type.kind() == TypeKind::kVariant;
  }

  absl::Span<const std::unique_ptr<ScalarType>> types() const { return types_; }

  absl::Span<std::unique_ptr<ScalarType>> types() {
    return absl::MakeSpan(types_);
  }

  std::string JsType() const override;

  std::string CcType() const override;

  std::string CcGetterType(CcGetterKind getter_kind) const override;

  std::string CcMlirType(FieldKind kind) const final;

  std::string TdType(FieldKind kind) const override;

 private:
  std::vector<std::unique_ptr<ScalarType>> types_;
};

enum class BuiltinTypeKind {
  kBool,
  kInt64,
  kDouble,
  kString,
};

class BuiltinType : public ScalarType {
 public:
  explicit BuiltinType(BuiltinTypeKind builtin_kind,
                       absl::string_view lang_name)
      : ScalarType(TypeKind::kBuiltin, lang_name),
        builtin_kind_(builtin_kind) {}

  static bool IsTheClassOf(const Type &type) {
    return type.kind() == TypeKind::kBuiltin;
  }

  BuiltinTypeKind builtin_kind() const { return builtin_kind_; }

  std::string JsType() const override;

  std::string CcType() const override;

  std::string CcGetterType(CcGetterKind getter_kind) const override;

  std::string CcMlirType(FieldKind kind) const final;

  std::string TdType(FieldKind kind) const override;

 private:
  BuiltinTypeKind builtin_kind_;
};

// Represents an enum type defined elsewhere.
class EnumType : public ScalarType {
 public:
  explicit EnumType(const Symbol &name, absl::string_view lang_name)
      : ScalarType(TypeKind::kEnum, lang_name), name_(name) {}

  static bool IsTheClassOf(const Type &type) {
    return type.kind() == TypeKind::kEnum;
  }

  const Symbol &name() const { return name_; }

  std::string JsType() const override;

  std::string CcType() const override;

  std::string CcGetterType(CcGetterKind getter_kind) const override;

  std::string CcMlirType(FieldKind kind) const final;

  std::string TdType(FieldKind kind) const override;

 private:
  Symbol name_;
};

// ClassType {
//   name: Symbol
// }
//
// Represents an AST node type defined elsewhere.
class ClassType : public ScalarType {
 public:
  explicit ClassType(const Symbol &name, absl::string_view lang_name)
      : ScalarType(TypeKind::kClass, lang_name), name_(name) {}

  static bool IsTheClassOf(const Type &type) {
    return type.kind() == TypeKind::kClass;
  }

  const Symbol &name() const { return name_; }

  std::string JsType() const override;

  std::string CcType() const override;

  std::string CcGetterType(CcGetterKind getter_kind) const override;

  std::string CcMlirType(FieldKind kind) const final;

  std::string TdType(FieldKind kind) const override;

 private:
  Symbol name_;
  const NodeDef *absl_nullable node_def_ = nullptr;

  friend class AstDef;
};

// Converts from TypePb to Type.
absl::StatusOr<std::unique_ptr<Type>> FromTypePb(const TypePb &pb,
                                                 absl::string_view lang_name);

}  // namespace maldoca

#endif  // MALDOCA_ASTGEN_TYPE_H_
