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

#include "maldoca/astgen/type.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/bind_front.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "maldoca/astgen/ast_def.pb.h"
#include "maldoca/astgen/symbol.h"
#include "maldoca/astgen/ast_def.h"
#include "maldoca/astgen/type.pb.h"
#include "maldoca/base/status_macros.h"

namespace maldoca {
namespace {

std::unique_ptr<BuiltinType> FromBoolTypePb(const BoolTypePb &pb) {
  return absl::make_unique<BuiltinType>(BuiltinTypeKind::kBool, "");
}

std::unique_ptr<BuiltinType> FromInt64TypePb(const Int64TypePb &pb) {
  return absl::make_unique<BuiltinType>(BuiltinTypeKind::kInt64, "");
}

std::unique_ptr<BuiltinType> FromDoubleTypePb(const DoubleTypePb &pb) {
  return absl::make_unique<BuiltinType>(BuiltinTypeKind::kDouble, "");
}

std::unique_ptr<BuiltinType> FromStringTypePb(const StringTypePb &pb) {
  return absl::make_unique<BuiltinType>(BuiltinTypeKind::kString, "");
}

std::unique_ptr<EnumType> FromEnumTypePb(absl::string_view enum_,
                                         absl::string_view lang_name_) {
  return absl::make_unique<EnumType>(Symbol(enum_), lang_name_);
}

std::unique_ptr<ClassType> FromClassTypePb(absl::string_view class_,
                                           absl::string_view lang_name_) {
  return absl::make_unique<ClassType>(Symbol(class_), lang_name_);
}

absl::StatusOr<std::unique_ptr<VariantType>> FromVariantTypePb(
    const VariantTypePb &pb, absl::string_view lang_name) {
  std::vector<std::unique_ptr<ScalarType>> types;
  for (const ScalarTypePb &type : pb.types()) {
    switch (type.kind_case()) {
      case ScalarTypePb::KindCase::KIND_NOT_SET:
        return absl::InvalidArgumentError(
            "Invalid variant element type: KIND_NOT_SET.");

      case ScalarTypePb::KindCase::kBool:
        types.push_back(FromBoolTypePb(type.bool_()));
        break;

      case ScalarTypePb::KindCase::kInt64:
        types.push_back(FromInt64TypePb(type.int64()));
        break;

      case ScalarTypePb::KindCase::kDouble:
        types.push_back(FromDoubleTypePb(type.double_()));
        break;

      case ScalarTypePb::KindCase::kString:
        types.push_back(FromStringTypePb(type.string()));
        break;

      case ScalarTypePb::KindCase::kEnum:
        types.push_back(FromEnumTypePb(type.enum_(), lang_name));
        break;

      case ScalarTypePb::KindCase::kClass:
        types.push_back(FromClassTypePb(type.class_(), lang_name));
        break;
    }
  }

  if (types.empty()) {
    return absl::InvalidArgumentError("Empty variant type.");
  }

  if (types.size() == 1) {
    return absl::InvalidArgumentError("Variant with only one case.");
  }

  return absl::make_unique<VariantType>(std::move(types), lang_name);
}

absl::StatusOr<std::unique_ptr<ListType>> FromListTypePb(
    const ListTypePb &pb, absl::string_view lang_name) {
  std::unique_ptr<NonListType> element_type;
  switch (pb.element_type().kind_case()) {
    case NonListTypePb::KIND_NOT_SET:
      return absl::InvalidArgumentError(
          "Invalid list element type: KIND_NOT_SET.");

    case NonListTypePb::KindCase::kBool:
      element_type = FromBoolTypePb(pb.element_type().bool_());
      break;

    case NonListTypePb::KindCase::kInt64:
      element_type = FromInt64TypePb(pb.element_type().int64());
      break;

    case NonListTypePb::KindCase::kDouble:
      element_type = FromDoubleTypePb(pb.element_type().double_());
      break;

    case NonListTypePb::KindCase::kString:
      element_type = FromStringTypePb(pb.element_type().string());
      break;

    case NonListTypePb::KindCase::kEnum:
      element_type = FromEnumTypePb(pb.element_type().enum_(), lang_name);
      break;

    case NonListTypePb::KindCase::kClass:
      element_type = FromClassTypePb(pb.element_type().class_(), lang_name);
      break;

    case NonListTypePb::kVariant: {
      MALDOCA_ASSIGN_OR_RETURN(
          element_type,
          FromVariantTypePb(pb.element_type().variant(), lang_name));
      break;
    }
  }

  return absl::make_unique<ListType>(
      std::move(element_type),
      pb.element_maybe_null() ? MaybeNull::kYes : MaybeNull::kNo, lang_name);
}

}  // namespace

absl::StatusOr<std::unique_ptr<Type>> FromTypePb(const TypePb &pb,
                                                 absl::string_view lang_name) {
  switch (pb.kind_case()) {
    case TypePb::KindCase::KIND_NOT_SET:
      return absl::InvalidArgumentError("Invalid TypePb: KIND_NOT_SET.");

    case TypePb::KindCase::kBool:
      return FromBoolTypePb(pb.bool_());

    case TypePb::KindCase::kInt64:
      return FromInt64TypePb(pb.int64());

    case TypePb::KindCase::kDouble:
      return FromDoubleTypePb(pb.double_());

    case TypePb::KindCase::kString:
      return FromStringTypePb(pb.string());

    case TypePb::KindCase::kEnum:
      return FromEnumTypePb(pb.enum_(), lang_name);

    case TypePb::KindCase::kClass:
      return FromClassTypePb(pb.class_(), lang_name);

    case TypePb::KindCase::kVariant:
      return FromVariantTypePb(pb.variant(), lang_name);

    case TypePb::KindCase::kList:
      return FromListTypePb(pb.list(), lang_name);
  }
}

// =============================================================================
// JsType()
// =============================================================================

std::string Type::JsType(MaybeNull maybe_null) const {
  std::string str = JsType();
  switch (maybe_null) {
    case MaybeNull::kYes:
      return absl::StrCat(std::move(str), " | null");
    case MaybeNull::kNo:
      return str;
  }
}

std::string ListType::JsType() const {
  return absl::StrCat("[ ", element_type().JsType(element_maybe_null()), " ]");
}

std::string VariantType::JsType() const {
  std::vector<std::string> type_strings;
  for (const auto &type : types()) {
    type_strings.push_back(type->JsType());
  }
  return absl::StrJoin(type_strings, " | ");
}

std::string BuiltinType::JsType() const {
  switch (builtin_kind()) {
    case BuiltinTypeKind::kBool:
      return "boolean";
    case BuiltinTypeKind::kInt64:
      return "/*int64*/number";
    case BuiltinTypeKind::kDouble:
      return "/*double*/number";
    case BuiltinTypeKind::kString:
      return "string";
  }
}

std::string EnumType::JsType() const { return name().ToPascalCase(); }

std::string ClassType::JsType() const { return name().ToPascalCase(); }

// =============================================================================
// CcType()
// =============================================================================

std::string Type::CcType(MaybeNull maybe_null) const {
  switch (maybe_null) {
    case MaybeNull::kYes:
      return CcType(OPTIONALNESS_MAYBE_NULL);
    case MaybeNull::kNo:
      return CcType();
  }
}

std::string Type::CcType(Optionalness optionalness) const {
  std::string str = CcType();
  switch (optionalness) {
    case OPTIONALNESS_MAYBE_NULL:
    case OPTIONALNESS_MAYBE_UNDEFINED:
      return absl::StrCat("std::optional<", std::move(str), ">");
    default:
      return str;
  }
}

std::string ListType::CcType() const {
  return absl::StrCat("std::vector<",
                      element_type().CcType(element_maybe_null()), ">");
}

std::string VariantType::CcType() const {
  std::vector<std::string> type_strings;
  for (const auto &type : types()) {
    type_strings.push_back(type->CcType());
  }
  return absl::StrCat("std::variant<", absl::StrJoin(type_strings, ", "), ">");
}

std::string BuiltinType::CcType() const {
  switch (builtin_kind()) {
    case BuiltinTypeKind::kBool:
      return "bool";
    case BuiltinTypeKind::kInt64:
      return "int64_t";
    case BuiltinTypeKind::kDouble:
      return "double";
    case BuiltinTypeKind::kString:
      return "std::string";
  }
}

std::string EnumType::CcType() const {
  return (Symbol(lang_name_) + name()).ToPascalCase();
}

std::string ClassType::CcType() const {
  return absl::StrCat("std::unique_ptr<",
                      (Symbol(lang_name_) + name()).ToPascalCase(), ">");
}

// =============================================================================
// CcGetterType()
// =============================================================================

std::string Type::CcMutableGetterType() const {
  return CcGetterType(CcGetterKind::kMutable);
}

std::string Type::CcMutableGetterType(MaybeNull maybe_null) const {
  return CcGetterType(CcGetterKind::kMutable, maybe_null);
}

std::string Type::CcMutableGetterType(Optionalness optionalness) const {
  return CcGetterType(CcGetterKind::kMutable, optionalness);
}

std::string Type::CcConstGetterType() const {
  return CcGetterType(CcGetterKind::kConst);
}

std::string Type::CcConstGetterType(MaybeNull maybe_null) const {
  return CcGetterType(CcGetterKind::kConst, maybe_null);
}

std::string Type::CcConstGetterType(Optionalness optionalness) const {
  return CcGetterType(CcGetterKind::kConst, optionalness);
}

std::string Type::CcGetterType(CcGetterKind getter_kind,
                               MaybeNull maybe_null) const {
  switch (maybe_null) {
    case MaybeNull::kYes:
      return CcGetterType(getter_kind, OPTIONALNESS_MAYBE_NULL);
    case MaybeNull::kNo:
      return CcGetterType(getter_kind, OPTIONALNESS_REQUIRED);
  }
}

std::string Type::CcGetterType(CcGetterKind getter_kind,
                               Optionalness optionalness) const {
  std::string str = CcGetterType(getter_kind);
  switch (optionalness) {
    case OPTIONALNESS_MAYBE_NULL:
    case OPTIONALNESS_MAYBE_UNDEFINED:
      return absl::StrCat("std::optional<", std::move(str), ">");
    default:
      return str;
  }
}

std::string ListType::CcGetterType(CcGetterKind getter_kind) const {
  switch (getter_kind) {
    case CcGetterKind::kMutable:
      return absl::StrCat(CcType(), "*");
    case CcGetterKind::kConst:
      return absl::StrCat("const ", CcType(), "*");
  }
}

std::string VariantType::CcGetterType(CcGetterKind getter_kind) const {
  std::vector<std::string> type_strings;
  for (const auto &type : types()) {
    type_strings.push_back(type->CcGetterType(getter_kind));
  }
  return absl::StrCat("std::variant<", absl::StrJoin(type_strings, ", "), ">");
}

std::string BuiltinType::CcGetterType(CcGetterKind getter_kind) const {
  switch (builtin_kind()) {
    case BuiltinTypeKind::kBool:
      return "bool";
    case BuiltinTypeKind::kInt64:
      return "int64_t";
    case BuiltinTypeKind::kDouble:
      return "double";
    case BuiltinTypeKind::kString:
      return "absl::string_view";
  }
}

std::string EnumType::CcGetterType(CcGetterKind getter_kind) const {
  return (Symbol(lang_name_) + name()).ToPascalCase();
}

std::string ClassType::CcGetterType(CcGetterKind getter_kind) const {
  const auto class_name = (Symbol(lang_name_) + name()).ToPascalCase();
  switch (getter_kind) {
    case CcGetterKind::kMutable:
      return absl::StrCat(class_name, "*");
    case CcGetterKind::kConst:
      return absl::StrCat("const ", class_name, "*");
  }
}

// =============================================================================
// CcMlirBuilderType() / CcMlirGetterType()
// =============================================================================

std::string ListType::CcMlirBuilderType(FieldKind kind) const {
  switch (kind) {
    case FIELD_KIND_UNSPECIFIED:
      LOG(FATAL) << "Unspecified FieldKind.";
    case FIELD_KIND_ATTR:
      return "mlir::ArrayAttr";
    case FIELD_KIND_LVAL:
    case FIELD_KIND_RVAL:
      return "std::vector<mlir::Value>";
    case FIELD_KIND_STMT:
      LOG(FATAL) << "List of statements not supported.";
  }
}

std::string ListType::CcMlirGetterType(FieldKind kind) const {
  switch (kind) {
    case FIELD_KIND_UNSPECIFIED:
      LOG(FATAL) << "Unspecified FieldKind.";
    case FIELD_KIND_ATTR:
      return "mlir::ArrayAttr";
    case FIELD_KIND_LVAL:
    case FIELD_KIND_RVAL:
      return "mlir::OperandRange";
    case FIELD_KIND_STMT:
      LOG(FATAL) << "List of statements not supported.";
  }
}

std::string VariantType::CcMlirType(FieldKind kind) const {
  absl::flat_hash_set<std::string> cc_mlir_types;
  for (const auto &type : types()) {
    cc_mlir_types.insert(type->CcMlirType(kind));
  }

  switch (kind) {
    case FIELD_KIND_UNSPECIFIED:
      LOG(FATAL) << "Unspecified FieldKind.";
    case FIELD_KIND_ATTR: {
      if (cc_mlir_types.size() == 1) {
        return *cc_mlir_types.begin();
      }
      return "mlir::Attribute";
    }
    case FIELD_KIND_LVAL:
    case FIELD_KIND_RVAL: {
      CHECK_EQ(cc_mlir_types.size(), 1);
      return *cc_mlir_types.begin();
    }
    case FIELD_KIND_STMT:
      LOG(FATAL) << "Variant of statements not supported.";
  }
}

std::string BuiltinType::CcMlirType(FieldKind kind) const {
  switch (kind) {
    case FIELD_KIND_UNSPECIFIED:
      LOG(FATAL) << "Unspecified FieldKind.";
    case FIELD_KIND_ATTR:
      break;
    case FIELD_KIND_LVAL:
    case FIELD_KIND_RVAL:
    case FIELD_KIND_STMT:
      LOG(FATAL) << "Invalid FieldKind: " << kind;
  }

  switch (builtin_kind()) {
    case BuiltinTypeKind::kBool:
      return "mlir::BoolAttr";
    case BuiltinTypeKind::kInt64:
      return "mlir::IntegerAttr";
    case BuiltinTypeKind::kDouble:
      return "mlir::FloatAttr";
    case BuiltinTypeKind::kString:
      return "mlir::StringAttr";
  }
}

std::string EnumType::CcMlirType(FieldKind kind) const {
  switch (kind) {
    case FIELD_KIND_UNSPECIFIED:
      LOG(FATAL) << "Unspecified FieldKind.";
    case FIELD_KIND_ATTR:
      break;
    case FIELD_KIND_LVAL:
    case FIELD_KIND_RVAL:
    case FIELD_KIND_STMT:
      LOG(FATAL) << "Invalid FieldKind: " << kind;
  }

  return "mlir::StringAttr";
}

std::string ClassType::CcMlirType(FieldKind kind) const {
  switch (kind) {
    case FIELD_KIND_UNSPECIFIED:
      LOG(FATAL) << "Unspecified FieldKind.";
    case FIELD_KIND_ATTR: {
      if (node_def_ != nullptr) {
        auto ir_op_name = node_def_->ir_op_name(lang_name_, kind);
        if (ir_op_name.has_value()) {
          return ir_op_name->ToPascalCase();
        }
      }

      auto ir_name = Symbol(absl::StrCat(lang_name_, "ir"));
      return (ir_name + name() + "Attr").ToPascalCase();
    }
    case FIELD_KIND_LVAL:
    case FIELD_KIND_RVAL:
      return "mlir::Value";
    case FIELD_KIND_STMT:
      LOG(FATAL) << "Invalid FieldKind: " << kind;
  }
}

// =============================================================================
// TdType()
// =============================================================================

std::string Type::TdType(MaybeNull maybe_null, FieldKind kind) const {
  switch (maybe_null) {
    case MaybeNull::kNo:
      return TdType(kind);

    case MaybeNull::kYes: {
      switch (kind) {
        case FIELD_KIND_UNSPECIFIED:
          LOG(FATAL) << "Unspecified FieldKind.";
        case FIELD_KIND_ATTR:
          return absl::StrCat("OptionalAttr<", TdType(kind), ">");
        case FIELD_KIND_LVAL:
        case FIELD_KIND_RVAL:
          return absl::StrCat("Optional<", TdType(kind), ">");
        case FIELD_KIND_STMT:
          LOG(FATAL) << "Statement fields are not supported.";
      }
    }
  }
}

std::string Type::TdType(Optionalness optionalness, FieldKind kind) const {
  switch (optionalness) {
    case OPTIONALNESS_UNSPECIFIED:
    case OPTIONALNESS_REQUIRED:
      return TdType(MaybeNull::kNo, kind);
    case OPTIONALNESS_MAYBE_NULL:
    case OPTIONALNESS_MAYBE_UNDEFINED:
      return TdType(MaybeNull::kYes, kind);
  }
}

std::string ListType::TdType(FieldKind kind) const {
  switch (kind) {
    case FIELD_KIND_UNSPECIFIED:
      LOG(FATAL) << "Unspecified FieldKind.";
    case FIELD_KIND_ATTR: {
      auto element_td_type = element_type().TdType(element_maybe_null(), kind);
      return absl::StrCat("TypedArrayAttrBase<", element_td_type, ", \"\">");
    }
    case FIELD_KIND_LVAL:
    case FIELD_KIND_RVAL: {
      // TODO(b/204592400) Variadic<Optional<AnyType>> is not supported.
      auto element_td_type = element_type().TdType(kind);
      return absl::StrCat("Variadic<", element_td_type, ">");
    }
    case FIELD_KIND_STMT:
      LOG(FATAL) << "Statement fields are not supported.";
  }
}

std::string VariantType::TdType(FieldKind kind) const {
  std::vector<TypeKind> type_kinds;
  for (const auto &type : types()) {
    type_kinds.push_back(type->kind());
  }

  auto VariantAttrTdType = [&] {
    std::vector<std::string> td_types;
    for (const auto &type : types()) {
      td_types.push_back(type->TdType(kind));
    }

    return absl::StrCat("AnyAttrOf<[", absl::StrJoin(td_types, ", "), "]>");
  };

  // Variant of builtin types.
  if (absl::c_all_of(type_kinds, absl::bind_front(std::equal_to<TypeKind>(),
                                                  TypeKind::kBuiltin))) {
    return VariantAttrTdType();
  }

  // Variant of class types.
  if (absl::c_all_of(type_kinds, absl::bind_front(std::equal_to<TypeKind>(),
                                                  TypeKind::kClass))) {
    switch (kind) {
      case FIELD_KIND_UNSPECIFIED:
        LOG(FATAL) << "Unspecified FieldKind.";
      case FIELD_KIND_ATTR:
        return VariantAttrTdType();
      case FIELD_KIND_LVAL:
      case FIELD_KIND_RVAL:
        return "AnyType";
      case FIELD_KIND_STMT:
        LOG(FATAL) << "Statement fields are not supported.";
    }
  }

  LOG(FATAL) << "We only support variants of builtin types or variants of "
                "class types.";
}

std::string BuiltinType::TdType(FieldKind kind) const {
  CHECK_EQ(kind, FIELD_KIND_ATTR)
      << "Invalid FieldKind for builtin type: " << kind;

  switch (builtin_kind()) {
    case BuiltinTypeKind::kBool:
      return "BoolAttr";
    case BuiltinTypeKind::kInt64:
      return "I64Attr";
    case BuiltinTypeKind::kDouble:
      return "F64Attr";
    case BuiltinTypeKind::kString:
      return "StrAttr";
  }
}

std::string EnumType::TdType(FieldKind kind) const {
  CHECK_EQ(kind, FIELD_KIND_ATTR)
      << "Invalid FieldKind for enum type: " << kind;

  // TODO(b/182441574): Properly support enums.
  return "StrAttr";
}

std::string ClassType::TdType(FieldKind kind) const {
  switch (kind) {
    case FIELD_KIND_UNSPECIFIED:
      LOG(FATAL) << "Unspecified FieldKind.";
    case FIELD_KIND_ATTR: {
      if (node_def_ != nullptr) {
        auto ir_op_name = node_def_->ir_op_name(lang_name_, kind);
        if (ir_op_name.has_value()) {
          return ir_op_name->ToPascalCase();
        }
      }
      return (Symbol(lang_name_ + "ir") + name() + "Attr").ToPascalCase();
    }
    case FIELD_KIND_LVAL:
    case FIELD_KIND_RVAL:
      break;
    case FIELD_KIND_STMT:
      LOG(FATAL) << "Statement fields are not supported.";
  }

  return "AnyType";
}

}  // namespace maldoca
