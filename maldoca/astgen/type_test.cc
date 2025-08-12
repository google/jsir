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

#include <memory>
#include <string>
#include <tuple>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "maldoca/astgen/ast_def.pb.h"
#include "maldoca/astgen/type.pb.h"
#include "maldoca/base/testing/status_matchers.h"
#include "maldoca/base/filesystem.h"

namespace maldoca {
namespace {

struct TypeTestCase {
  const char *type_pb;
  const char *js_type;
  const char *cc_type;
  const char *cc_getter_type;
  const char *cc_const_getter_type;
  const char *cc_lang_name;
  absl::flat_hash_map<FieldKind, std::string> td_types;
  absl::flat_hash_map<FieldKind, std::string> cc_mlir_builder_type;
  absl::flat_hash_map<FieldKind, std::string> cc_mlir_getter_type;
};

void TestTypePbToTypeAndPrint(TypeTestCase type_test_case) {
  TypePb type_pb;
  MALDOCA_ASSERT_OK(ParseTextProto(type_test_case.type_pb,
                                   "type_test_case.type_pb", &type_pb));
  MALDOCA_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Type> type,
      FromTypePb(type_pb, type_test_case.cc_lang_name != nullptr
                              ? type_test_case.cc_lang_name
                              : ""));
  EXPECT_EQ(type->JsType(), type_test_case.js_type);
  EXPECT_EQ(type->CcType(), type_test_case.cc_type);
  EXPECT_EQ(type->CcMutableGetterType(), type_test_case.cc_getter_type);
  EXPECT_EQ(type->CcConstGetterType(), type_test_case.cc_const_getter_type);
  for (const auto &pair : type_test_case.td_types) {
    // We don't allow C++17 in the codebase for compatibility reasons, so we
    // cannot use structured binding.
    FieldKind field_kind;
    std::string td_type;
    std::tie(field_kind, td_type) = pair;

    EXPECT_EQ(type->TdType(field_kind), td_type);
  }
  for (const auto &pair : type_test_case.cc_mlir_builder_type) {
    // We don't allow C++17 in the codebase for compatibility reasons, so we
    // cannot use structured binding.
    FieldKind field_kind;
    std::string cc_mlir_builder_type;
    std::tie(field_kind, cc_mlir_builder_type) = pair;

    EXPECT_EQ(type->CcMlirBuilderType(field_kind), cc_mlir_builder_type);
  }
  for (const auto &pair : type_test_case.cc_mlir_getter_type) {
    // We don't allow C++17 in the codebase for compatibility reasons, so we
    // cannot use structured binding.
    // FieldKind field_kind;
    // std::string cc_mlir_getter_type;
    // std::tie(field_kind, cc_mlir_getter_type) = pair;
    auto [field_kind, cc_mlir_getter_type] = pair;

    EXPECT_EQ(type->CcMlirGetterType(field_kind), cc_mlir_getter_type);
  }
}

TEST(TypeTest, ConvertBuiltinType) {
  TestTypePbToTypeAndPrint({
      .type_pb = "bool {}",
      .js_type = "boolean",
      .cc_type = "bool",
      .cc_getter_type = "bool",
      .cc_const_getter_type = "bool",
      .td_types =
          {
              {FIELD_KIND_ATTR, "BoolAttr"},
          },
      .cc_mlir_builder_type =
          {
              {FIELD_KIND_ATTR, "mlir::BoolAttr"},
          },
      .cc_mlir_getter_type =
          {
              {FIELD_KIND_ATTR, "mlir::BoolAttr"},
          },
  });

  TestTypePbToTypeAndPrint({
      .type_pb = "int64 {}",
      .js_type = "/*int64*/number",
      .cc_type = "int64_t",
      .cc_getter_type = "int64_t",
      .cc_const_getter_type = "int64_t",
      .td_types =
          {
              {FIELD_KIND_ATTR, "I64Attr"},
          },
      .cc_mlir_builder_type =
          {
              {FIELD_KIND_ATTR, "mlir::IntegerAttr"},
          },
      .cc_mlir_getter_type =
          {
              {FIELD_KIND_ATTR, "mlir::IntegerAttr"},
          },
  });

  TestTypePbToTypeAndPrint({
      .type_pb = "double {}",
      .js_type = "/*double*/number",
      .cc_type = "double",
      .cc_getter_type = "double",
      .cc_const_getter_type = "double",
      .td_types =
          {
              {FIELD_KIND_ATTR, "F64Attr"},
          },
      .cc_mlir_builder_type =
          {
              {FIELD_KIND_ATTR, "mlir::FloatAttr"},
          },
      .cc_mlir_getter_type =
          {
              {FIELD_KIND_ATTR, "mlir::FloatAttr"},
          },
  });

  TestTypePbToTypeAndPrint({
      .type_pb = "string {}",
      .js_type = "string",
      .cc_type = "std::string",
      .cc_getter_type = "absl::string_view",
      .cc_const_getter_type = "absl::string_view",
      .td_types =
          {
              {FIELD_KIND_ATTR, "StrAttr"},
          },
      .cc_mlir_builder_type =
          {
              {FIELD_KIND_ATTR, "mlir::StringAttr"},
          },
      .cc_mlir_getter_type =
          {
              {FIELD_KIND_ATTR, "mlir::StringAttr"},
          },
  });
}

TEST(TypeTest, ConvertEnumType) {
  TestTypePbToTypeAndPrint({
      .type_pb = R"pb(enum: "BinaryOperator")pb",
      .js_type = "BinaryOperator",
      .cc_type = "TestLangNameBinaryOperator",
      .cc_getter_type = "TestLangNameBinaryOperator",
      .cc_const_getter_type = "TestLangNameBinaryOperator",
      .cc_lang_name = "TestLangName",
      .td_types =
          {
              {FIELD_KIND_ATTR, "StrAttr"},
          },
      .cc_mlir_builder_type =
          {
              {FIELD_KIND_ATTR, "mlir::StringAttr"},
          },
      .cc_mlir_getter_type =
          {
              {FIELD_KIND_ATTR, "mlir::StringAttr"},
          },
  });
}

TEST(TypeTest, ConvertClassType) {
  TestTypePbToTypeAndPrint({
      .type_pb = R"pb(class: "BinaryExpression")pb",
      .js_type = "BinaryExpression",
      .cc_type = "std::unique_ptr<TestLangNameBinaryExpression>",
      .cc_getter_type = "TestLangNameBinaryExpression*",
      .cc_const_getter_type = "const TestLangNameBinaryExpression*",
      .cc_lang_name = "TestLangName",
      .td_types =
          {
              {FIELD_KIND_RVAL, "AnyType"},
          },
      .cc_mlir_builder_type =
          {
              {FIELD_KIND_RVAL, "mlir::Value"},
          },
      .cc_mlir_getter_type =
          {
              {FIELD_KIND_RVAL, "mlir::Value"},
          },
  });
}

TEST(TypeTest, ConvertVariantType) {
  TestTypePbToTypeAndPrint({
      .type_pb = R"pb(
        variant {
          types { bool {} }
          types { string {} }
        }
      )pb",
      .js_type = "boolean | string",
      .cc_type = "std::variant<bool, std::string>",
      .cc_getter_type = "std::variant<bool, absl::string_view>",
      .cc_const_getter_type = "std::variant<bool, absl::string_view>",
      .td_types =
          {
              {FIELD_KIND_ATTR, "AnyAttrOf<[BoolAttr, StrAttr]>"},
          },
      .cc_mlir_builder_type =
          {
              {FIELD_KIND_ATTR, "mlir::Attribute"},
          },
      .cc_mlir_getter_type =
          {
              {FIELD_KIND_ATTR, "mlir::Attribute"},
          },
  });

  TestTypePbToTypeAndPrint({
      .type_pb = R"pb(
        variant {
          types { class: "Expression" }
          types { class: "Pattern" }
        }
      )pb",
      .js_type = "Expression | Pattern",
      .cc_type = "std::variant<std::unique_ptr<TestLangNameExpression>, "
                 "std::unique_ptr<TestLangNamePattern>>",
      .cc_getter_type =
          "std::variant<TestLangNameExpression*, TestLangNamePattern*>",
      .cc_const_getter_type = "std::variant<const TestLangNameExpression*, "
                              "const TestLangNamePattern*>",
      .cc_lang_name = "TestLangName",
      .td_types =
          {
              {FIELD_KIND_RVAL, "AnyType"},
          },
      .cc_mlir_builder_type =
          {
              {FIELD_KIND_RVAL, "mlir::Value"},
          },
      .cc_mlir_getter_type =
          {
              {FIELD_KIND_RVAL, "mlir::Value"},
          },
  });
}

TEST(TypeTest, ConvertListType) {
  TestTypePbToTypeAndPrint({
      .type_pb = R"pb(
        list { element_type { class: "Expression" } }
      )pb",
      .js_type = "[ Expression ]",
      .cc_type = "std::vector<std::unique_ptr<TestLangNameExpression>>",
      .cc_getter_type = "std::vector<std::unique_ptr<TestLangNameExpression>>*",
      .cc_const_getter_type =
          "const std::vector<std::unique_ptr<TestLangNameExpression>>*",
      .cc_lang_name = "TestLangName",
      .td_types =
          {
              {
                  {FIELD_KIND_RVAL, "Variadic<AnyType>"},
              },
          },
      .cc_mlir_builder_type =
          {
              {FIELD_KIND_RVAL, "std::vector<mlir::Value>"},
          },
      .cc_mlir_getter_type =
          {
              {FIELD_KIND_RVAL, "mlir::OperandRange"},
          },
  });

  TestTypePbToTypeAndPrint({
      .type_pb = R"pb(
        list {
          element_type { class: "Expression" }
          element_maybe_null: true
        }
      )pb",
      .js_type = "[ Expression | null ]",
      .cc_type = "std::vector<std::optional<std::unique_ptr<"
                 "TestLangNameExpression>>>",
      .cc_getter_type = "std::vector<std::optional<std::unique_ptr<"
                        "TestLangNameExpression>>>*",
      .cc_const_getter_type = "const "
                              "std::vector<std::optional<std::unique_ptr<"
                              "TestLangNameExpression>>>*",
      .cc_lang_name = "TestLangName",
      .td_types =
          {
              {
                  {FIELD_KIND_RVAL, "Variadic<AnyType>"},
              },
          },
      .cc_mlir_builder_type =
          {
              {FIELD_KIND_RVAL, "std::vector<mlir::Value>"},
          },
      .cc_mlir_getter_type =
          {
              {FIELD_KIND_RVAL, "mlir::OperandRange"},
          },
  });

  TestTypePbToTypeAndPrint({
      .type_pb = R"pb(
        list {
          element_type {
            variant {
              types { class: "Expression" }
              types { class: "Pattern" }
            }
          }
          element_maybe_null: true
        }
      )pb",
      .js_type = "[ Expression | Pattern | null ]",
      .cc_type =
          "std::vector<std::optional<std::variant<std::unique_ptr<"
          "TestLangNameExpression>, std::unique_ptr<TestLangNamePattern>>>>",
      .cc_getter_type = "std::vector<std::optional<std::variant<std::unique_"
                        "ptr<TestLangNameExpression>, "
                        "std::unique_ptr<TestLangNamePattern>>>>*",
      .cc_const_getter_type = "const "
                              "std::vector<std::optional<std::variant<std::"
                              "unique_ptr<TestLangNameExpression>"
                              ", std::unique_ptr<TestLangNamePattern>>>>*",
      .cc_lang_name = "TestLangName",
      .td_types =
          {
              {
                  {FIELD_KIND_RVAL, "Variadic<AnyType>"},
              },
          },
      .cc_mlir_builder_type =
          {
              {FIELD_KIND_RVAL, "std::vector<mlir::Value>"},
          },
      .cc_mlir_getter_type =
          {
              {FIELD_KIND_RVAL, "mlir::OperandRange"},
          },
  });
}

TEST(TypeTest, IsABuiltinType) {
  TypePb type_pb;
  MALDOCA_ASSERT_OK(maldoca::ParseTextProto(R"pb(
    bool {}
  )pb", "TypePb for IsABuiltinType", &type_pb));

  MALDOCA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Type> type,
                               FromTypePb(type_pb, "TestLangName"));

  EXPECT_TRUE(type->IsA<BuiltinType>());

  EXPECT_TRUE(type->IsA<ScalarType>());
  EXPECT_TRUE(type->IsA<NonListType>());
  EXPECT_TRUE(type->IsA<Type>());

  EXPECT_FALSE(type->IsA<ClassType>());
  EXPECT_FALSE(type->IsA<VariantType>());
  EXPECT_FALSE(type->IsA<ListType>());
}

TEST(TypeTest, IsAClassType) {
  TypePb type_pb;
  MALDOCA_ASSERT_OK(maldoca::ParseTextProto(R"pb(
    class: "Expression"
  )pb", "TypePb for IsAClassType", &type_pb));

  MALDOCA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Type> type,
                               FromTypePb(type_pb, "TestLangName"));

  EXPECT_TRUE(type->IsA<ClassType>());

  EXPECT_TRUE(type->IsA<ScalarType>());
  EXPECT_TRUE(type->IsA<NonListType>());
  EXPECT_TRUE(type->IsA<Type>());

  EXPECT_FALSE(type->IsA<BuiltinType>());
  EXPECT_FALSE(type->IsA<VariantType>());
  EXPECT_FALSE(type->IsA<ListType>());
}

TEST(TypeTest, IsAVariantType) {
  TypePb type_pb;
  MALDOCA_ASSERT_OK(maldoca::ParseTextProto(R"pb(
                                              variant {
                                                types { bool {} }
                                                types { string {} }
                                              }
                                            )pb",
                                            "TypePb for IsAVariantType",
                                            &type_pb));

  MALDOCA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Type> type,
                               FromTypePb(type_pb, "TestLangName"));

  EXPECT_TRUE(type->IsA<VariantType>());

  EXPECT_TRUE(type->IsA<NonListType>());
  EXPECT_TRUE(type->IsA<Type>());

  EXPECT_FALSE(type->IsA<BuiltinType>());
  EXPECT_FALSE(type->IsA<ClassType>());
  EXPECT_FALSE(type->IsA<ScalarType>());
  EXPECT_FALSE(type->IsA<ListType>());
}

TEST(TypeTest, IsAListType) {
  TypePb type_pb;
  MALDOCA_ASSERT_OK(maldoca::ParseTextProto(R"pb(
                                              list { element_type { bool {} } }
                                            )pb",
                                            "TypePb for IsAListType",
                                            &type_pb));

  MALDOCA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Type> type,
                               FromTypePb(type_pb, "TestLangName"));

  EXPECT_TRUE(type->IsA<ListType>());

  EXPECT_TRUE(type->IsA<Type>());

  EXPECT_FALSE(type->IsA<BuiltinType>());
  EXPECT_FALSE(type->IsA<ClassType>());
  EXPECT_FALSE(type->IsA<ScalarType>());
  EXPECT_FALSE(type->IsA<VariantType>());
  EXPECT_FALSE(type->IsA<NonListType>());
}

TEST(TypeTest, EmptyTypeIsInvalid) {
  TypePb type_pb;
  MALDOCA_ASSERT_OK(maldoca::ParseTextProto("", "empty TypePb", &type_pb));
  EXPECT_THAT(FromTypePb(type_pb, "TestLangName"),
              maldoca::testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                         "Invalid TypePb: KIND_NOT_SET."));

  MALDOCA_ASSERT_OK(maldoca::ParseTextProto(R"pb(
    variant {}
  )pb", "empty variant", &type_pb));
  EXPECT_THAT(FromTypePb(type_pb, "TestLangName"),
              maldoca::testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                         "Empty variant type."));

  MALDOCA_ASSERT_OK(maldoca::ParseTextProto(R"pb(
    variant { types {} }
  )pb", "variant with empty type", &type_pb));
  EXPECT_THAT(FromTypePb(type_pb, "TestLangName"),
              maldoca::testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  "Invalid variant element type: KIND_NOT_SET."));

  MALDOCA_ASSERT_OK(maldoca::ParseTextProto(R"pb(
    list { element_type {} }
  )pb", "list with empty element type", &type_pb));
  EXPECT_THAT(
      FromTypePb(type_pb, "TestLangName"),
      maldoca::testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                 "Invalid list element type: KIND_NOT_SET."));
}

}  // namespace
}  // namespace maldoca
