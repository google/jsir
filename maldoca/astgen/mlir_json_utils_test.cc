// Copyright 2026 Google LLC
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

#include "maldoca/astgen/mlir_json_utils.h"

#include <cstdint>
#include <limits>

#include "llvm/ADT/APInt.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"

namespace maldoca {
namespace {

TEST(MlirJsonUtilsTest, RoundTripsBabelExtra) {
  mlir::MLIRContext context;
  auto json = nlohmann::json::parse(R"json({
    "raw": "0x10",
    "rawValue": 16,
    "parenthesized": true,
    "parenStart": 3,
    "expressionValue": "\u00e9",
    "ratio": 1.5,
    "nothing": null,
    "list": [1, "a", false, {"nested": []}],
    "object": {}
  })json");

  mlir::DictionaryAttr attr = JsonToMlirDictionaryAttr(&context, json);
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr.size(), 9);

  auto result = MlirAttrToJson(attr);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(*result, json);
}

TEST(MlirJsonUtilsTest, NonObjectIsNotADictionary) {
  mlir::MLIRContext context;
  EXPECT_EQ(JsonToMlirDictionaryAttr(&context, nlohmann::json::array()),
            nullptr);
  EXPECT_EQ(JsonToMlirDictionaryAttr(&context, nlohmann::json("x")), nullptr);
}

TEST(MlirJsonUtilsTest, BoolIsNotAnInteger) {
  mlir::MLIRContext context;
  auto result = MlirAttrToJson(JsonToMlirAttr(&context, nlohmann::json(true)));
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->is_boolean());
}

TEST(MlirJsonUtilsTest, UnsignedIntegerStaysNonNegative) {
  mlir::MLIRContext context;
  mlir::Builder builder(&context);
  constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
  auto attr = builder.getIntegerAttr(
      builder.getIntegerType(64, /*isSigned=*/false), llvm::APInt(64, kMax));

  auto result = MlirAttrToJson(attr);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->get<uint64_t>(), kMax);
}

TEST(MlirJsonUtilsTest, IntegerWiderThan64BitsIsAnError) {
  mlir::MLIRContext context;
  mlir::Builder builder(&context);
  llvm::APInt value = llvm::APInt::getOneBitSet(128, 100);

  EXPECT_FALSE(
      MlirAttrToJson(builder.getIntegerAttr(builder.getIntegerType(128), value))
          .ok());
  EXPECT_FALSE(MlirAttrToJson(
                   builder.getIntegerAttr(
                       builder.getIntegerType(128, /*isSigned=*/false), value))
                   .ok());
}

}  // namespace
}  // namespace maldoca
