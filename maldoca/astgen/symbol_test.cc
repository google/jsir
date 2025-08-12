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

#include "maldoca/astgen/symbol.h"

#include "gtest/gtest.h"

namespace maldoca {
namespace {

TEST(SymbolTest, FromPascalCase) {
  Symbol symbol("GetLeftHandSide");
  EXPECT_EQ(symbol.ToPascalCase(), "GetLeftHandSide");
  EXPECT_EQ(symbol.ToCamelCase(), "getLeftHandSide");
  EXPECT_EQ(symbol.ToSnakeCase(), "get_left_hand_side");
}

TEST(SymbolTest, FromCamelCase) {
  Symbol symbol("getLeftHandSide");
  EXPECT_EQ(symbol.ToPascalCase(), "GetLeftHandSide");
  EXPECT_EQ(symbol.ToCamelCase(), "getLeftHandSide");
  EXPECT_EQ(symbol.ToSnakeCase(), "get_left_hand_side");
}

TEST(SymbolTest, FromSnakeCase) {
  Symbol symbol("get_left_hand_side");
  EXPECT_EQ(symbol.ToPascalCase(), "GetLeftHandSide");
  EXPECT_EQ(symbol.ToCamelCase(), "getLeftHandSide");
  EXPECT_EQ(symbol.ToSnakeCase(), "get_left_hand_side");
}

TEST(SymbolTest, ExtraUnderscoresAreIgnored) {
  Symbol symbol("_get_left_hand_side");
  EXPECT_EQ(symbol.ToPascalCase(), "GetLeftHandSide");
  EXPECT_EQ(symbol.ToCamelCase(), "getLeftHandSide");
  EXPECT_EQ(symbol.ToSnakeCase(), "get_left_hand_side");

  symbol = Symbol("get_left_hand_side_");
  EXPECT_EQ(symbol.ToPascalCase(), "GetLeftHandSide_");
  EXPECT_EQ(symbol.ToCamelCase(), "getLeftHandSide_");
  EXPECT_EQ(symbol.ToSnakeCase(), "get_left_hand_side_");

  symbol = Symbol("get__left_hand_side");
  EXPECT_EQ(symbol.ToPascalCase(), "GetLeftHandSide");
  EXPECT_EQ(symbol.ToCamelCase(), "getLeftHandSide");
  EXPECT_EQ(symbol.ToSnakeCase(), "get_left_hand_side");
}

TEST(SymbolTest, ConcatenateSymbols) {
  Symbol first("get_left");
  Symbol second("HandSide");
  Symbol symbol = first + second;
  EXPECT_EQ(symbol.ToPascalCase(), "GetLeftHandSide");
  EXPECT_EQ(symbol.ToCamelCase(), "getLeftHandSide");
  EXPECT_EQ(symbol.ToSnakeCase(), "get_left_hand_side");
}

TEST(SymbolTest, ConcatenateSymbolWithString) {
  Symbol first("get_left");
  auto second = "HandSide";
  Symbol symbol = first + second;
  EXPECT_EQ(symbol.ToPascalCase(), "GetLeftHandSide");
  EXPECT_EQ(symbol.ToCamelCase(), "getLeftHandSide");
  EXPECT_EQ(symbol.ToSnakeCase(), "get_left_hand_side");
}

TEST(SymbolTest, AvoidCppKeyword) {
  Symbol symbol("operator");
  EXPECT_EQ(symbol.ToCcVarName(), "operator_");
  EXPECT_EQ(Symbol(symbol.ToCcVarName()).ToPascalCase(), "Operator_");
  EXPECT_EQ((Symbol("get") + symbol.ToCcVarName() + "attr").ToCamelCase(),
            "getOperator_Attr");
  EXPECT_EQ((Symbol("get") + symbol.ToCcVarName() + "attr").ToSnakeCase(),
            "get_operator__attr");
}

}  // namespace
}  // namespace maldoca
