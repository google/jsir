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

#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "maldoca/astgen/test/typed_lambda/ast.generated.h"
#include "maldoca/base/testing/status_matchers.h"

namespace maldoca {
namespace astgen {
namespace {

// Even though JSON only has one "number" type, nlohmann::json tries to infer
// the most appropriate type. Here it detects that "1" is an integer.
TEST(TypedLambdaAstTest, LiteralInteger) {
  constexpr char kJsonString[] = R"(
    {
      "type": "Literal",
      "value": 1
    }
  )";

  nlohmann::json json = nlohmann::json::parse(kJsonString, /*cb=*/nullptr,
                                              /*allow_exceptions=*/false);

  MALDOCA_ASSERT_OK_AND_ASSIGN(auto expr, TlExpression::FromJson(json));

  auto *literal = dynamic_cast<const TlLiteral *>(expr.get());
  ASSERT_NE(literal, nullptr);

  EXPECT_TRUE(std::holds_alternative<int64_t>(literal->value()));
}

// Even though JSON only has one "number" type, nlohmann::json tries to infer
// the most appropriate type. Here it detects that "1.0" is a double.
TEST(TypedLambdaAstTest, LiteralDouble) {
  constexpr char kJsonString[] = R"(
    {
      "type": "Literal",
      "value": 1.0
    }
  )";

  nlohmann::json json = nlohmann::json::parse(kJsonString, /*cb=*/nullptr,
                                              /*allow_exceptions=*/false);

  MALDOCA_ASSERT_OK_AND_ASSIGN(auto expr, TlExpression::FromJson(json));

  auto *literal = dynamic_cast<const TlLiteral *>(expr.get());
  ASSERT_NE(literal, nullptr);

  EXPECT_TRUE(std::holds_alternative<double>(literal->value()));
}

}  // namespace
}  // namespace astgen
}  // namespace maldoca
