// Copyright 2025 Google LLC
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

#include "maldoca/js/ir/analyses/constant_propagation/analysis.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

namespace maldoca {

TEST(AnalysisTest, JoinUnknownConstant) {
  mlir::MLIRContext context;
  mlir::OpBuilder builder(&context);

  auto one_float = builder.getF64FloatAttr(1.0);

  auto unknown = JsirConstantPropagationValue::Unknown();
  auto one = JsirConstantPropagationValue(one_float);
  EXPECT_EQ(unknown.Join(one), mlir::ChangeResult::NoChange);
  EXPECT_TRUE(unknown.IsUnknown());
}

TEST(AnalysisTest, JoinUninitializedConstant) {
  mlir::MLIRContext context;
  mlir::OpBuilder builder(&context);

  auto two_float = builder.getF64FloatAttr(2.0);

  auto uninitialized = JsirConstantPropagationValue::Uninitialized();
  auto two = JsirConstantPropagationValue(two_float);
  EXPECT_EQ(uninitialized.Join(two), mlir::ChangeResult::Change);
  EXPECT_TRUE(!uninitialized.IsUninitialized() && !uninitialized.IsUnknown());
  EXPECT_EQ(*uninitialized, two_float);
}

TEST(AnalysisTest, JoinUninitializedUninitialized) {
  auto uninit = JsirConstantPropagationValue::Uninitialized();
  EXPECT_EQ(uninit.Join(uninit), mlir::ChangeResult::NoChange);
  EXPECT_TRUE(uninit.IsUninitialized());
}

TEST(AnalysisTest, JoinUnknownUnknown) {
  auto unknown = JsirConstantPropagationValue::Unknown();
  EXPECT_EQ(unknown.Join(unknown), mlir::ChangeResult::NoChange);
  EXPECT_TRUE(unknown.IsUnknown());
}

TEST(AnalysisTest, JoinConstantConstant) {
  mlir::MLIRContext context;
  mlir::OpBuilder builder(&context);

  auto one_float = builder.getF64FloatAttr(1.0);
  auto two_float = builder.getF64FloatAttr(2.0);

  auto one = JsirConstantPropagationValue(one_float);
  auto two = JsirConstantPropagationValue(two_float);

  EXPECT_EQ(one.Join(two), mlir::ChangeResult::Change);
  EXPECT_TRUE(one.IsUnknown());
}

TEST(AnalysisTest, JoinConstantUnknown) {
  mlir::MLIRContext context;
  mlir::OpBuilder builder(&context);

  auto one_float = builder.getF64FloatAttr(1.0);

  auto one = JsirConstantPropagationValue(one_float);
  auto unknown = JsirConstantPropagationValue::Unknown();

  EXPECT_EQ(one.Join(unknown), mlir::ChangeResult::Change);
  EXPECT_TRUE(one.IsUnknown());
}

TEST(AnalysisTest, JoinConstantUninitialized) {
  mlir::MLIRContext context;
  mlir::OpBuilder builder(&context);

  auto two_float = builder.getF64FloatAttr(2.0);

  auto two = JsirConstantPropagationValue(two_float);
  auto uninit = JsirConstantPropagationValue::Uninitialized();

  EXPECT_EQ(two.Join(uninit), mlir::ChangeResult::NoChange);
  EXPECT_TRUE(!two.IsUninitialized() && !two.IsUnknown());
  EXPECT_EQ(*two, two_float);
}

}  // namespace maldoca
