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

#include "maldoca/js/ir/analyses/dynamic_constant_propagation/dynamic_prelude.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "maldoca/js/quickjs/quickjs.h"
#include "maldoca/js/quickjs_babel/quickjs_babel.h"
#include "maldoca/js/quickjs_babel/quickjs_babel.h"
#include "quickjs/quickjs.h"
#include "maldoca/base/testing/status_matchers.h"

namespace maldoca {
namespace {

using ::testing::UnorderedElementsAreArray;

TEST(DynamicPreludeTest, Create) {
  JsirAnalysisConfig::DynamicConstantPropagation config;
  config.set_prelude_source(R"js(
    const const_variable = 42;
    let let_variable = 42;
    var var_variable = 42;
    function foo() {
      return 42;
    }
  )js");

  QuickJsBabel babel;
  MALDOCA_ASSERT_OK_AND_ASSIGN(DynamicPrelude prelude,
                       DynamicPrelude::Create(config, babel));

  // TODO(tzx): Include `const` and `let` variables in the prelude. One way of
  // doing this is to pre-process the prelude source and replace all `const` and
  // `let` declarations with `var`.
  EXPECT_THAT(prelude.GetPreludeSymbols(),
              UnorderedElementsAreArray({"var_variable", "foo"}));
}

TEST(DynamicPreludeTest, RunFunction) {
  JsirAnalysisConfig::DynamicConstantPropagation config;
  config.set_prelude_source(R"js(
    function add(a, b) {
      return a + b;
    }
  )js");

  QuickJsBabel babel;
  MALDOCA_ASSERT_OK_AND_ASSIGN(DynamicPrelude prelude,
                       DynamicPrelude::Create(config, babel));

  ASSERT_THAT(prelude.GetPreludeSymbols(), UnorderedElementsAreArray({"add"}));

  std::optional<QjsValue> qjs_function = prelude.GetFunction("add");
  ASSERT_TRUE(qjs_function.has_value());

  JSContext *qjs_context = prelude.GetQjsContext();

  QjsValue qjs_global{qjs_context, JS_GetGlobalObject(qjs_context)};

  std::vector<QjsValue> qjs_arguments = {
      QjsValue{qjs_context, JS_NewInt32(qjs_context, 1)},
      QjsValue{qjs_context, JS_NewInt32(qjs_context, 2)},
  };

  std::vector<JSValue> qjs_argument_refs;
  qjs_argument_refs.reserve(qjs_arguments.size());
  for (const auto &qjs_argument : qjs_arguments) {
    qjs_argument_refs.push_back(qjs_argument.get());
  }

  QjsValue qjs_result{
      qjs_context,
      JS_Call(qjs_context, /*func_obj=*/qjs_function->get(),
              /*this_obj=*/qjs_global.get(), qjs_argument_refs.size(),
              qjs_argument_refs.data()),
  };

  int32_t result;
  EXPECT_EQ(JS_ToInt32(qjs_context, &result, qjs_result.get()), 0);
  EXPECT_EQ(result, 3);
}

}  // namespace
}  // namespace maldoca
