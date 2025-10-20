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

#include "maldoca/js/ast/transforms/extract_prelude/pass.h"

#include <cstddef>
#include <optional>

#include "gtest/gtest.h"
#include "absl/container/flat_hash_set.h"
#include "absl/time/time.h"
#include "maldoca/base/testing/status_matchers.h"
#include "maldoca/js/babel/babel.pb.h"
#include "maldoca/js/driver/conversion.h"
#include "maldoca/js/driver/driver.h"
#include "maldoca/js/quickjs_babel/quickjs_babel.h"

namespace maldoca {
namespace {

static constexpr char kSource[] = R"js(
// 0:
// exec:begin
function foo() {
  console.log("foo");
}
// exec:end
// 1:
let a = 1;
// 2:
function bar() {
  console.log("bar");
}
)js";

TEST(ExtractPreludePassTest, ExtractPreludeByIndices) {
  absl::flat_hash_set<size_t> indices = {1};

  static constexpr char kExpectedPrelude[] = R"js(let a = 1;
)js";

  static constexpr char kExpectedSourceWithoutPrelude[] = R"js(// 0:
// exec:begin
function foo() {
  console.log("foo");
}
// exec:end
// 1:

// 2:
function bar() {
  console.log("bar");
})js";

  QuickJsBabel babel;

  BabelParseRequest parse_request;
  parse_request.set_compute_scopes(true);

  MALDOCA_ASSERT_OK_AND_ASSIGN(
      JsAstRepr repr, ToJsAstRepr::FromJsSourceRepr(kSource, parse_request,
                                                    absl::InfiniteDuration(),
                                                    std::nullopt, babel));

  JsirAnalysisConfig::DynamicConstantPropagation prelude =
      ExtractPreludeByIndices(kSource, indices, *repr.ast);

  MALDOCA_ASSERT_OK_AND_ASSIGN(
      JsSourceRepr source_repr,
      ToJsSourceRepr::FromJsAstRepr(*repr.ast, {}, absl::InfiniteDuration(),
                                    babel));

  EXPECT_EQ(source_repr.source, kExpectedSourceWithoutPrelude);

  EXPECT_EQ(prelude.prelude_source(), kExpectedPrelude);
  EXPECT_EQ(prelude.extracted_from_scope_uid(), 0);
}

TEST(ExtractPreludePassTest, ExtractPreludeByAnnotations) {
  static constexpr char kExpectedPrelude[] = R"js(function foo() {
  console.log("foo");
}
)js";

  static constexpr char kExpectedSourceWithoutPrelude[] = R"js(// exec:end
// 1:
let a = 1;
// 2:
function bar() {
  console.log("bar");
})js";

  QuickJsBabel babel;

  BabelParseRequest parse_request;
  parse_request.set_compute_scopes(true);

  MALDOCA_ASSERT_OK_AND_ASSIGN(
      JsAstRepr repr, ToJsAstRepr::FromJsSourceRepr(kSource, parse_request,
                                                    absl::InfiniteDuration(),
                                                    std::nullopt, babel));

  JsirAnalysisConfig::DynamicConstantPropagation prelude =
      ExtractPreludeByAnnotations(kSource, *repr.ast);

  MALDOCA_ASSERT_OK_AND_ASSIGN(
      JsSourceRepr source_repr,
      ToJsSourceRepr::FromJsAstRepr(*repr.ast, {}, absl::InfiniteDuration(),
                                    babel));

  EXPECT_EQ(source_repr.source, kExpectedSourceWithoutPrelude);

  EXPECT_EQ(prelude.prelude_source(), kExpectedPrelude);
  EXPECT_EQ(prelude.extracted_from_scope_uid(), 0);
}

TEST(ExtractPreludePassTest, ExtractPreludeByIndicesAndAnnotations) {
  absl::flat_hash_set<size_t> indices = {1};

  static constexpr char kExpectedPrelude[] = R"js(function foo() {
  console.log("foo");
}
let a = 1;
)js";

  static constexpr char kExpectedSourceWithoutPrelude[] = R"js(// 2:
function bar() {
  console.log("bar");
})js";

  QuickJsBabel babel;

  BabelParseRequest parse_request;
  parse_request.set_compute_scopes(true);

  MALDOCA_ASSERT_OK_AND_ASSIGN(
      JsAstRepr repr, ToJsAstRepr::FromJsSourceRepr(kSource, parse_request,
                                                    absl::InfiniteDuration(),
                                                    std::nullopt, babel));

  JsirAnalysisConfig::DynamicConstantPropagation prelude =
      ExtractPreludeByIndicesAndAnnotations(kSource, indices, *repr.ast);

  MALDOCA_ASSERT_OK_AND_ASSIGN(
      JsSourceRepr source_repr,
      ToJsSourceRepr::FromJsAstRepr(*repr.ast, {}, absl::InfiniteDuration(),
                                    babel));

  EXPECT_EQ(source_repr.source, kExpectedSourceWithoutPrelude);

  EXPECT_EQ(prelude.prelude_source(), kExpectedPrelude);
  EXPECT_EQ(prelude.extracted_from_scope_uid(), 0);
}

TEST(ExtractPreludePassTest, ReuseBabel) {
  static constexpr char kExpectedPrelude[] = R"js(function foo() {
  console.log("foo");
}
)js";

  QuickJsBabel babel;

  BabelParseRequest parse_request;
  parse_request.set_compute_scopes(true);

  // Parse the source code once so that Babel increments the scope uid counter.
  {
    MALDOCA_ASSERT_OK_AND_ASSIGN(
        JsAstRepr repr, ToJsAstRepr::FromJsSourceRepr(kSource, parse_request,
                                                      absl::InfiniteDuration(),
                                                      std::nullopt, babel));
  }

  // This time the global scope uid is not 0.
  MALDOCA_ASSERT_OK_AND_ASSIGN(
      JsAstRepr repr, ToJsAstRepr::FromJsSourceRepr(kSource, parse_request,
                                                    absl::InfiniteDuration(),
                                                    std::nullopt, babel));

  JsirAnalysisConfig::DynamicConstantPropagation prelude =
      ExtractPreludeByAnnotations(kSource, *repr.ast);

  EXPECT_EQ(prelude.prelude_source(), kExpectedPrelude);
  EXPECT_EQ(prelude.extracted_from_scope_uid(), 3);
}

}  // namespace
}  // namespace maldoca
