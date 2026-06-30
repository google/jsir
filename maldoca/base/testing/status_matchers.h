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

// Testing utilities for working with absl::Status, absl::StatusOr.
//
// Defines the following utilities:
//
//   =================
//   MALDOCA_EXPECT_OK(s)
//
//   MALDOCA_ASSERT_OK(s)
//   =================
//   Convenience macros for `EXPECT_THAT(s, IsOk())`, where `s` is either
//   a `Status` or a `StatusOr<T>`.
//
//   There are no MALDOCA_EXPECT_NOT_OK/MALDOCA_ASSERT_NOT_OK macros since they
//   would not provide much value (when they fail, they would just print the OK
//   status which conveys no more information than EXPECT_FALSE(s.ok()); If you
//   want to check for particular errors, better alternatives are:
//   EXPECT_THAT(s, StatusIs(expected_error));
//   EXPECT_THAT(s, StatusIs(_, _, HasSubstr("expected error")));
//
//   ===============
//   IsOkAndHolds(m)
//   ===============
//
//   This gMock matcher matches a StatusOr<T> value whose status is OK
//   and whose inner value matches matcher m.  Example:
//
//     using ::testing::MatchesRegex;
//     using maldoca::testing::IsOkAndHolds;
//     ...
//     StatusOr<string> maybe_name = ...;
//     EXPECT_THAT(maybe_name, IsOkAndHolds(MatchesRegex("John .*")));
//
//   ===============================
//   StatusIs(status_code_matcher,
//            error_message_matcher)
//   ===============================
//
//   This gMock matcher matches a Status or StatusOr<T> value if
//   all of the following are true:
//
//     - the status' error_code() matches status_code_matcher, and
//     - the status' error_message() matches error_message_matcher.
//
//   Example:
//
//     using ::absl::StatusOr;
//     using ::testing::HasSubstr;
//     using ::testing::MatchesRegex;
//     using ::testing::Ne;
//     using ::testing::_;
//     using ::maldoca::testing::StatusIs;
//     StatusOr<string> GetName(int id);
//     ...
//
//     // The status code must be
//     // kServerError; the error message can be anything.
//     EXPECT_THAT(GetName(42),
//                 StatusIs(kServerError, _));
//     // The status code can be
//     // anything; the error message must match the regex.
//     EXPECT_THAT(GetName(43),
//                 StatusIs(_,
//                          MatchesRegex("server.*time-out")));
//
//     // The status code
//     // should not be kServerError; the error message can be
//     // anything with "client" in it.
//     EXPECT_CALL(mock_env, HandleStatus(
//         StatusIs(Ne(kServerError),
//                  HasSubstr("client"))));
//
//   ===============================
//   StatusIs(status_code_matcher)
//   ===============================
//
//   This is a shorthand for
//     StatusIs(status_code_matcher,
//              testing::_)
//   In other words, it's like the two-argument StatusIs(), except that it
//   ignores error message.
//
//   ===============
//   IsOk()
//   ===============
//
//   Matches a absl::Status or absl::StatusOr<T> value
//   whose status value is absl::OkStatus().
//   Equivalent to 'StatusIs(absl::StatusCode::kOk)'.
//   Example:
//     using maldoca::testing::IsOk;
//     ...
//     StatusOr<string> maybe_name = ...;
//     EXPECT_THAT(maybe_name, IsOk());
//     Status s = ...;
//     EXPECT_THAT(s, IsOk());

#ifndef MALDOCA_BASE_TESTING_STATUS_MATCHERS_H_
#define MALDOCA_BASE_TESTING_STATUS_MATCHERS_H_

#include <string_view>

#include "gmock/gmock.h"
#include "absl/status/status_builder.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"

namespace maldoca {
namespace testing {
namespace internal_status {

void AddFatalFailure(std::string_view expression,
                     const absl::StatusBuilder& builder);

}  // namespace internal_status

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;


// Macros for testing the results of functions that return absl::Status or
// absl::StatusOr<T> (for any type T).
#define MALDOCA_EXPECT_OK(expression) ABSL_EXPECT_OK(expression)
#define MALDOCA_ASSERT_OK(expression) ABSL_ASSERT_OK(expression)

// Executes an expression that returns a absl::StatusOr, and assigns the
// contained variable to lhs if the error code is OK.
// If the Status is non-OK, generates a test failure and returns from the
// current function, which must have a void return type.
//
// Example: Declaring and initializing a new value
//   MALDOCA_ASSERT_OK_AND_ASSIGN(const ValueType& value, MaybeGetValue(arg));
//
// Example: Assigning to an existing value
//   ValueType value;
//   MALDOCA_ASSERT_OK_AND_ASSIGN(value, MaybeGetValue(arg));
//
// The value assignment example would expand into something like:
//   auto status_or_value = MaybeGetValue(arg);
//   MALDOCA_ASSERT_OK(status_or_value.status());
//   value = std::move(status_or_value).value();
//
// WARNING: Like ABSL_ASSIGN_OR_RETURN, MALDOCA_ASSERT_OK_AND_ASSIGN expands
//   into multiple statements; it cannot be used in a single statement (e.g. as
//   the body of an if statement without {})!
#define MALDOCA_ASSERT_OK_AND_ASSIGN(lhs, rexpr)                              \
  ABSL_ASSIGN_OR_RETURN(/* NOLINT(clang-diagnostic-shadow) */                 \
                        lhs, rexpr,                                           \
                        ::maldoca::testing::internal_status::AddFatalFailure( \
                            #rexpr, _))

// Executes an expression that returns a absl::StatusOr, and compares the
// contained variable to rexpr if the error code is OK.
// If the Status is non-OK it generates a nonfatal test failure
#define MALDOCA_EXPECT_OK_AND_EQ(lhs, rexpr) \
  EXPECT_THAT(lhs, ::absl_testing::IsOkAndHolds(rexpr));

}  // namespace testing
}  // namespace maldoca

#endif  // MALDOCA_BASE_TESTING_STATUS_MATCHERS_H_
