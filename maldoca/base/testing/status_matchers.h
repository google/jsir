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
// Defines the following utilities not found in absl:
//
//   =========================================
//   MALDOCA_ASSERT_OK_AND_ASSIGN(lhs, rexpr)
//   =========================================
//
//   Executes an expression that returns an absl::StatusOr, and assigns the
//   contained variable to lhs if the error code is OK.
//   If the Status is non-OK, generates a test failure and returns from the
//   current function, which must have a void return type.

#ifndef MALDOCA_BASE_TESTING_STATUS_MATCHERS_H_
#define MALDOCA_BASE_TESTING_STATUS_MATCHERS_H_

#include <string_view>

#include "gmock/gmock.h"
#include "absl/status/status_builder.h"
#include "absl/status/status_macros.h"

namespace maldoca {
namespace testing {
namespace internal_status {

void AddFatalFailure(std::string_view expression,
                     const absl::StatusBuilder& builder);

}  // namespace internal_status

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
//   ABSL_ASSERT_OK(status_or_value.status());
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

}  // namespace testing
}  // namespace maldoca

#endif  // MALDOCA_BASE_TESTING_STATUS_MATCHERS_H_
