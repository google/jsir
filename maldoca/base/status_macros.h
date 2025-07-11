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

#ifndef MALDOCA_BASE_STATUS_MACROS_H_
#define MALDOCA_BASE_STATUS_MACROS_H_

#include <utility>

#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "maldoca/base/source_location.h"
#include "maldoca/base/status_builder.h"  // IWYU pragma: export

// Evaluates an expression that produces a `absl::Status`. If the status is not
// ok, returns it from the current function.
//
// For example:
//   absl::Status MultiStepFunction() {
//     MALDOCA_RETURN_IF_ERROR(Function(args...));
//     MALDOCA_RETURN_IF_ERROR(foo.Method(args...));
//     return absl::OkStatus();
//   }
//
// The macro ends with a `StatusBuilder` which allows the returned status
// to be extended with more details.  Any chained expressions after the macro
// will not be evaluated unless there is an error.
//
// For example:
//   absl::Status MultiStepFunction() {
//     MALDOCA_RETURN_IF_ERROR(Function(args...)) << "in MultiStepFunction";
//     MALDOCA_RETURN_IF_ERROR(foo.Method(args...))
//         << "while processing query: " << query.DebugString();
//     return absl::OkStatus();
//   }
//
// If using this macro inside a lambda, you need to annotate the return type
// to avoid confusion between a `StatusBuilder` and an `absl::Status` type.
// E.g.
//
//   []() -> absl::Status {
//     MALDOCA_RETURN_IF_ERROR(Function(args...));
//     MALDOCA_RETURN_IF_ERROR(foo.Method(args...));
//     return absl::OkStatus();
//   }
#define MALDOCA_RETURN_IF_ERROR(expr)                              \
  MALDOCA_STATUS_MACROS_IMPL_ELSE_BLOCKER_                         \
  if (::maldoca::status_macro_internal::StatusAdaptorForMacros     \
          status_macro_internal_adaptor = {(expr), MALDOCA_LOC}) { \
  } else /* NOLINT */                                              \
    return status_macro_internal_adaptor.Consume()

// Executes an expression `rexpr` that returns a `StatusOr<T>`. On OK, moves its
// value into the variable defined by `lhs`, otherwise returns from the current
// function. By default the error status is returned unchanged, but it may be
// modified by an `error_expression`. If there is an error, `lhs` is not
// evaluated; thus any side effects that `lhs` may have only occur in the
// success case.
//
// Interface:
//
//   MALDOCA_ASSIGN_OR_RETURN(lhs, rexpr)
//   MALDOCA_ASSIGN_OR_RETURN(lhs, rexpr, error_expression);
//
// WARNING: if lhs is parenthesized, the parentheses are removed. See examples
// for more details.
//
// WARNING: expands into multiple statements; it cannot be used in a single
// statement (e.g. as the body of an if statement without {})!
//
// Example: Declaring and initializing a new variable (ValueType can be anything
//          that can be initialized with assignment, including references):
//   MALDOCA_ASSIGN_OR_RETURN(ValueType value, MaybeGetValue(arg));
//
// Example: Assigning to an existing variable:
//   ValueType value;
//   MALDOCA_ASSIGN_OR_RETURN(value, MaybeGetValue(arg));
//
// Example: Assigning to an expression with side effects:
//   MyProto data;
//   MALDOCA_ASSIGN_OR_RETURN(*data.mutable_str(), MaybeGetValue(arg));
//   // No field "str" is added on error.
//
// Example: Assigning to a std::unique_ptr.
//   MALDOCA_ASSIGN_OR_RETURN(std::unique_ptr<T> ptr, MaybeGetPtr(arg));
//
// Example: Assigning to a map. Because of C preprocessor
// limitation, the type used in MALDOCA_ASSIGN_OR_RETURN cannot contain comma,
// so wrap lhs in parentheses:
//   MALDOCA_ASSIGN_OR_RETURN((absl::flat_hash_map<Foo, Bar> my_map), GetMap());
// Or use auto if the type is obvious enough:
//   MALDOCA_ASSIGN_OR_RETURN(const auto& my_map, GetMapRef());
//
// Example: Assigning to structured bindings. The same situation with comma as
// in map, so wrap the statement in parentheses.
//   MALDOCA_ASSIGN_OR_RETURN((const auto& [first, second]), GetPair());
//
// If passed, the `error_expression` is evaluated to produce the return
// value. The expression may reference any variable visible in scope, as
// well as a `StatusBuilder` object populated with the error and named by a
// single underscore `_`. The expression typically uses the builder to modify
// the status and is returned directly in manner similar to
// MALDOCA_RETURN_IF_ERROR. The expression may, however, evaluate to any type
// returnable by the function, including (void). For example:
//
// Example: Adjusting the error message.
//   MALDOCA_ASSIGN_OR_RETURN(ValueType value, MaybeGetValue(query),
//                        _ << "while processing " << query.DebugString());
//
// Example: Logging the error on failure.
//   MALDOCA_ASSIGN_OR_RETURN(ValueType value, MaybeGetValue(query),
//   _.LogError());
//
#define MALDOCA_ASSIGN_OR_RETURN(...)                               \
  MALDOCA_STATUS_MACROS_IMPL_GET_VARIADIC_(                         \
      (__VA_ARGS__, MALDOCA_STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_3_, \
       MALDOCA_STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_2_))             \
  (__VA_ARGS__)

// =================================================================
// == Implementation details, do not rely on anything below here. ==
// =================================================================

// MSVC incorrectly expands variadic macros, splice together a macro call to
// work around the bug.
#define MALDOCA_STATUS_MACROS_IMPL_GET_VARIADIC_HELPER_(_1, _2, _3, NAME, ...) \
  NAME
#define MALDOCA_STATUS_MACROS_IMPL_GET_VARIADIC_(args) \
  MALDOCA_STATUS_MACROS_IMPL_GET_VARIADIC_HELPER_ args

#define MALDOCA_STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_2_(lhs, rexpr) \
  MALDOCA_STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_3_(lhs, rexpr, std::move(_))
#define MALDOCA_STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_3_(lhs, rexpr,         \
                                                       error_expression)   \
  MALDOCA_STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_(                            \
      MALDOCA_STATUS_MACROS_IMPL_CONCAT_(_status_or_value, __LINE__), lhs, \
      rexpr, error_expression)
#define MALDOCA_STATUS_MACROS_IMPL_ASSIGN_OR_RETURN_(statusor, lhs, rexpr, \
                                                     error_expression)     \
  auto statusor = (rexpr);                                                 \
  if (ABSL_PREDICT_FALSE(!statusor.ok())) {                                \
    ::maldoca::StatusBuilder _(std::move(statusor).status(), MALDOCA_LOC); \
    (void)_; /* error_expression is allowed to not use this variable */    \
    return (error_expression);                                             \
  }                                                                        \
  {                                                                        \
    static_assert(                                                         \
        #lhs[0] != '(' || #lhs[sizeof(#lhs) - 2] != ')' ||                 \
            !::maldoca::status_macro_internal::                            \
                HasPotentialConditionalOperator(#lhs, sizeof(#lhs) - 2),   \
        "Identified potential conditional operator, consider not "         \
        "using MALDOCA_ASSIGN_OR_RETURN");                                 \
  }                                                                        \
  MALDOCA_STATUS_MACROS_IMPL_UNPARENTHESIZE_IF_PARENTHESIZED(lhs) =        \
      std::move(statusor).value()

// Internal helpers for macro expansion.
#define MALDOCA_STATUS_MACROS_IMPL_EAT(...)
#define MALDOCA_STATUS_MACROS_IMPL_REM(...) __VA_ARGS__
#define MALDOCA_STATUS_MACROS_IMPL_EMPTY()

// __VA_OPT__ expands to nothing if __VA_ARGS__ are empty, and otherwise expands
// to its argument. We use __VA_OPT__ here to expand to true if __VA_ARGS__ is
// empty and false otherwise- the `EMPTY_I` helper macro expands to the first
// argument.
#define MALDOCA_STATUS_MACROS_IMPL_IS_EMPTY(...) \
  MALDOCA_STATUS_MACROS_IMPL_IS_EMPTY_I(__VA_OPT__(0, ) 1)
#define MALDOCA_STATUS_MACROS_IMPL_IS_EMPTY_I(is_empty, ...) is_empty

// Internal helpers for if statement.
#define MALDOCA_STATUS_MACROS_IMPL_IF_1(_Then, _Else) _Then
#define MALDOCA_STATUS_MACROS_IMPL_IF_0(_Then, _Else) _Else
#define MALDOCA_STATUS_MACROS_IMPL_IF(_Cond, _Then, _Else)                  \
  MALDOCA_STATUS_MACROS_IMPL_CONCAT_(MALDOCA_STATUS_MACROS_IMPL_IF_, _Cond) \
  (_Then, _Else)

// Expands to 1 if the input is parenthesized. Otherwise expands to 0.
#define MALDOCA_STATUS_MACROS_IMPL_IS_PARENTHESIZED(...) \
  MALDOCA_STATUS_MACROS_IMPL_IS_EMPTY(                   \
      MALDOCA_STATUS_MACROS_IMPL_EAT __VA_ARGS__)

// If the input is parenthesized, removes the parentheses. Otherwise expands to
// the input unchanged.
#define MALDOCA_STATUS_MACROS_IMPL_UNPARENTHESIZE_IF_PARENTHESIZED(...)   \
  MALDOCA_STATUS_MACROS_IMPL_IF(                                          \
      MALDOCA_STATUS_MACROS_IMPL_IS_PARENTHESIZED(__VA_ARGS__),           \
      MALDOCA_STATUS_MACROS_IMPL_REM, MALDOCA_STATUS_MACROS_IMPL_EMPTY()) \
  __VA_ARGS__

// Internal helper for concatenating macro values.
#define MALDOCA_STATUS_MACROS_IMPL_CONCAT_INNER_(x, y) x##y
#define MALDOCA_STATUS_MACROS_IMPL_CONCAT_(x, y) \
  MALDOCA_STATUS_MACROS_IMPL_CONCAT_INNER_(x, y)

// The GNU compiler emits a warning for code like:
//
//   if (foo)
//     if (bar) { } else baz;
//
// because it thinks you might want the else to bind to the first if.  This
// leads to problems with code like:
//
//   if (do_expr) MALDOCA_RETURN_IF_ERROR(expr) << "Some message";
//
// The "switch (0) case 0:" idiom is used to suppress this.
#define MALDOCA_STATUS_MACROS_IMPL_ELSE_BLOCKER_ \
  switch (0)                                     \
  case 0:                                        \
  default:  // NOLINT

namespace maldoca {
namespace status_macro_internal {

// Some builds do not support C++14 fully yet, using C++11 constexpr technique.
constexpr bool HasPotentialConditionalOperator(const char* lhs, int index) {
  return (index == -1 ? false
                      : (lhs[index] == '?' ? true
                                           : HasPotentialConditionalOperator(
                                                 lhs, index - 1)));
}

// Provides a conversion to bool so that it can be used inside an if statement
// that declares a variable.
class StatusAdaptorForMacros {
 public:
  StatusAdaptorForMacros(const absl::Status& status, SourceLocation loc)
      : builder_(status, loc) {}

  StatusAdaptorForMacros(absl::Status&& status, SourceLocation loc)
      : builder_(std::move(status), loc) {}

  StatusAdaptorForMacros(const StatusBuilder& builder, SourceLocation loc)
      : builder_(builder) {}

  StatusAdaptorForMacros(StatusBuilder&& builder, SourceLocation loc)
      : builder_(std::move(builder)) {}

  StatusAdaptorForMacros(const StatusAdaptorForMacros&) = delete;
  StatusAdaptorForMacros& operator=(const StatusAdaptorForMacros&) = delete;

  explicit operator bool() const { return ABSL_PREDICT_TRUE(builder_.ok()); }

  StatusBuilder&& Consume() { return std::move(builder_); }

 private:
  StatusBuilder builder_;
};

}  // namespace status_macro_internal
}  // namespace maldoca

#endif  // MALDOCA_BASE_STATUS_MACROS_H_
