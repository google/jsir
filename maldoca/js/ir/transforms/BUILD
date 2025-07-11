# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//maldoca/js:allowed_users.bzl", "JSIR_ALLOWED_USERS")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = JSIR_ALLOWED_USERS,
)

cc_library(
    name = "transform",
    srcs = ["transform.cc"],
    hdrs = ["transform.h"],
    deps = [
        "//maldoca/base:ret_check",
        "//maldoca/base:status",
        "//maldoca/js/ast",
        "//maldoca/js/babel",
        "//maldoca/js/babel:babel_cc_proto",
        "//maldoca/js/driver:driver_cc_proto",
        "//maldoca/js/ir",
        "//maldoca/js/ir/conversion:utils",
        "//maldoca/js/ir/transforms/constant_propagation:pass",
        "//maldoca/js/ir/transforms/move_named_functions:pass",
        "//maldoca/js/ir/transforms/normalize_object_properties:pass",
        "//maldoca/js/ir/transforms/peel_parentheses:pass",
        "//maldoca/js/ir/transforms/remove_directives:pass",
        "//maldoca/js/ir/transforms/split_declaration_statements:pass",
        "//maldoca/js/ir/transforms/split_sequence_expressions:pass",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)
