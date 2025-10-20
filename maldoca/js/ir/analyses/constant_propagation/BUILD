# Copyright 2025 Google LLC
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
load("@rules_cc//cc:cc_test.bzl", "cc_test")

licenses(["notice"])

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//maldoca/js:__subpackages__",
    ],
)

cc_library(
    name = "analysis",
    srcs = ["analysis.cc"],
    hdrs = ["analysis.h"],
    deps = [
        "//maldoca/js/ast",
        "//maldoca/js/babel:babel_cc_proto",
        "//maldoca/js/ir",
        "//maldoca/js/ir/analyses:conditional_forward_per_var_dataflow_analysis",
        "//maldoca/js/ir/analyses:dataflow_analysis",
        "//maldoca/js/ir/analyses:per_var_state",
        "//maldoca/js/ir/analyses:scope",
        "//maldoca/js/ir/analyses:state",
        "//maldoca/js/quickjs",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@quickjs//:libquickjs",
    ],
)

cc_test(
    name = "analysis_test",
    srcs = ["analysis_test.cc"],
    deps = [
        ":analysis",
        "@googletest//:gtest_main",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
    ],
)
