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
        ":dataflow_analysis",
        "//maldoca/base:ret_check",
        "//maldoca/base:status",
        "//maldoca/js/ast",
        "//maldoca/js/babel",
        "//maldoca/js/babel:babel_cc_proto",
        "//maldoca/js/driver:driver_cc_proto",
        "//maldoca/js/ir",
        "//maldoca/js/ir/analyses/constant_propagation:analysis",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "state",
    hdrs = ["state.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
    ],
)

cc_library(
    name = "dataflow_analysis",
    srcs = ["dataflow_analysis.cc"],
    hdrs = ["dataflow_analysis.h"],
    deps = [
        ":jump_env",
        ":state",
        "//maldoca/js/ast",
        "//maldoca/js/ir",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/log:check",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "conditional_forward_dataflow_analysis",
    hdrs = ["conditional_forward_dataflow_analysis.h"],
    deps = [
        ":dataflow_analysis",
        "//maldoca/js/ir",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "per_var_state",
    hdrs = ["per_var_state.h"],
    deps = [
        ":scope",
        ":state",
        "//maldoca/js/ast",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
    ],
)

cc_test(
    name = "per_var_state_test",
    srcs = ["per_var_state_test.cc"],
    deps = [
        ":dataflow_analysis",
        ":per_var_state",
        "//maldoca/js/ast",
        "//maldoca/js/babel:babel_cc_proto",
        "@googletest//:gtest_main",
        "@llvm-project//mlir:Analysis",
    ],
)

cc_library(
    name = "conditional_forward_per_var_dataflow_analysis",
    hdrs = ["conditional_forward_per_var_dataflow_analysis.h"],
    deps = [
        ":conditional_forward_dataflow_analysis",
        ":dataflow_analysis",
        ":per_var_state",
        ":scope",
        "//maldoca/js/ast",
        "//maldoca/js/babel:babel_cc_proto",
        "//maldoca/js/ir",
        "@abseil-cpp//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "scope",
    srcs = ["scope.cc"],
    hdrs = ["scope.h"],
    deps = [
        "//maldoca/js/ast",
        "//maldoca/js/babel:babel_cc_proto",
        "//maldoca/js/babel:scope",
        "//maldoca/js/ir",
        "@abseil-cpp//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "jump_env",
    hdrs = ["jump_env.h"],
    deps = [
        "@abseil-cpp//absl/cleanup",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
    ],
)
