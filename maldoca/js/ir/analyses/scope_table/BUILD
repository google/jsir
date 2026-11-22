# Copyright 2026 Google LLC
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

# JSIR scopes and declarations, registered by the code that creates them
# (JsirScopeRegistry, an MLIR analysis holding one JsirScopeTable per scope).

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
    name = "jsir_scope_table",
    srcs = ["jsir_scope_table.cc"],
    hdrs = ["jsir_scope_table.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "jsir_scope_registry",
    srcs = ["jsir_scope_registry.cc"],
    hdrs = ["jsir_scope_registry.h"],
    deps = [
        ":jsir_scope_table",
        "//maldoca/js/ir",
        "//maldoca/js/ir:jsir_utils",
        "@abseil-cpp//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "jsir_scope_table_test",
    srcs = ["jsir_scope_table_test.cc"],
    deps = [
        ":jsir_scope_table",
        "@googletest//:gtest_main",
        "@llvm-project//mlir:IR",
    ],
)

cc_test(
    name = "jsir_scope_registry_test",
    srcs = ["jsir_scope_registry_test.cc"],
    deps = [
        ":jsir_scope_registry",
        ":jsir_scope_table",
        "//maldoca/js/ir",
        "@googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)
