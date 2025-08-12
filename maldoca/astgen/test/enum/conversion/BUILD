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
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_applicable_licenses = ["//:license"])

licenses(["notice"])

exports_files([
    "ast_to_eir.generated.cc",
    "eir_to_ast.generated.cc",
])

cc_library(
    name = "ast_to_eir",
    srcs = ["ast_to_eir.generated.cc"],
    hdrs = ["ast_to_eir.h"],
    deps = [
        "//maldoca/astgen/test/enum:ast",
        "//maldoca/astgen/test/enum:ir",
        "@abseil-cpp//absl/cleanup",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/types:optional",
        "@abseil-cpp//absl/types:variant",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "eir_to_ast",
    srcs = ["eir_to_ast.generated.cc"],
    hdrs = ["eir_to_ast.h"],
    deps = [
        "//maldoca/astgen/test/enum:ast",
        "//maldoca/astgen/test/enum:ir",
        "//maldoca/base:status",
        "@abseil-cpp//absl/cleanup",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:optional",
        "@abseil-cpp//absl/types:variant",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_test(
    name = "conversion_test",
    srcs = ["conversion_test.cc"],
    deps = [
        ":ast_to_eir",
        ":eir_to_ast",
        "//maldoca/astgen/test:conversion_test_util",
        "//maldoca/astgen/test/enum:ast",
        "//maldoca/astgen/test/enum:ir",
        "@googletest//:gtest_main",
    ],
)
