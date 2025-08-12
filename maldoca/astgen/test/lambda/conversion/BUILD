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
    "ast_to_lair.generated.cc",
])

cc_library(
    name = "ast_to_lair",
    srcs = ["ast_to_lair.generated.cc"],
    hdrs = ["ast_to_lair.h"],
    deps = [
        "//maldoca/astgen/test/lambda:ast",
        "//maldoca/astgen/test/lambda:ir",
        "@abseil-cpp//absl/cleanup",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
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
        ":ast_to_lair",
        "//maldoca/astgen/test:conversion_test_util",
        "//maldoca/astgen/test/lambda:ast",
        "//maldoca/astgen/test/lambda:ir",
        "@googletest//:gtest_main",
    ],
)
