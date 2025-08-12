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

licenses(["notice"])

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//:__subpackages__",
    ],
)

cc_library(
    name = "ast_gen_test_util",
    testonly = True,
    srcs = ["ast_gen_test_util.cc"],
    hdrs = ["ast_gen_test_util.h"],
    deps = [
        "//maldoca/astgen:ast_def",
        "//maldoca/astgen:ast_def_cc_proto",
        "//maldoca/astgen:ast_gen",
        "//maldoca/base:filesystem",
        "//maldoca/base:get_runfiles_dir",
        "//maldoca/base:status",
        "//maldoca/base/testing:status_matchers",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:optional",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "conversion_test_util",
    testonly = True,
    hdrs = ["conversion_test_util.h"],
    deps = [
        "//maldoca/base/testing:status_matchers",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@nlohmann_json//:json",
    ],
)
