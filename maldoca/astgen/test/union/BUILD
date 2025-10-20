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

licenses(["notice"])

package(default_applicable_licenses = ["//:license"])

cc_test(
    name = "ast_gen_test",
    srcs = ["ast_gen_test.cc"],
    data = [
        "ast.generated.cc",
        "ast.generated.h",
        "ast_def.textproto",
        "ast_from_json.generated.cc",
        "ast_to_json.generated.cc",
        "ast_ts_interface.generated",
    ],
    deps = [
        "//maldoca/astgen/test:ast_gen_test_util",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "ast",
    srcs = [
        "ast.generated.cc",
        "ast_from_json.generated.cc",
        "ast_to_json.generated.cc",
    ],
    hdrs = [
        "ast.generated.h",
    ],
    deps = [
        "//maldoca/base:status",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@nlohmann_json//:json",
    ],
)
