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

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_python//python:py_test.bzl", "py_test")
load("//maldoca/js:allowed_users.bzl", "JSIR_ALLOWED_USERS")

licenses(["notice"])

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = JSIR_ALLOWED_USERS,
)

cc_test(
    name = "ast_gen_test",
    srcs = ["ast_gen_test.cc"],
    data = [
        "ast.generated.cc",
        "ast.generated.h",
        "ast_def.textproto",
        "ast_from_json.generated.cc",
        "ast_to_json.generated.cc",
        "//maldoca/js/ir:jsir_ops.generated.td",
        "//maldoca/js/ir/conversion:ast_to_jsir.generated.cc",
        "//maldoca/js/ir/conversion:jsir_to_ast.generated.cc",
    ],
    deps = [
        "//maldoca/astgen/test:ast_gen_test_util",
        "@googletest//:gtest_main",
    ],
)

# Golden input/output files for the Python astgen golden-diff test (see
# docs/migrate_astgen_to_python.md) -- mirrors the `data` of the cc_test
# above.
filegroup(
    name = "testdata",
    testonly = True,
    srcs = [
        "ast.generated.cc",
        "ast.generated.h",
        "ast_def.textproto",
        "ast_from_json.generated.cc",
        "ast_to_json.generated.cc",
    ],
)

py_test(
    name = "ast_gen_test_py",
    srcs = ["ast_gen_test.py"],
    main = "ast_gen_test.py",
    deps = [
        "//maldoca/astgen/test:ast_gen_test_util_py",
    ],
    data = [
        ":testdata",
        "//maldoca/js/ir:jsir_ops_generated_td",
        "//maldoca/js/ir/conversion:generated_conversion_sources",
    ],
)

cc_library(
    name = "ast",
    srcs = [
        "ast.generated.cc",
        "ast_from_json.generated.cc",
        "ast_to_json.generated.cc",
    ],
    hdrs = ["ast.generated.h"],
    deps = [
        "//maldoca/astgen:ast_from_json_utils",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@nlohmann_json//:json",
    ],
)

cc_library(
    name = "ast_visitor",
    hdrs = [
        "ast_visitor.h",
    ],
    deps = [
        ":ast",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/memory",
    ],
)

cc_library(
    name = "ast_walker",
    hdrs = [
        "ast_walker.h",
    ],
    deps = [
        ":ast",
        ":ast_visitor",
        "@abseil-cpp//absl/log",
    ],
)

cc_library(
    name = "ast_util",
    srcs = ["ast_util.cc"],
    hdrs = ["ast_util.h"],
    deps = [
        ":ast",
        ":ast_visitor",
        ":ast_walker",
        "//maldoca/js/babel",
        "//maldoca/js/babel:babel_cc_proto",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/time",
        "@abseil-cpp//absl/types:optional",
        "@nlohmann_json//:json",
    ],
)
