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



licenses(["notice"])

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//:__subpackages__",
    ],
)

proto_library(
    name = "ast_def_proto",
    srcs = ["ast_def.proto"],
    deps = [":type_proto"],
)

cc_proto_library(
    name = "ast_def_cc_proto",
    deps = ["ast_def_proto"],
)

cc_library(
    name = "ast_def",
    srcs = [
        "ast_def.cc",
        "type.cc",
    ],
    hdrs = [
        "ast_def.h",
        "type.h",
    ],
    deps = [
        ":ast_def_cc_proto",
        ":symbol",
        ":type_cc_proto",
        "//maldoca/base:status",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/container:btree",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/functional:bind_front",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_library(
    name = "ast_gen",
    srcs = ["ast_gen.cc"],
    hdrs = ["ast_gen.h"],
    deps = [
        ":ast_def",
        ":ast_def_cc_proto",
        ":symbol",
        "//maldoca/base:path",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/cleanup",
        "@abseil-cpp//absl/container:btree",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@com_google_protobuf//src/google/protobuf/io",
        "@com_google_protobuf//src/google/protobuf/io:printer",
    ],
)

cc_binary(
    name = "ast_gen_main",
    srcs = ["ast_gen_main.cc"],
    deps = [
        ":ast_def",
        ":ast_def_cc_proto",
        ":ast_gen",
        "//maldoca/base:filesystem",
        "//maldoca/base:path",
        "//maldoca/base:status",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/strings",
    ],
)

cc_test(
    name = "ast_gen_test",
    srcs = ["ast_gen_test.cc"],
    deps = [
        ":ast_def",
        ":ast_def_cc_proto",
        ":ast_gen",
        ":symbol",
        "//maldoca/base:filesystem",
        "//maldoca/base/testing:status_matchers",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/strings",
        "@com_google_protobuf//src/google/protobuf/io",
        "@googletest//:gtest_main",
    ],
)

proto_library(
    name = "type_proto",
    srcs = ["type.proto"],
)

cc_proto_library(
    name = "type_cc_proto",
    deps = ["type_proto"],
)

cc_test(
    name = "type_test",
    srcs = ["type_test.cc"],
    deps = [
        ":ast_def",
        ":ast_def_cc_proto",
        ":type_cc_proto",
        "//maldoca/base:filesystem",
        "//maldoca/base/testing:status_matchers",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "symbol",
    srcs = ["symbol.cc"],
    hdrs = ["symbol.h"],
    deps = [
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/strings",
    ],
)

cc_test(
    name = "symbol_test",
    srcs = ["symbol_test.cc"],
    deps = [
        ":symbol",
        "@googletest//:gtest_main",
    ],
)
