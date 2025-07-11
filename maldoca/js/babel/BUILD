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
load("//maldoca/js:allowed_users.bzl", "JSIR_ALLOWED_USERS")

licenses(["notice"])

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = JSIR_ALLOWED_USERS,
)

proto_library(
    name = "babel_proto",
    srcs = ["babel.proto"],
)

proto_library(
    name = "babel_internal_proto",
    srcs = ["babel_internal.proto"],
    deps = [":babel_proto"],
)

cc_proto_library(
    name = "babel_cc_proto",
    visibility = [
        "//maldoca/js:__subpackages__",
    ],
    deps = [":babel_proto"],
)

cc_proto_library(
    name = "babel_internal_cc_proto",
    deps = [":babel_internal_proto"],
)

cc_library(
    name = "babel",
    hdrs = ["babel.h"],
    visibility = [
        "//maldoca/js:__subpackages__",
    ],
    deps = [
        ":babel_cc_proto",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/time",
    ],
)

cc_library(
    name = "scope",
    srcs = ["scope.cc"],
    hdrs = ["scope.h"],
    deps = [
        ":babel_cc_proto",
        "//maldoca/js/ast",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
    ],
)

cc_library(
    name = "babel_test",
    testonly = 1,
    srcs = ["babel_test.cc"],
    hdrs = ["babel_test.h"],
    deps = [
        ":babel",
        ":babel_cc_proto",
        ":scope",
        "//maldoca/base/testing:protocol-buffer-matchers",
        "//maldoca/base/testing:status_matchers",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "babel_internal",
    srcs = ["babel_internal.cc"],
    hdrs = ["babel_internal.h"],
    deps = [
        ":babel_cc_proto",
        ":babel_internal_cc_proto",
        "@nlohmann_json//:json",
    ],
)

cc_test(
    name = "babel_json_test",
    srcs = ["babel_json_test.cc"],
    deps = [
        ":babel_cc_proto",
        "//maldoca/base/testing:protocol-buffer-matchers",
        "//maldoca/base/testing:status_matchers",
        "@com_google_protobuf//:json",
        "@googletest//:gtest_main",
    ],
)
