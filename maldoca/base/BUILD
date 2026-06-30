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
        "//:__subpackages__",
    ],
)

cc_library(
    name = "error_code_to_status",
    srcs = ["error_code_to_status.cc"],
    hdrs = ["error_code_to_status.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_builder",
    ],
)

cc_test(
    name = "error_code_to_status_test",
    srcs = ["error_code_to_status_test.cc"],
    deps = [
        ":error_code_to_status",
        "//maldoca/base/testing:status_matchers",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "path",
    srcs = ["path.cc"],
    hdrs = ["path.h"],
    deps = [
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/strings",
    ],
)

cc_library(
    name = "filesystem",
    srcs = ["filesystem.cc"],
    hdrs = ["filesystem.h"],
    deps = [
        ":error_code_to_status",
        ":status",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_builder",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@com_google_protobuf//:protobuf",
        "@com_google_protobuf//src/google/protobuf/io:tokenizer",
    ],
)

cc_library(
    name = "get_runfiles_dir",
    srcs = ["get_runfiles_dir.cc"],
    hdrs = ["get_runfiles_dir.h"],
    deps = [
        ":path",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/strings:string_view",
        "@rules_cc//cc/runfiles",
    ],
)

cc_library(
    name = "ret_check",
    srcs = ["ret_check.cc"],
    hdrs = ["ret_check.h"],
    deps = [
        ":status",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:log_severity",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_builder",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:source_location",
    ],
)

# TODO(tzx): Remove this target once all references have been updated to
# @abseil-cpp//absl/status.
cc_library(
    name = "status",
    hdrs = [
        "status_macros.h",
    ],
    visibility = [
        "//maldoca/base:__subpackages__",
    ],
    deps = [
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_builder",
        "@abseil-cpp//absl/types:source_location",
    ],
)
