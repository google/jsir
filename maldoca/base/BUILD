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
        ":status",
        "@abseil-cpp//absl/status",
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
    name = "filesystem",
    srcs = ["filesystem.cc"],
    hdrs = ["filesystem.h"],
    deps = [
        ":error_code_to_status",
        ":status",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@com_google_protobuf//:protobuf",
        "@com_google_protobuf//src/google/protobuf/io:tokenizer",
    ],
)

cc_library(
    name = "source_location",
    hdrs = ["source_location.h"],
    deps = [
    ],
)

cc_library(
    name = "ret_check",
    srcs = ["ret_check.cc"],
    hdrs = ["ret_check.h"],
    deps = [
        ":source_location",
        ":status",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:log_severity",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
    ],
)

cc_library(
    name = "symbolized_stacktrace",
    srcs = ["symbolized_stacktrace.cc"],
    hdrs = ["symbolized_stacktrace.h"],
    deps = [
        "@abseil-cpp//absl/debugging:stacktrace",
        "@abseil-cpp//absl/debugging:symbolize",
        "@abseil-cpp//absl/strings:str_format",
    ],
)

cc_test(
    name = "symbolized_stacktrace_test",
    srcs = ["symbolized_stacktrace_test.cc"],
    deps = [
        ":symbolized_stacktrace",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "status",
    srcs = ["status_builder.cc"],
    hdrs = [
        "status_builder.h",
        "status_macros.h",
    ],
    deps = [
        ":source_location",
        ":symbolized_stacktrace",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:log_severity",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:log_entry",
        "@abseil-cpp//absl/log:log_sink",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@abseil-cpp//absl/types:span",
    ],
)
