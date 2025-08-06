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

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//maldoca/js:__subpackages__",
    ],
)

cc_library(
    name = "pass",
    srcs = ["pass.cc"],
    hdrs = ["pass.h"],
    deps = [
        "//maldoca/js/ast",
        "//maldoca/js/babel:babel_cc_proto",
        "//maldoca/js/driver:driver_cc_proto",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:variant",
    ],
)

cc_test(
    name = "pass_test",
    srcs = ["pass_test.cc"],
    deps = [
        ":pass",
        "//maldoca/base/testing:status_matchers",
        "//maldoca/js/babel:babel_cc_proto",
        "//maldoca/js/driver",
        "//maldoca/js/driver:conversion",
        "//maldoca/js/quickjs_babel",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
    ],
)
