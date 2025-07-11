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

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//maldoca/js:__subpackages__"],
)

licenses(["notice"])

cc_library(
    name = "quickjs",
    hdrs = ["quickjs.h"],
    deps = [
        "@abseil-cpp//absl/cleanup",
        "@quickjs//:libquickjs",
    ],
)

cc_test(
    name = "quickjs_test",
    srcs = ["quickjs_test.cc"],
    deps = [
        ":quickjs",
        "@abseil-cpp//absl/cleanup",
        "@abseil-cpp//absl/strings:string_view",
        "@googletest//:gtest_main",
        "@quickjs//:libquickjs",
    ],
)
