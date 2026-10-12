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
    default_visibility = [
        "//maldoca/js:__subpackages__",
    ],
)

cc_library(
    name = "quickjs_babel",
    srcs = ["quickjs_babel.cc"],
    hdrs = ["quickjs_babel.h"],
    deps = [
        "//maldoca/js/babel",
        "//maldoca/js/babel:babel_cc_proto",
        "//maldoca/js/babel:babel_internal",
        "//maldoca/js/babel:babel_internal_cc_proto",
        "//maldoca/js/babel_ts:babel_bundle_embed",
        "//maldoca/js/quickjs",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/time",
        "@com_google_protobuf//:json",
        "@nlohmann_json//:json",
        "@quickjs//:libquickjs",
    ],
)

cc_test(
    name = "quickjs_babel_test",
    srcs = ["quickjs_babel_test.cc"],
    deps = [
        ":quickjs_babel",
        "//maldoca/js/babel:babel_cc_proto",
        "//maldoca/js/babel:babel_test",
        "@googletest//:gtest_main",
    ],
)
