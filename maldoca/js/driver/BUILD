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

load("//maldoca/js:allowed_users.bzl", "JSIR_ALLOWED_USERS")

licenses(["notice"])

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = JSIR_ALLOWED_USERS,
)

proto_library(
    name = "driver_proto",
    srcs = ["driver.proto"],
    deps = [
        "//maldoca/js/babel:babel_proto",
        "@protobuf//:duration_proto",
    ],
)

cc_proto_library(
    name = "driver_cc_proto",
    deps = [":driver_proto"],
)

cc_library(
    name = "driver",
    srcs = [
        "driver.cc",
        "internal/conversions.cc",
    ],
    hdrs = [
        "driver.h",
        "internal/conversions.h",
    ],
    deps = [
        ":driver_cc_proto",
        "//maldoca/base:ret_check",
        "//maldoca/base:status",
        "//maldoca/js/ast",
        "//maldoca/js/ast:ast_util",
        "//maldoca/js/ast/transforms:transform",
        "//maldoca/js/babel",
        "//maldoca/js/ir",
        "//maldoca/js/ir/analyses:analysis",
        "//maldoca/js/ir/conversion:utils",
        "//maldoca/js/ir/transforms:transform",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/time",
        "@abseil-cpp//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@nlohmann_json//:json",
        "@protobuf//:duration_cc_proto",
    ],
)
