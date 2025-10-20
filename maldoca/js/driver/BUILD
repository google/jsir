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
        "@com_google_protobuf//:duration_proto",
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
        "@com_google_protobuf//:duration_cc_proto",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@nlohmann_json//:json",
    ],
)

cc_library(
    name = "conversion",
    srcs = ["conversion.cc"],
    hdrs = ["conversion.h"],
    deps = [
        ":driver",
        "//maldoca/base:status",
        "//maldoca/js/ast",
        "//maldoca/js/ast:ast_util",
        "//maldoca/js/babel",
        "//maldoca/js/babel:babel_cc_proto",
        "//maldoca/js/ir",
        "//maldoca/js/ir/conversion:utils",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/time",
        "@llvm-project//mlir:IR",
    ],
)

cc_test(
    name = "conversion_test",
    srcs = ["conversion_test.cc"],
    data = [
        "test_babel_scopes.txtpb",
        "test_hir.mlir.test",
        "test_parsed_ast.json",
        "test_serialized_ast.json",
        "test_source.js.test",
    ],
    deps = [
        ":conversion",
        ":driver",
        "//maldoca/base:filesystem",
        "//maldoca/base:get_runfiles_dir",
        "//maldoca/base:path",
        "//maldoca/base:status",
        "//maldoca/base/testing:protocol-buffer-matchers",
        "//maldoca/base/testing:status_matchers",
        "//maldoca/js/ast",
        "//maldoca/js/babel:babel_cc_proto",
        "//maldoca/js/ir/conversion:utils",
        "//maldoca/js/quickjs_babel",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@nlohmann_json//:json",
    ],
)
