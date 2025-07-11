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

exports_files([
    "ast_to_jsir.generated.cc",
    "jsir_to_ast.generated.cc",
])

cc_library(
    name = "ast_to_jsir",
    srcs = [
        "ast_to_jsir.generated.cc",
        "ast_to_jsir.handwritten.cc",
    ],
    hdrs = ["ast_to_jsir.h"],
    deps = [
        "//maldoca/js/ast",
        "//maldoca/js/ir",
        "//maldoca/js/ir:trivia",
        "@abseil-cpp//absl/cleanup",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:optional",
        "@abseil-cpp//absl/types:variant",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "jsir_to_ast",
    srcs = [
        "jsir_to_ast.generated.cc",
        "jsir_to_ast.handwritten.cc",
    ],
    hdrs = [
        "jsir_to_ast.h",
    ],
    deps = [
        "//maldoca/base:status",
        "//maldoca/js/ast",
        "//maldoca/js/ir",
        "//maldoca/js/ir:cast",
        "//maldoca/js/ir:trivia",
        "@abseil-cpp//absl/cleanup",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:optional",
        "@abseil-cpp//absl/types:variant",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "jslir_jump_env",
    srcs = ["jslir_jump_env.cc"],
    hdrs = ["jslir_jump_env.h"],
    deps = [
        "@abseil-cpp//absl/cleanup",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "jshir_to_jslir",
    srcs = [
        "jshir_to_jslir.cc",
    ],
    hdrs = [
        "jshir_to_jslir.h",
    ],
    deps = [
        ":jslir_jump_env",
        "//maldoca/js/ast",
        "//maldoca/js/ir",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/types:optional",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "jslir_to_jshir",
    srcs = [
        "jslir_to_jshir.cc",
    ],
    hdrs = [
        "jslir_to_jshir.h",
    ],
    deps = [
        "//maldoca/base:status",
        "//maldoca/js/ir",
        "//maldoca/js/ir:jslir_visitor",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":ast_to_jsir",
        ":jshir_to_jslir",
        ":jsir_to_ast",
        ":jslir_to_jshir",
        "//maldoca/base:ret_check",
        "//maldoca/js/ast",
        "//maldoca/js/ir",
        "@abseil-cpp//absl/status:statusor",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)
