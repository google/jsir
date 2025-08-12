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

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

licenses(["notice"])

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//maldoca/astgen:__subpackages__",
    ],
)

cc_test(
    name = "ast_gen_test",
    srcs = ["ast_gen_test.cc"],
    data = [
        "ast.generated.cc",
        "ast.generated.h",
        "ast_def.textproto",
        "ast_from_json.generated.cc",
        "ast_to_json.generated.cc",
        "ast_ts_interface.generated",
        "vir_ops.generated.td",
        "//maldoca/astgen/test/variant/conversion:ast_to_vir.generated.cc",
        "//maldoca/astgen/test/variant/conversion:vir_to_ast.generated.cc",
    ],
    deps = [
        "//maldoca/astgen/test:ast_gen_test_util",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "ast",
    srcs = [
        "ast.generated.cc",
        "ast_from_json.generated.cc",
        "ast_to_json.generated.cc",
    ],
    hdrs = ["ast.generated.h"],
    deps = [
        "//maldoca/base:status",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@nlohmann_json//:json",
    ],
)

td_library(
    name = "interfaces_td_files",
    srcs = [
        "interfaces.td",
    ],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "interfaces_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "interfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "interfaces.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "interfaces.td",
    deps = [":interfaces_td_files"],
)

td_library(
    name = "vir_dialect_td_files",
    srcs = [
        "vir_dialect.td",
    ],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "vir_dialect_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=vir",
            ],
            "vir_dialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=vir",
            ],
            "vir_dialect.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "vir_dialect.td",
    deps = [":vir_dialect_td_files"],
)

td_library(
    name = "vir_types_td_files",
    srcs = [
        "vir_types.td",
    ],
    deps = [
        ":vir_dialect_td_files",
    ],
)

gentbl_cc_library(
    name = "vir_types_inc_gen",
    tbl_outs = [
        (
            ["-gen-typedef-decls"],
            "vir_types.h.inc",
        ),
        (
            ["-gen-typedef-defs"],
            "vir_types.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "vir_types.td",
    deps = [":vir_types_td_files"],
)

td_library(
    name = "vir_ops_generated_td_files",
    srcs = [
        "vir_ops.generated.td",
    ],
    deps = [
        ":interfaces_td_files",
        ":vir_dialect_td_files",
        ":vir_types_td_files",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:LoopLikeInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "vir_ops_generated_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "vir_ops.generated.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "vir_ops.generated.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "vir_ops.generated.td",
    deps = [":vir_ops_generated_td_files"],
)

cc_library(
    name = "ir",
    srcs = ["ir.cc"],
    hdrs = ["ir.h"],
    deps = [
        ":interfaces_inc_gen",
        ":vir_dialect_inc_gen",
        ":vir_ops_generated_inc_gen",
        ":vir_types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
    ],
)
