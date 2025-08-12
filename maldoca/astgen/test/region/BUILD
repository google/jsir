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
        "rir_ops.generated.td",
        "//maldoca/astgen/test/region/conversion:ast_to_rir.generated.cc",
        "//maldoca/astgen/test/region/conversion:rir_to_ast.generated.cc",
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
    tbl_outs = {
        "interfaces.h.inc": ["-gen-op-interface-decls"],
        "interfaces.cc.inc": ["-gen-op-interface-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "interfaces.td",
    deps = [":interfaces_td_files"],
)

td_library(
    name = "rir_dialect_td_files",
    srcs = [
        "rir_dialect.td",
    ],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "rir_dialect_inc_gen",
    tbl_outs = {
        "rir_dialect.h.inc": [
            "-gen-dialect-decls",
            "-dialect=rir",
        ],
        "rir_dialect.cc.inc": [
            "-gen-dialect-defs",
            "-dialect=rir",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "rir_dialect.td",
    deps = [":rir_dialect_td_files"],
)

td_library(
    name = "rir_types_td_files",
    srcs = [
        "rir_types.td",
    ],
    deps = [
        ":rir_dialect_td_files",
    ],
)

gentbl_cc_library(
    name = "rir_types_inc_gen",
    tbl_outs = {
        "rir_types.h.inc": ["-gen-typedef-decls"],
        "rir_types.cc.inc": ["-gen-typedef-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "rir_types.td",
    deps = [":rir_types_td_files"],
)

td_library(
    name = "rir_ops_td_files",
    srcs = [
        "rir_ops.generated.td",
        "rir_ops.td",
    ],
    deps = [
        ":interfaces_td_files",
        ":rir_dialect_td_files",
        ":rir_types_td_files",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:LoopLikeInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "rir_ops_inc_gen",
    tbl_outs = {
        "rir_ops.h.inc": ["-gen-op-decls"],
        "rir_ops.cc.inc": ["-gen-op-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "rir_ops.td",
    deps = [":rir_ops_td_files"],
)

cc_library(
    name = "ir",
    srcs = ["ir.cc"],
    hdrs = ["ir.h"],
    deps = [
        ":interfaces_inc_gen",
        ":rir_dialect_inc_gen",
        ":rir_ops_inc_gen",
        ":rir_types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
    ],
)
