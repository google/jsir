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

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//maldoca/js:allowed_users.bzl", "JSIR_ALLOWED_USERS")

licenses(["notice"])

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = JSIR_ALLOWED_USERS,
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
            ["-gen-attr-interface-decls"],
            "attr_interfaces.h.inc",
        ),
        (
            ["-gen-attr-interface-defs"],
            "attr_interfaces.cc.inc",
        ),
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
    name = "jsir_dialect_td_files",
    srcs = [
        "jsir_dialect.td",
    ],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "jsir_dialect_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=jsir",
            ],
            "jsir_dialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=jsir",
            ],
            "jsir_dialect.cc.inc",
        ),
        (
            [
                "-gen-dialect-decls",
                "-dialect=jshir",
            ],
            "jshir_dialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=jshir",
            ],
            "jshir_dialect.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "jsir_attrs.td",
    deps = [":jsir_attrs_td_files"],
)

td_library(
    name = "jsir_attrs_td_files",
    srcs = [
        "jsir_attrs.td",
    ],
    deps = [
        ":interfaces_td_files",
        ":jsir_dialect_td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "jsir_attrs_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-attrdef-decls",
                "-attrdefs-dialect=jsir",
            ],
            "jsir_attrs.h.inc",
        ),
        (
            [
                "--gen-attrdef-defs",
                "-attrdefs-dialect=jsir",
            ],
            "jsir_attrs.cc.inc",
        ),
        (
            [
                "-gen-enum-decls",
            ],
            "jsir_enum_attrs.h.inc",
        ),
        (
            [
                "--gen-enum-defs",
            ],
            "jsir_enum_attrs.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "jsir_attrs.td",
    deps = [
        ":jsir_attrs_td_files",
    ],
)

td_library(
    name = "jsir_types_td_files",
    srcs = [
        "jsir_types.td",
    ],
    deps = [
        ":jsir_dialect_td_files",
    ],
)

gentbl_cc_library(
    name = "jsir_types_inc_gen",
    tbl_outs = [
        (
            ["-gen-typedef-decls"],
            "jsir_types.h.inc",
        ),
        (
            ["-gen-typedef-defs"],
            "jsir_types.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "jsir_types.td",
    deps = [":jsir_types_td_files"],
)

td_library(
    name = "jsir_ops_td_files",
    srcs = [
        "jsir_ops.generated.td",
        "jsir_ops.td",
    ],
    deps = [
        ":interfaces_td_files",
        ":jsir_attrs_td_files",
        ":jsir_dialect_td_files",
        ":jsir_types_td_files",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:LoopLikeInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "jsir_ops_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-op-decls",
                "-op-include-regex=jsir",
            ],
            "jsir_ops.h.inc",
        ),
        (
            [
                "-gen-op-defs",
                "-op-include-regex=jsir",
            ],
            "jsir_ops.cc.inc",
        ),
        (
            [
                "-gen-op-decls",
                "-op-include-regex=jshir",
            ],
            "jshir_ops.h.inc",
        ),
        (
            [
                "-gen-op-defs",
                "-op-include-regex=jshir",
            ],
            "jshir_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "jsir_ops.td",
    deps = [
        ":jsir_ops_td_files",
    ],
)

cc_library(
    name = "ir",
    srcs = [
        "ir.cc",
        "jshir_ops.cc",
        "jsir_ops.cc",
    ],
    hdrs = ["ir.h"],
    deps = [
        ":interfaces_inc_gen",
        ":jsir_attrs_inc_gen",
        ":jsir_dialect_inc_gen",
        ":jsir_ops_inc_gen",
        ":jsir_types_inc_gen",
        "//maldoca/js/quickjs",
        "@abseil-cpp//absl/cleanup",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
        "@quickjs//:libquickjs",
    ],
)

cc_library(
    name = "trivia",
    srcs = ["trivia.cc"],
    hdrs = ["trivia.h"],
    deps = [
        ":ir",
        "//maldoca/js/ast",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "cast",
    hdrs = ["cast.h"],
    deps = [
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "jsir_utils",
    srcs = ["jsir_utils.cc"],
    hdrs = ["jsir_utils.h"],
    deps = [
        ":cast",
        ":ir",
        "//maldoca/base:status",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "jsir_gen_lib",
    srcs = ["jsir_gen_lib.cc"],
    hdrs = ["jsir_gen_lib.h"],
    deps = [
        "//maldoca/base:ret_check",
        "//maldoca/js/babel",
        "//maldoca/js/babel:babel_cc_proto",
        "//maldoca/js/driver",
        "//maldoca/js/driver:driver_cc_proto",
        "//maldoca/js/ir/conversion:utils",
        "//maldoca/js/quickjs_babel",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@com_google_protobuf//:duration_cc_proto",
        "@llvm-project//mlir:Pass",
    ],
)

cc_binary(
    name = "jsir_gen",
    srcs = ["jsir_gen.cc"],
    deps = [
        ":jsir_gen_lib",
        "//maldoca/base:filesystem",
        "//maldoca/js/babel:babel_cc_proto",
        "//maldoca/js/driver:driver_cc_proto",
        "//maldoca/js/quickjs_babel",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/flags:parse",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
    ],
)

filegroup(
    name = "lit_test_files",
    testonly = True,
    data = [
        "//maldoca/js/ir:jsir_gen",
        "//maldoca/js/ir:lit.cfg.py",
        "@llvm-project//llvm:FileCheck",
    ],
)
