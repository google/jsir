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
        "//maldoca/js:__subpackages__",
    ],
)

td_library(
    name = "jsir_builtin_dialect_td_files",
    srcs = [
        "jsir_builtin_dialect.td",
    ],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "jsir_builtin_dialect_inc_gen",
    tbl_outs = {
        "jsir_builtin_dialect.h.inc": [
            "-gen-dialect-decls",
            "-dialect=jsir_builtin",
        ],
        "jsir_builtin_dialect.cc.inc": [
            "-gen-dialect-defs",
            "-dialect=jsir_builtin",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "attrs.td",
    deps = [":attrs_td_files"],
)

td_library(
    name = "attrs_td_files",
    srcs = [
        "attrs.td",
    ],
    deps = [
        ":jsir_builtin_dialect_td_files",
        "//maldoca/js/ir:interfaces_td_files",
        "//maldoca/js/ir:jsir_attrs_td_files",
        "//maldoca/js/ir:jsir_dialect_td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "attrs_inc_gen",
    tbl_outs = {
        "attrs.h.inc": [
            "-gen-attrdef-decls",
            "-attrdefs-dialect=jsir_builtin",
        ],
        "attrs.cc.inc": [
            "--gen-attrdef-defs",
            "-attrdefs-dialect=jsir_builtin",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "attrs.td",
    deps = [
        ":attrs_td_files",
    ],
)

cc_library(
    name = "attrs",
    srcs = ["attrs.cc"],
    hdrs = ["attrs.h"],
    deps = [
        ":attrs_inc_gen",
        ":jsir_builtin_dialect_inc_gen",
        "//maldoca/js/ir",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "dynamic_prelude",
    srcs = ["dynamic_prelude.cc"],
    hdrs = ["dynamic_prelude.h"],
    deps = [
        "//maldoca/base:status",
        "//maldoca/js/ast",
        "//maldoca/js/babel",
        "//maldoca/js/driver:driver_cc_proto",
        "//maldoca/js/quickjs",
        "@abseil-cpp//absl/cleanup",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/time",
        "@quickjs//:libquickjs",
    ],
)

cc_test(
    name = "dynamic_prelude_test",
    srcs = ["dynamic_prelude_test.cc"],
    deps = [
        ":dynamic_prelude",
        "//maldoca/base/testing:status_matchers",
        "//maldoca/js/quickjs",
        "//maldoca/js/quickjs_babel",
        "@googletest//:gtest_main",
        "@quickjs//:libquickjs",
    ],
)

cc_library(
    name = "symbol_mutation_info",
    srcs = ["symbol_mutation_info.cc"],
    hdrs = ["symbol_mutation_info.h"],
    deps = [
        "//maldoca/js/ast",
        "//maldoca/js/babel:scope",
        "//maldoca/js/ir",
        "//maldoca/js/ir:jsir_utils",
        "//maldoca/js/ir/analyses:scope",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_test(
    name = "symbol_mutation_info_test",
    srcs = ["symbol_mutation_info_test.cc"],
    deps = [
        ":symbol_mutation_info",
        "//maldoca/base/testing:status_matchers",
        "//maldoca/js/ast",
        "//maldoca/js/driver",
        "//maldoca/js/driver:conversion",
        "//maldoca/js/driver:driver_cc_proto",
        "//maldoca/js/ir",
        "//maldoca/js/ir/conversion:utils",
        "//maldoca/js/quickjs_babel",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "analysis",
    srcs = ["analysis.cc"],
    hdrs = ["analysis.h"],
    deps = [
        ":attrs",
        ":dynamic_prelude",
        ":symbol_mutation_info",
        "//maldoca/js/ast",
        "//maldoca/js/driver:driver_cc_proto",
        "//maldoca/js/ir",
        "//maldoca/js/ir:jsir_utils",
        "//maldoca/js/ir/analyses:dataflow_analysis",
        "//maldoca/js/ir/analyses:scope",
        "//maldoca/js/ir/analyses/constant_propagation:analysis",
        "//maldoca/js/quickjs",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/cleanup",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@quickjs//:libquickjs",
    ],
)

cc_test(
    name = "analysis_test",
    srcs = ["analysis_test.cc"],
    deps = [
        ":analysis",
        ":attrs",
        ":dynamic_prelude",
        "//maldoca/base/testing:status_matchers",
        "//maldoca/js/ast",
        "//maldoca/js/ast:ast_util",
        "//maldoca/js/ast/transforms/extract_prelude:pass",
        "//maldoca/js/babel",
        "//maldoca/js/driver",
        "//maldoca/js/driver:conversion",
        "//maldoca/js/ir",
        "//maldoca/js/ir/analyses:dataflow_analysis",
        "//maldoca/js/ir/analyses/constant_propagation:analysis",
        "//maldoca/js/ir/conversion:utils",
        "//maldoca/js/quickjs_babel",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
