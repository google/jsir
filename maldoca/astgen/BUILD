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

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@com_google_protobuf//bazel:py_proto_library.bzl", "py_proto_library")
load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")



licenses(["notice"])

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//:__subpackages__",
    ],
)

proto_library(
    name = "ast_def_proto",
    srcs = ["ast_def.proto"],
    deps = [":type_proto"],
)

cc_proto_library(
    name = "ast_def_cc_proto",
    deps = ["ast_def_proto"],
)

py_proto_library(
    name = "ast_def_py_pb2",
    deps = ["ast_def_proto"],
)

cc_library(
    name = "ast_def",
    srcs = [
        "ast_def.cc",
        "type.cc",
    ],
    hdrs = [
        "ast_def.h",
        "type.h",
    ],
    deps = [
        ":ast_def_cc_proto",
        ":symbol",
        ":type_cc_proto",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/container:btree",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/functional:bind_front",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
    ],
)

py_library(
    name = "type_py",
    srcs = ["type.py"],
    deps = [
        ":ast_def_py_pb2",
        ":symbol_py",
        ":type_py_pb2",
    ],
)

py_test(
    name = "type_py_test",
    srcs = ["type_test.py"],
    main = "type_test.py",
    deps = [":type_py"],
)

py_library(
    name = "ast_def_py",
    srcs = ["ast_def.py"],
    deps = [
        ":ast_def_py_pb2",
        ":symbol_py",
        ":type_py",
    ],
)

py_test(
    name = "ast_def_py_test",
    srcs = ["ast_def_test.py"],
    main = "ast_def_test.py",
    deps = [":ast_def_py"],
)

cc_library(
    name = "printer_base",
    hdrs = ["printer_base.h"],
    deps = [
        "@com_google_protobuf//src/google/protobuf/io",
        "@com_google_protobuf//src/google/protobuf/io:printer",
    ],
)

py_library(
    name = "printer_base_py",
    srcs = ["printer_base.py"],
)

py_test(
    name = "printer_base_py_test",
    srcs = ["printer_base_test.py"],
    main = "printer_base_test.py",
    deps = [":printer_base_py"],
)

cc_library(
    name = "ast_gen_utils",
    hdrs = ["ast_gen_utils.h"],
    deps = [
        ":ast_def",
        "//maldoca/base:path",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/strings",
        "@com_google_protobuf//src/google/protobuf/io:printer",
    ],
)

py_library(
    name = "ast_gen_utils_py",
    srcs = ["ast_gen_utils.py"],
    deps = [
        ":ast_def_py",
        ":printer_base_py",
    ],
)

cc_library(
    name = "cc_printer_base",
    srcs = ["cc_printer_base.cc"],
    hdrs = ["cc_printer_base.h"],
    deps = [
        ":ast_def",
        ":ast_gen_utils",
        ":printer_base",
        "@abseil-cpp//absl/strings",
        "@com_google_protobuf//src/google/protobuf/io",
    ],
)

py_library(
    name = "cc_printer_base_py",
    srcs = ["cc_printer_base.py"],
    deps = [
        ":ast_def_py",
        ":printer_base_py",
    ],
)

py_test(
    name = "cc_printer_base_py_test",
    srcs = ["cc_printer_base_test.py"],
    main = "cc_printer_base_test.py",
    deps = [":cc_printer_base_py"],
)

cc_library(
    name = "ast_header_printer",
    srcs = ["ast_header_printer.cc"],
    hdrs = ["ast_header_printer.h"],
    deps = [
        ":ast_def",
        ":ast_gen_utils",
        ":cc_printer_base",
        ":symbol",
        "@abseil-cpp//absl/strings",
        "@com_google_protobuf//src/google/protobuf/io",
    ],
)

py_library(
    name = "ast_header_printer_py",
    srcs = ["ast_header_printer.py"],
    deps = [
        ":ast_def_py",
        ":ast_gen_utils_py",
        ":cc_printer_base_py",
        ":symbol_py",
    ],
)

cc_library(
    name = "ast_source_printer",
    srcs = ["ast_source_printer.cc"],
    hdrs = ["ast_source_printer.h"],
    deps = [
        ":ast_def",
        ":ast_def_cc_proto",
        ":ast_gen_utils",
        ":cc_printer_base",
        ":symbol",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@com_google_protobuf//src/google/protobuf/io",
    ],
)

py_library(
    name = "ast_source_printer_py",
    srcs = ["ast_source_printer.py"],
    deps = [
        ":ast_def_py",
        ":ast_def_py_pb2",
        ":ast_gen_utils_py",
        ":cc_printer_base_py",
        ":symbol_py",
        ":type_py",
    ],
)

cc_library(
    name = "ast_serialize_printer",
    srcs = ["ast_serialize_printer.cc"],
    hdrs = ["ast_serialize_printer.h"],
    deps = [
        ":ast_def",
        ":ast_gen_utils",
        ":cc_printer_base",
        ":symbol",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@com_google_protobuf//src/google/protobuf/io",
    ],
)

py_library(
    name = "ast_serialize_printer_py",
    srcs = ["ast_serialize_printer.py"],
    deps = [
        ":ast_def_py",
        ":ast_def_py_pb2",
        ":ast_gen_utils_py",
        ":cc_printer_base_py",
        ":symbol_py",
        ":type_py",
    ],
)

cc_library(
    name = "ast_from_json_printer",
    srcs = ["ast_from_json_printer.cc"],
    hdrs = ["ast_from_json_printer.h"],
    deps = [
        ":ast_def",
        ":ast_def_cc_proto",
        ":ast_from_json_utils",
        ":ast_gen_utils",
        ":cc_printer_base",
        ":symbol",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/cleanup",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/strings",
        "@com_google_protobuf//src/google/protobuf/io",
    ],
)

py_library(
    name = "ast_from_json_printer_py",
    srcs = ["ast_from_json_printer.py"],
    deps = [
        ":ast_def_py",
        ":ast_def_py_pb2",
        ":ast_gen_utils_py",
        ":cc_printer_base_py",
        ":symbol_py",
        ":type_py",
    ],
)

cc_library(
    name = "ts_interface_printer",
    srcs = ["ts_interface_printer.cc"],
    hdrs = ["ts_interface_printer.h"],
    deps = [
        ":ast_def",
        ":ast_def_cc_proto",
        ":ast_gen_utils",
        ":printer_base",
        "@abseil-cpp//absl/strings",
        "@com_google_protobuf//src/google/protobuf/io",
    ],
)

py_library(
    name = "ts_interface_printer_py",
    srcs = ["ts_interface_printer.py"],
    deps = [
        ":ast_def_py",
        ":ast_def_py_pb2",
        ":ast_gen_utils_py",
        ":printer_base_py",
        ":type_py",
    ],
)

cc_library(
    name = "ir_table_gen_printer",
    srcs = ["ir_table_gen_printer.cc"],
    hdrs = ["ir_table_gen_printer.h"],
    deps = [
        ":ast_def",
        ":ast_gen_utils",
        ":cc_printer_base",
        ":symbol",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/strings",
        "@com_google_protobuf//src/google/protobuf/io",
    ],
)

py_library(
    name = "ir_table_gen_printer_py",
    srcs = ["ir_table_gen_printer.py"],
    deps = [
        ":ast_def_py",
        ":ast_def_py_pb2",
        ":ast_gen_utils_py",
        ":cc_printer_base_py",
        ":symbol_py",
        ":type_py",
    ],
)

cc_library(
    name = "ast_to_ir_source_printer",
    srcs = ["ast_to_ir_source_printer.cc"],
    hdrs = ["ast_to_ir_source_printer.h"],
    deps = [
        ":ast_def",
        ":ast_def_cc_proto",
        ":ast_gen_utils",
        ":cc_printer_base",
        ":symbol",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@com_google_protobuf//src/google/protobuf/io",
    ],
)

py_library(
    name = "ast_to_ir_source_printer_py",
    srcs = ["ast_to_ir_source_printer.py"],
    deps = [
        ":ast_def_py",
        ":ast_def_py_pb2",
        ":ast_gen_utils_py",
        ":cc_printer_base_py",
        ":symbol_py",
        ":type_py",
    ],
)

cc_library(
    name = "ir_to_ast_source_printer",
    srcs = ["ir_to_ast_source_printer.cc"],
    hdrs = ["ir_to_ast_source_printer.h"],
    deps = [
        ":ast_def",
        ":ast_def_cc_proto",
        ":ast_gen_utils",
        ":cc_printer_base",
        ":printer_base",
        ":symbol",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:btree",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/strings",
        "@com_google_protobuf//src/google/protobuf/io",
        "@com_google_protobuf//src/google/protobuf/io:printer",
    ],
)

py_library(
    name = "ir_to_ast_source_printer_py",
    srcs = ["ir_to_ast_source_printer.py"],
    deps = [
        ":ast_def_py",
        ":ast_def_py_pb2",
        ":ast_gen_utils_py",
        ":cc_printer_base_py",
        ":symbol_py",
        ":type_py",
    ],
)

cc_library(
    name = "ast_from_json_utils",
    hdrs = ["ast_from_json_utils.h"],
    deps = [
        "@abseil-cpp//absl/functional:any_invocable",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@nlohmann_json//:json",
    ],
)

cc_library(
    name = "ir_to_ast_util",
    hdrs = ["ir_to_ast_util.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_binary(
    name = "ast_gen_main",
    srcs = ["ast_gen_main.cc"],
    deps = [
        ":ast_def",
        ":ast_def_cc_proto",
        ":ast_from_json_printer",
        ":ast_header_printer",
        ":ast_serialize_printer",
        ":ast_source_printer",
        ":ast_to_ir_source_printer",
        ":ir_table_gen_printer",
        ":ir_to_ast_source_printer",
        "//maldoca/base:filesystem",
        "//maldoca/base:path",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/strings",
    ],
)

py_binary(
    name = "ast_gen_main_py",
    srcs = ["ast_gen_main.py"],
    main = "ast_gen_main.py",
    deps = [
        ":ast_def_py",
        ":ast_def_py_pb2",
        ":ast_from_json_printer_py",
        ":ast_header_printer_py",
        ":ast_serialize_printer_py",
        ":ast_source_printer_py",
        ":ast_to_ir_source_printer_py",
        ":ir_table_gen_printer_py",
        ":ir_to_ast_source_printer_py",
    ],
)

cc_test(
    name = "ast_gen_test",
    srcs = ["ast_gen_test.cc"],
    deps = [
        ":ast_def",
        ":ast_def_cc_proto",
        ":ast_from_json_printer",
        ":ast_header_printer",
        ":symbol",
        ":ts_interface_printer",
        "//maldoca/base:filesystem",
        "//maldoca/base/testing:status_matchers",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/strings",
        "@com_google_protobuf//src/google/protobuf/io",
        "@googletest//:gtest_main",
    ],
)

proto_library(
    name = "type_proto",
    srcs = ["type.proto"],
)

cc_proto_library(
    name = "type_cc_proto",
    deps = ["type_proto"],
)

py_proto_library(
    name = "type_py_pb2",
    deps = ["type_proto"],
)

cc_test(
    name = "type_test",
    srcs = ["type_test.cc"],
    deps = [
        ":ast_def",
        ":ast_def_cc_proto",
        ":type_cc_proto",
        "//maldoca/base:filesystem",
        "//maldoca/base/testing:status_matchers",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "symbol",
    srcs = ["symbol.cc"],
    hdrs = ["symbol.h"],
    deps = [
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/strings",
    ],
)

cc_test(
    name = "symbol_test",
    srcs = ["symbol_test.cc"],
    deps = [
        ":symbol",
        "@googletest//:gtest_main",
    ],
)

# Python port of the above. See docs/migrate_astgen_to_python.md.
py_library(
    name = "symbol_py",
    srcs = ["symbol.py"],
)

py_test(
    name = "symbol_py_test",
    srcs = ["symbol_test.py"],
    main = "symbol_test.py",
    deps = [":symbol_py"],
)
