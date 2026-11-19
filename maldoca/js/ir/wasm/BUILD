# Copyright 2026 Google LLC
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

# In-browser JSIR demo: Babel parses JavaScript in the page, and the C++ half of
# JSIR (AST -> JSIR conversion and printing) runs as WebAssembly.
#
# In the open-source repo, everything here is tagged "manual", so that
# `bazel build //...` doesn't download the Emscripten SDK and compile LLVM a
# second time (for wasm).

load("@aspect_rules_esbuild//esbuild:defs.bzl", "esbuild")
load("@emsdk//emscripten_toolchain:wasm_rules.bzl", "wasm_cc_binary")
load("@rules_cc//cc:defs.bzl", "cc_binary")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")

licenses(["notice"])

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:private"],
)

# C++ half of the demo: Babel AST JSON -> JSHIR, compiled to WebAssembly.
cc_binary(
    name = "jsir_wasm_bin",
    srcs = ["jsir_wasm.cc"],
    linkopts = [
        "-lembind",
        "-sALLOW_MEMORY_GROWTH=1",
        "-sASSERTIONS=1",
        "-sMODULARIZE=1",
        "-sEXPORT_NAME=loadJsirWasm",
        # Embed the .wasm in the JS glue, so that the page needs no fetch().
        "-sSINGLE_FILE=1",
    ],
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [
        "//maldoca/js/ast",
        "//maldoca/js/ir",
        "//maldoca/js/ir/conversion:utils",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@nlohmann_json//:json",
    ],
)

# Outputs jsir_wasm/jsir_wasm_bin.js.
wasm_cc_binary(
    name = "jsir_wasm",
    cc_target = ":jsir_wasm_bin",
    tags = ["manual"],
    threads = "off",
)

# UI logic of the demo page, bundled into a single script. Internally this is
# type-checked by the TypeScript compiler; here esbuild only strips the types.
esbuild(
    name = "app",
    srcs = [
        "main.ts",
        "source_map.ts",
    ],
    entry_point = "main.ts",
    format = "iife",
    sourcemap = False,
    tags = ["manual"],
    target = "es2020",
)

# The files of the demo site. Babel (parsing) runs directly in the browser,
# using the same glue code that JSIR uses to drive Babel inside QuickJS, so the
# AST JSON fed into C++ is identical to what the native pipeline sees.
py_library(
    name = "site",
    srcs = ["site.py"],
    data = [
        "index.html",
        ":app.js",
        ":jsir_wasm",
        "//maldoca/js/quickjs_babel:native.js",
        "//third_party/babel_standalone:babel.min.js",
        "@monaco_editor//:vs",
    ],
    tags = ["manual"],
    deps = [
        "@rules_python//python/runfiles",
    ],
)

# Local web server for the demo.
py_binary(
    name = "server",
    srcs = ["server.py"],
    tags = [
        "manual",
        "notap",
    ],
    deps = [":site"],
)

# Copies the demo site into a directory, e.g. for GitHub Pages.
py_binary(
    name = "stage_site",
    srcs = ["stage_site.py"],
    tags = [
        "manual",
        "notap",
    ],
    deps = [":site"],
)
