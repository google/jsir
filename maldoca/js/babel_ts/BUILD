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

# The TypeScript glue between JSIR and Babel. This is the single source of truth
# for every JavaScript runtime that JSIR drives Babel in:
#
# - Node.js (sandboxed_nodejs_babel): uses `:babel_ts` with the `@babel/*` npm
#   packages.
# - Embedded engines (V8, QuickJS, the wasm demo in the browser): use
#   `:babel_bundle.js`, i.e. babel-standalone followed by `bundle_main.ts`
#   compiled and bundled into a classic script by esbuild.
#
# In google3, esbuild comes from //third_party/golang/esbuild. In the GitHub
# repo, it is a prebuilt binary fetched by //bazel:extensions.bzl, and the
# `esbuild_binary` macro comes from //bazel:esbuild.bzl.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//bazel:esbuild.bzl", "esbuild_binary")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//maldoca/js:__subpackages__"],
)

licenses(["notice"])

# Engine-agnostic sources. They only import *types* from `@babel/*`; the Babel
# runtime is passed in as a `BabelPackages`.
_CORE_SRCS = [
    "babel_generate.ts",
    "babel_packages.ts",
    "babel_parse.ts",
    "base64_encode_decode_string_values.ts",
    "comment_uid.ts",
    "traverse_object.ts",
]

# bundle_main.ts and its dependencies, compiled into a single classic script
# (IIFE) that expects babel-standalone to be already loaded.
esbuild_binary(
    name = "babel_glue",
    srcs = _CORE_SRCS + ["bundle_main.ts"],
    out = "babel_glue.js",
    entrypoints = ["bundle_main.ts"],
    inject_import_meta = False,
    platform = "browser",
    # QuickJS supports ES2020; V8 and browsers support more.
    target = "es2020",
)

# babel-standalone + the glue, as one classic script. Evaluating it defines
# `globalThis.jsirBabel = {parse, generate}`; see bundle_main.ts.
genrule(
    name = "babel_bundle",
    srcs = [
        "//third_party/babel_standalone:babel.min.js",
        ":babel_glue.js",
    ],
    outs = ["babel_bundle.js"],
    cmd = "(cat $(location //third_party/babel_standalone:babel.min.js) " +
          "&& echo ';' " +
          "&& cat $(location :babel_glue.js)) > $@",
)

# babel_bundle.js as a C++ char array:
#   constexpr char kBabelBundle[] = {...};  // Not null-terminated.
genrule(
    name = "babel_bundle_embed_genrule",
    srcs = [":babel_bundle.js"],
    outs = ["babel_bundle_embed.h"],
    cmd = "(echo '#pragma once' " +
          "&& echo 'constexpr char kBabelBundle[] = {' " +
          "&& od -v -An -tx1 $< | sed -e 's/ *\\([0-9a-f][0-9a-f]\\)/0x\\1,/g' " +
          "&& echo '};') > $@",
)

cc_library(
    name = "babel_bundle_embed",
    hdrs = ["babel_bundle_embed.h"],
)
