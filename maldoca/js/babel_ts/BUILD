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
#   `:babel_bundle.js`, i.e. babel-standalone and `bundle_main.ts` bundled into
#   a classic script by esbuild.
#
# In google3, esbuild comes from //third_party/golang/esbuild
# (`esbuild_binary`). In the GitHub repo, it comes from aspect_rules_esbuild
# (`esbuild`), and babel-standalone comes from npm. Copybara strip/insert
# comments switch between the two.

load("@aspect_rules_esbuild//esbuild:defs.bzl", "esbuild")
load("@npm//:jasmine/package_json.bzl", jasmine_bin = "bin")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

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

# babel-standalone (babel.min.js), which esbuild bundles as a plain JS library.
# `@babel/standalone` imports resolve to it via the "browser" field of
# package.json.
_BABEL_STANDALONE = "//:node_modules/@babel/standalone"

_BABEL_BUNDLE_SRCS = _CORE_SRCS + [
    "bundle_main.ts",
    "package.json",
    _BABEL_STANDALONE,
]

# bundle_main.ts, its dependencies and babel-standalone, bundled into a single
# classic script (IIFE). Evaluating it defines
# `globalThis.jsirBabel = {parse, generate}`; see bundle_main.ts.
#
# The bundle is minified, since esbuild re-prints babel.min.js anyway. It
# targets ES2020 for QuickJS; V8 and browsers support more.
esbuild(
    name = "babel_bundle",
    srcs = _BABEL_BUNDLE_SRCS,
    entry_point = "bundle_main.ts",
    minify = True,
    output = "babel_bundle.js",
    platform = "browser",
    sourcemap = None,
    target = "es2020",
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

# Tests of the core logic. Like babel_bundle.js, they are bundled by esbuild,
# which consumes babel-standalone as a plain JS library.
_TEST_BUNDLE_SRCS = _CORE_SRCS + [
    "babel_implementations_for_test.ts",
    "package.json",
    _BABEL_STANDALONE,
]

_TEST_BUNDLE_EXTERNAL = [
    "@babel/generator",
    "@babel/parser",
    "@babel/traverse",
    "@babel/types",
    "jasmine",
]

esbuild(
    name = "babel_parse_test_bundle",
    testonly = True,
    srcs = _TEST_BUNDLE_SRCS + ["babel_parse_test.ts"],
    entry_point = "babel_parse_test.ts",
    external = _TEST_BUNDLE_EXTERNAL,
    output = "babel_parse_test_bundle.js",
    # For the "browser" field of package.json.
    platform = "browser",
    sourcemap = None,
    target = "node18",
)

jasmine_bin.jasmine_test(
    name = "babel_parse_test",
    args = ["$(rootpath :babel_parse_test_bundle)"],
    data = [":babel_parse_test_bundle"],
)
