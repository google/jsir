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

licenses(["notice"])

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "libquickjs",
    copts = [
        "-fpermissive",
        "-Wno-unused-variable",
        "-fno-sanitize=float-cast-overflow",
        "-D_GNU_SOURCE",
        "-DCONFIG_VERSION='\"2024-01-13\"'",
    ],
    hdrs = [
        "quickjs/cutils.h",
        "quickjs/libbf.h",
        "quickjs/libregexp.h",
        "quickjs/libregexp-opcode.h",
        "quickjs/libunicode.h",
        "quickjs/libunicode-table.h",
        "quickjs/list.h",
        "quickjs/quickjs.h",
        "quickjs/quickjs-atom.h",
        "quickjs/quickjs-libc.h",
        "quickjs/quickjs-opcode.h",
    ],
    srcs = [
        "quickjs/cutils.c",
        "quickjs/libbf.c",
        "quickjs/libregexp.c",
        "quickjs/libunicode.c",
        "quickjs/quickjs.c",
        "quickjs/quickjs-libc.c",
    ],
    linkopts = [
        "-ldl",
        "-lm",
    ],
    local_defines = ["CONFIG_BIGNUM"] + select({
        # QuickJS checks for EMSCRIPTEN, which newer versions of Emscripten no
        # longer define (only __EMSCRIPTEN__).
        "@platforms//os:emscripten": ["EMSCRIPTEN"],
        "//conditions:default": [],
    }),
)
