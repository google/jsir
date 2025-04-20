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

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//maldoca/js:__subpackages__",
    ],
)

genrule(
    name = "native_cc_embed_data_genrule",
    srcs = ["native.js"],
    outs = ["native_cc_embed_data.cc.inc"],
    cmd = "( echo \"constexpr char kNative[] = {\"" +
          "&& od -t x1 $(location native.js) | sed -e 's/ /,0x/g' -e 's/$$/,/' -e 's/[^,]*,//' -e 's/0x,//g' && echo \"};\"" +
          "&& echo -n \"constexpr size_t kNativeSize = \"" +
          "&& wc -c < $(location native.js)" +
          "&& echo \";\" )" +
          "> \"$@\"",
)

cc_library(
    name = "native_cc_embed_data",
    hdrs = ["native_cc_embed_data.cc.inc"],
    deps = [":native_cc_embed_data_genrule"],
)

genrule(
    name = "babel_standalone_cc_embed_data_genrule",
    srcs = ["//third_party/babel_standalone:babel.min.js"],
    outs = ["babel_standalone_cc_embed_data.cc.inc"],
    cmd = "( echo \"constexpr char kBabelStandalone[] = {\"" +
          "&& od -t x1 $(location //third_party/babel_standalone:babel.min.js) | sed -e 's/ /,0x/g' -e 's/$$/,/' -e 's/[^,]*,//' -e 's/0x,//g' && echo \"};\"" +
          "&& echo -n \"constexpr size_t kBabelStandaloneSize = \"" +
          "&& wc -c < $(location //third_party/babel_standalone:babel.min.js)" +
          "&& echo \";\" )" +
          "> \"$@\"",
)

cc_library(
    name = "babel_standalone_cc_embed_data",
    hdrs = ["babel_standalone_cc_embed_data.cc.inc"],
    deps = [":babel_standalone_cc_embed_data_genrule"],
)

cc_library(
    name = "quickjs_babel",
    srcs = ["quickjs_babel.cc"],
    hdrs = ["quickjs_babel.h"],
    deps = [
        ":babel_standalone_cc_embed_data",
        ":native_cc_embed_data",
        "//maldoca/base:status",
        "//maldoca/js/babel",
        "//maldoca/js/babel:babel_cc_proto",
        "//maldoca/js/babel:babel_internal",
        "//maldoca/js/babel:babel_internal_cc_proto",
        "//maldoca/js/quickjs",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/time",
        "@nlohmann_json//:json",
        "@protobuf//:json",
        "@quickjs//:libquickjs",
    ],
)

cc_test(
    name = "quickjs_babel_test",
    srcs = ["quickjs_babel_test.cc"],
    deps = [
        ":quickjs_babel",
        "//maldoca/js/babel:babel_cc_proto",
        "//maldoca/js/babel:babel_test",
        "@googletest//:gtest_main",
    ],
)
