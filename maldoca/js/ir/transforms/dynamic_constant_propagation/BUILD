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

load("@rules_cc//cc:cc_library.bzl", "cc_library")

licenses(["notice"])

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//maldoca/js:__subpackages__",
    ],
)

# Pass class merged into constant_propagation. This target remains for
# PerformDynamicConstantPropagation declarations used by tests.
cc_library(
    name = "pass",
    hdrs = ["pass.h"],
    deps = [
        "//maldoca/js/babel",
        "//maldoca/js/ir/analyses/dynamic_constant_propagation:analysis",
        "//maldoca/js/ir/analyses/dynamic_constant_propagation:dynamic_prelude",
        "//maldoca/js/ir/transforms/constant_propagation:pass",
        "@abseil-cpp//absl/base:nullability",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
