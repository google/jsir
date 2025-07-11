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

"""Lit test config."""

# Lint for undefined variables is disabled as config is not defined inside this
# file, instead config is injected by way of evaluating runlit.cfg.py from
# runlit.site.cfg.py which in turn is evaluated by lit.py.
# pylint: disable=undefined-variable

import os
import pathlib

import lit.formats

config.name = "JSIR"
config.suffixes = [".js", ".lit", ".mlir", ".txt"]
config.test_format = lit.formats.ShTest(execute_external=True)

runfiles_dir = pathlib.Path(os.environ["RUNFILES_DIR"])
llvm_dir = runfiles_dir.joinpath("llvm-project/llvm")
jsir_dir = runfiles_dir.joinpath("_main/maldoca/js/ir")

config.environment["PATH"] = (
    f"{str(llvm_dir)}:{str(jsir_dir)}:{os.environ['PATH']}"
)
config.environment["RUNFILES_DIR"] = os.environ["RUNFILES_DIR"]
