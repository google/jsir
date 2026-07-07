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
"""Port of maldoca/js/ast/ast_gen_test.cc to Python.

This is the real end-to-end check: unlike the 9 small schemas under
maldoca/astgen/test/, this runs the Python astgen port against the actual
production JS AST schema (maldoca/js/ast/ast_def.textproto, ~3400 lines)
and diffs against the real checked-in generated C++/MLIR sources.
"""

import os
import unittest

from maldoca.astgen.test import ast_gen_test_util

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))


class JsAstGenTest(ast_gen_test_util.AstGenTest):
  PARAM = ast_gen_test_util.AstGenTestParam(
      test_dir=_TEST_DIR,
      cc_namespace="maldoca",
      ast_path="maldoca/js/ast",
      ir_path="maldoca/js/ir",
      expected_ast_header_path="ast.generated.h",
      expected_ast_source_path="ast.generated.cc",
      expected_ast_to_json_path="ast_to_json.generated.cc",
      expected_ast_from_json_path="ast_from_json.generated.cc",
      expected_ir_tablegen_path="../ir/jsir_ops.generated.td",
      expected_ast_to_ir_source_path="../ir/conversion/ast_to_jsir.generated.cc",
      expected_ir_to_ast_source_path="../ir/conversion/jsir_to_ast.generated.cc",
  )


if __name__ == "__main__":
  unittest.main()
