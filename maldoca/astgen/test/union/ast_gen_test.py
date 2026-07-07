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
"""Port of maldoca/astgen/test/union/ast_gen_test.cc to Python."""

import os
import unittest

from maldoca.astgen.test import ast_gen_test_util

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))


class UnionAstGenTest(ast_gen_test_util.AstGenTest):
  # NOTE: ir_path *is* set here (unlike multiple_inheritance/typed_lambda),
  # so the IR printers run for real -- but there are no IR goldens to check
  # against (this schema doesn't set should_generate_ir_op/kinds anywhere,
  # so the IR output is just boilerplate).
  PARAM = ast_gen_test_util.AstGenTestParam(
      test_dir=_TEST_DIR,
      cc_namespace="maldoca",
      ast_path="maldoca/astgen/test/union",
      ir_path="maldoca/astgen/test/union",
      ts_interface_path="ast_ts_interface.generated",
      expected_ast_header_path="ast.generated.h",
      expected_ast_source_path="ast.generated.cc",
      expected_ast_to_json_path="ast_to_json.generated.cc",
      expected_ast_from_json_path="ast_from_json.generated.cc",
  )


if __name__ == "__main__":
  unittest.main()
