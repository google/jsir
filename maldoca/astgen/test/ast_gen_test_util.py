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
"""Port of maldoca/astgen/test/ast_gen_test_util.{h,cc} to Python.

Golden-diff test harness shared by all 9 test-case directories under
maldoca/astgen/test/: loads a case's `ast_def.textproto`, runs each
printer, and (whitespace-stripped) compares the output against the
checked-in golden file. A case that doesn't set a given `expected_*_path`
skips that comparison, mirroring the C++ `std::optional` fields on
`AstGenTestParam` (see e.g. `multiple_inheritance` and `union`, which have
no MLIR/IR side, or `lambda`, which has no `ir_to_ast` golden).
"""

from __future__ import annotations

import dataclasses
import os
from typing import Optional
import unittest

from google.protobuf import text_format

from maldoca.astgen import ast_def_pb2
from maldoca.astgen.ast_def import AstDef
from maldoca.astgen.ast_from_json_printer import print_ast_from_json
from maldoca.astgen.ast_header_printer import print_ast_header
from maldoca.astgen.ast_serialize_printer import print_ast_to_json
from maldoca.astgen.ast_source_printer import print_ast_source
from maldoca.astgen.ast_to_ir_source_printer import print_ast_to_ir_source
from maldoca.astgen.ir_table_gen_printer import print_ir_table_gen
from maldoca.astgen.ir_to_ast_source_printer import print_ir_to_ast_source
from maldoca.astgen.ts_interface_printer import print_ts_interface


@dataclasses.dataclass
class AstGenTestParam:
  # Absolute path to the test-case directory (holds ast_def.textproto and
  # all the golden files). Concrete test files set this to
  # os.path.dirname(os.path.abspath(__file__)).
  test_dir: str

  # cc_namespace/ast_path/ir_path mirror the real invocation in
  # ast_gen_main.cc's doc comment: the *canonical* (not filesystem) path,
  # e.g. "maldoca/astgen/test/lambda" -- baked into header guards and
  # #includes in the printed output, so it must match what originally
  # produced the golden files.
  cc_namespace: str
  ast_path: str
  # Defaults to "" (not None) to match the C++ AstGenTestParam::ir_path,
  # which is a plain std::string (not std::optional) -- test cases that
  # don't set it (e.g. multiple_inheritance, typed_lambda) still have the
  # IR printers invoked with an empty ir_path (a no-op there, since those
  # schemas never set should_generate_ir_op/kinds), just with no golden to
  # compare against. See union for a case that sets ir_path but still has
  # no IR goldens -- its printers *do* run, just unchecked.
  ir_path: str = ""

  # Filenames (relative to test_dir) of the golden files. None means "don't
  # check this printer for this test case".
  ts_interface_path: Optional[str] = None
  expected_ast_header_path: Optional[str] = None
  expected_ast_source_path: Optional[str] = None
  expected_ast_to_json_path: Optional[str] = None
  expected_ast_from_json_path: Optional[str] = None
  expected_ir_tablegen_path: Optional[str] = None
  expected_ast_to_ir_source_path: Optional[str] = None
  expected_ir_to_ast_source_path: Optional[str] = None


class AstGenTest(unittest.TestCase):
  """Base class mirroring test/ast_gen_test_util.h's AstGenTest fixture.

  Concrete subclasses (one per test-case directory) set the `PARAM` class
  attribute to an `AstGenTestParam`.
  """

  PARAM: AstGenTestParam

  def _read(self, relative_path: str) -> str:
    with open(os.path.join(self.PARAM.test_dir, relative_path)) as f:
      return f.read()

  def _load_ast_def(self) -> AstDef:
    pb = ast_def_pb2.AstDefPb()
    text_format.Parse(self._read("ast_def.textproto"), pb)
    return AstDef.from_proto(pb)

  def _assert_matches_golden(
      self, actual: str, expected_relative_path: Optional[str]
  ) -> None:
    if expected_relative_path is None:
      return
    expected = self._read(expected_relative_path)
    self.assertEqual(actual.strip(), expected.strip())

  def test_print_ts_interface(self) -> None:
    ast_def = self._load_ast_def()
    actual = print_ts_interface(ast_def)
    self._assert_matches_golden(actual, self.PARAM.ts_interface_path)

  def test_ast_hdr(self) -> None:
    ast_def = self._load_ast_def()
    actual = print_ast_header(
        ast_def, self.PARAM.cc_namespace, self.PARAM.ast_path
    )
    self._assert_matches_golden(actual, self.PARAM.expected_ast_header_path)

  def test_ast_src(self) -> None:
    ast_def = self._load_ast_def()
    actual = print_ast_source(
        ast_def, self.PARAM.cc_namespace, self.PARAM.ast_path
    )
    self._assert_matches_golden(actual, self.PARAM.expected_ast_source_path)

  def test_ast_to_json(self) -> None:
    ast_def = self._load_ast_def()
    actual = print_ast_to_json(
        ast_def, self.PARAM.cc_namespace, self.PARAM.ast_path
    )
    self._assert_matches_golden(
        actual, self.PARAM.expected_ast_to_json_path
    )

  def test_ast_from_json(self) -> None:
    ast_def = self._load_ast_def()
    actual = print_ast_from_json(
        ast_def, self.PARAM.cc_namespace, self.PARAM.ast_path
    )
    self._assert_matches_golden(
        actual, self.PARAM.expected_ast_from_json_path
    )

  def test_ir_table_gen(self) -> None:
    ast_def = self._load_ast_def()
    actual = print_ir_table_gen(ast_def, self.PARAM.ir_path)
    self._assert_matches_golden(
        actual, self.PARAM.expected_ir_tablegen_path
    )

  def test_ast_to_ir(self) -> None:
    ast_def = self._load_ast_def()
    actual = print_ast_to_ir_source(
        ast_def, self.PARAM.cc_namespace, self.PARAM.ast_path,
        self.PARAM.ir_path,
    )
    self._assert_matches_golden(
        actual, self.PARAM.expected_ast_to_ir_source_path
    )

  def test_ir_to_ast(self) -> None:
    ast_def = self._load_ast_def()
    actual = print_ir_to_ast_source(
        ast_def, self.PARAM.cc_namespace, self.PARAM.ast_path,
        self.PARAM.ir_path,
    )
    self._assert_matches_golden(
        actual, self.PARAM.expected_ir_to_ast_source_path
    )
