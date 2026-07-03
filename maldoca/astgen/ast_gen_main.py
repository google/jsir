#!/usr/bin/env python3
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
"""Port of maldoca/astgen/ast_gen_main.cc to Python.

  bazel build //maldoca/astgen:ast_gen_main_py

  ./bazel-bin/maldoca/astgen/ast_gen_main_py \
     --ast_def_path="maldoca/js/ast/ast_def.textproto" \
     --cc_namespace="maldoca" \
     --ast_path="maldoca/js/ast" \
     --ir_path="maldoca/js/ir"
"""

from __future__ import annotations

import argparse
import os
import sys

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


def _write_file(path: str, content: str) -> None:
  print(f"Writing to {path}")
  with open(path, "w") as f:
    f.write(content)


def _ast_gen_main(
    ast_def_path: str, cc_namespace: str, ast_path: str, ir_path: str
) -> None:
  with open(ast_def_path) as f:
    pb = ast_def_pb2.AstDefPb()
    text_format.Parse(f.read(), pb)
  ast_def = AstDef.from_proto(pb)

  ast_hdr = print_ast_header(ast_def, cc_namespace, ast_path)
  _write_file(os.path.join(ast_path, "ast.generated.h"), ast_hdr)

  ast_src = print_ast_source(ast_def, cc_namespace, ast_path)
  _write_file(os.path.join(ast_path, "ast.generated.cc"), ast_src)

  ast_to_json = print_ast_to_json(ast_def, cc_namespace, ast_path)
  _write_file(
      os.path.join(ast_path, "ast_to_json.generated.cc"), ast_to_json
  )

  ast_from_json = print_ast_from_json(ast_def, cc_namespace, ast_path)
  _write_file(
      os.path.join(ast_path, "ast_from_json.generated.cc"), ast_from_json
  )

  if ir_path:
    ir_tablegen = print_ir_table_gen(ast_def, ir_path)
    _write_file(
        os.path.join(ir_path, f"{ast_def.lang_name}ir_ops.generated.td"),
        ir_tablegen,
    )

    ast_to_ir = print_ast_to_ir_source(
        ast_def, cc_namespace, ast_path, ir_path
    )
    _write_file(
        os.path.join(
            ir_path, "conversion", f"ast_to_{ast_def.lang_name}ir.generated.cc"
        ),
        ast_to_ir,
    )

    ir_to_ast = print_ir_to_ast_source(
        ast_def, cc_namespace, ast_path, ir_path
    )
    _write_file(
        os.path.join(
            ir_path, "conversion", f"{ast_def.lang_name}ir_to_ast.generated.cc"
        ),
        ir_to_ast,
    )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--ast_def_path",
      required=True,
      help="The path to the ast_def.textproto file.",
  )
  parser.add_argument(
      "--cc_namespace",
      required=True,
      help="The C++ namespace for the AST classes in C++.",
  )
  parser.add_argument(
      "--ast_path",
      required=True,
      help="The directory for the AST code in C++.",
  )
  parser.add_argument(
      "--ir_path",
      default="",
      help="The directory for the IR code in TableGen and C++.",
  )
  args = parser.parse_args()

  try:
    _ast_gen_main(
        args.ast_def_path, args.cc_namespace, args.ast_path, args.ir_path
    )
  except (OSError, ValueError, text_format.ParseError) as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
  main()
