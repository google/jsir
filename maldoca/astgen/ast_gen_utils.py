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
"""Port of maldoca/astgen/ast_gen_utils.h to Python."""

from __future__ import annotations

import dataclasses
import posixpath
from typing import Callable, Optional

from maldoca.astgen.ast_def import FieldDef
from maldoca.astgen.printer_base import Printer

JSON_VALUE_VARIABLE_NAME = "json"
OS_VALUE_VARIABLE_NAME = "os"


def get_ast_header_path(ast_path: str) -> str:
  return posixpath.join(ast_path, "ast.generated.h")


@dataclasses.dataclass
class TabPrinterOptions:
  print_prefix: Optional[Callable[[], None]] = None
  print_separator: Optional[Callable[[], None]] = None
  print_postfix: Optional[Callable[[], None]] = None


class TabPrinter:
  """Prints a prefix before the first call to `print()`, a separator before

  every subsequent call, and (via the context manager) a postfix after the
  last call -- but only if `print()` was called at least once.

  Usage:
    with TabPrinter(options) as tab:
      for item in items:
        tab.print()
        PrintItem(item)
  """

  def __init__(self, options: TabPrinterOptions):
    self._options = options
    self._is_first = True

  def print(self) -> None:
    if self._is_first:
      if self._options.print_prefix:
        self._options.print_prefix()
      self._is_first = False
    else:
      if self._options.print_separator:
        self._options.print_separator()

  def __enter__(self) -> "TabPrinter":
    return self

  def __exit__(self, *exc_info) -> None:
    if not self._is_first and self._options.print_postfix:
      self._options.print_postfix()


@dataclasses.dataclass
class IfStmtCase:
  condition: Callable[[], None]
  body: Callable[[], None]


class IfStmtPrinter:
  """Helper for printing an if-statement.

  Usage:
    printer = IfStmtPrinter(p)
    printer.print_case(IfStmtCase(
        condition=lambda: PrintConditionHere(),
        body=lambda: PrintBodyHere(),
    ))
    printer.print_case(IfStmtCase(
        condition=lambda: PrintAnotherConditionHere(),
        body=lambda: PrintAnotherBodyHere(),
    ))

  This helper adds the "else" keyword to all subsequent cases.
  """

  def __init__(self, printer: Printer):
    self._printer = printer
    self._is_first = True

  def print_case(self, case: IfStmtCase) -> None:
    if self._is_first:
      self._printer.print("if (")
      self._is_first = False
    else:
      self._printer.print(" else if (")
    case.condition()
    self._printer.print(") {\n")
    with self._printer.with_indent():
      case.body()
    self._printer.print("}")


def un_indented_source(source: str) -> str:
  """Consistently unindents lines of code so the outmost line has no indent.

  Example:

  Input:
  ```
    abc
      abc
     abc
  ```

  Output:
  ```
  abc
    abc
   abc
  ```
  """
  source = source.rstrip()
  lines = source.split("\n")

  # Remove leading empty lines.
  first_non_empty = 0
  while first_non_empty < len(lines) and lines[first_non_empty] == "":
    first_non_empty += 1
  lines = lines[first_non_empty:]

  min_indent: Optional[int] = None
  for line in lines:
    leading_spaces = len(line) - len(line.lstrip(" "))
    if leading_spaces == len(line):
      # Line is empty or all-spaces; doesn't constrain the indent.
      continue
    if min_indent is None or leading_spaces < min_indent:
      min_indent = leading_spaces
  if min_indent is None:
    min_indent = 0

  return "\n".join(
      line[min_indent:] if len(line) >= min_indent else line for line in lines
  )


# FieldIs{Argument,Region}:
#
# If a field has ignore_in_ir, then we don't define anything in the op.
#
# Example: Node.start does not lead to any argument/region in JSIR because we
# want to store the information in mlir::Location.
#
# If a field has enclose_in_region, then it's an MLIR "region"; otherwise
# it's an MLIR "argument".
#
# An argument is either an mlir::Attribute or an mlir::Value;
# A region is an mlir::Region.
#
# See FieldDefPb.enclose_in_region for why we need to enclose certain fields
# in a region.
def field_is_argument(field: FieldDef) -> bool:
  return not field.ignore_in_ir and not field.enclose_in_region


def field_is_region(field: FieldDef) -> bool:
  return not field.ignore_in_ir and field.enclose_in_region
