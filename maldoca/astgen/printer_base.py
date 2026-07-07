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
"""Port of maldoca/astgen/printer_base.h to Python.

The C++ printers are built on `google::protobuf::io::Printer`: `$var$`
substitution (with `$$` as a literal `$`), 2-space-per-level indentation
that's automatically applied at the start of each output line (but *not* to
otherwise-blank lines), and a `WithVars`/`WithIndent` RAII scope mechanism.
Python has no equivalent built in, so `Printer` below reimplements the
subset of that API the astgen printers actually use.
"""

from __future__ import annotations

import contextlib
import re

_VAR_PATTERN = re.compile(r"\$(\w*)\$")


class Printer:
  """Port of `AstGenPrinterBase` (which itself extends protobuf's Printer)."""

  def __init__(self):
    self._chunks: list[str] = []
    # Each entry is the width (in spaces) of one active `with_indent()`
    # scope; the current indent is their sum. Most scopes are 2 spaces, but
    # e.g. AstHeaderPrinter.print_constructor() uses a 4-space scope for
    # wrapped constructor argument lists.
    self._indent_stack: list[int] = []
    self._at_line_start = True
    self._var_stack: list[dict[str, str]] = []

  # ===========================================================================
  # Print() / Println()
  # ===========================================================================

  def print(self, text: str = "", **variables: str) -> None:
    self._write_raw(self._substitute(text, variables))

  def println(self, text: str = "", **variables: str) -> None:
    self.print(text, **variables)
    self._write_raw("\n")

  def _substitute(self, text: str, inline_variables: dict[str, str]) -> str:
    resolved: dict[str, str] = {}
    for scope in self._var_stack:
      resolved.update(scope)
    resolved.update(inline_variables)

    def repl(m: re.Match[str]) -> str:
      name = m.group(1)
      if not name:
        return "$"
      if name not in resolved:
        raise KeyError(f"Unknown variable: ${name}$")
      return str(resolved[name])

    return _VAR_PATTERN.sub(repl, text)

  def _write_raw(self, data: str) -> None:
    lines = data.split("\n")
    for i, line in enumerate(lines):
      if i > 0:
        self._chunks.append("\n")
        self._at_line_start = True
      if line:
        if self._at_line_start:
          self._chunks.append(" " * sum(self._indent_stack))
          self._at_line_start = False
        self._chunks.append(line)

  # ===========================================================================
  # WithIndent() / WithVars()
  # ===========================================================================

  @contextlib.contextmanager
  def with_indent(self, width: int = 2):
    self._indent_stack.append(width)
    try:
      yield self
    finally:
      self._indent_stack.pop()

  # Raw (non-scoped) indent/outdent, for the rare case where the indent
  # doesn't nest cleanly within a single Python `with` block -- e.g. a
  # TabPrinter whose prefix callback indents and whose postfix callback
  # (which may run in a different lexical scope) outdents.
  def indent(self, width: int = 2) -> None:
    self._indent_stack.append(width)

  def outdent(self) -> None:
    self._indent_stack.pop()

  @contextlib.contextmanager
  def with_vars(self, variables: dict[str, str]):
    self._var_stack.append(variables)
    try:
      yield self
    finally:
      self._var_stack.pop()

  # ===========================================================================
  # Output
  # ===========================================================================

  def content(self) -> str:
    return "".join(self._chunks)
