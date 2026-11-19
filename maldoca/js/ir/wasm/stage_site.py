# Copyright 2026 Google LLC
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

"""Copies the JSIR WebAssembly demo site into a directory.

The result is a static site, e.g. for GitHub Pages:

  bazel run //maldoca/js/ir/wasm:stage_site -- <output directory>

The output directory must be empty or not exist yet. A relative path is
relative to the directory `bazel run` was invoked from.
"""

import os
import shutil
import sys

from maldoca.js.ir.wasm import site


def main():
  if len(sys.argv) != 2:
    sys.exit(f"Usage: {sys.argv[0]} <output directory>")

  dest = os.path.join(
      os.environ.get("BUILD_WORKING_DIRECTORY", os.getcwd()), sys.argv[1]
  )
  if os.path.exists(dest) and os.listdir(dest):
    sys.exit(f"{dest} is not empty.")

  files = site.site_files()
  for name, path in files.items():
    target = os.path.join(dest, name)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    # Bazel outputs are read-only; the copies shouldn't be.
    shutil.copyfile(path, target)

  print(f"Copied {len(files)} files to {dest}")


if __name__ == "__main__":
  main()
