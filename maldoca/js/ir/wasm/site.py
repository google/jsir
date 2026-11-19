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

"""The files that make up the JSIR WebAssembly demo site."""

import os

from python.runfiles import runfiles

# Site path -> runfiles path.
_FILES = {
    "index.html": "_main/maldoca/js/ir/wasm/index.html",
    "app.js": "_main/maldoca/js/ir/wasm/app.js",
    # The .wasm is embedded in its JS glue (SINGLE_FILE=1).
    "jsir_wasm_bin.js": "_main/maldoca/js/ir/wasm/jsir_wasm/jsir_wasm_bin.js",
    "babel.min.js": "_main/third_party/babel_standalone/babel.min.js",
    "native.js": "_main/maldoca/js/quickjs_babel/native.js",
}

# Monaco is a whole directory tree (`vs/`) that its AMD loader fetches from.
_MONACO_LOADER = "monaco_editor/package/min/vs/loader.js"

_RUNFILES = runfiles.Create()


def _resolve(path: str) -> str:
  resolved = _RUNFILES.Rlocation(path)
  if resolved is None or not os.path.exists(resolved):
    raise FileNotFoundError(path)
  return resolved


def site_files() -> dict[str, str]:
  """Returns a map from each path of the site to the file that provides it."""
  files = {name: _resolve(path) for name, path in _FILES.items()}
  vs_dir = os.path.dirname(_resolve(_MONACO_LOADER))
  for root, _, names in os.walk(vs_dir, followlinks=True):
    for name in names:
      path = os.path.join(root, name)
      files[os.path.join("vs", os.path.relpath(path, vs_dir))] = path
  return files
