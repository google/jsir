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

"""Local web server for the JSIR WebAssembly browser demo."""

import http.server
import os
import socket
import socketserver
import sys
import tempfile

from maldoca.js.ir.wasm import site


def main():
  port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080

  serve_dir = tempfile.mkdtemp(prefix="jsir_wasm_demo_")
  for name, path in site.site_files().items():
    link = os.path.join(serve_dir, name)
    os.makedirs(os.path.dirname(link), exist_ok=True)
    os.symlink(path, link)

  class Handler(http.server.SimpleHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
      super().__init__(*args, directory=serve_dir, **kwargs)

    def end_headers(self):
      # Always serve fresh builds.
      self.send_header("Cache-Control", "no-store")
      super().end_headers()

  hostname = socket.getfqdn()
  print("=" * 60)
  print("JSIR WebAssembly Browser Demo running at:")
  print(f"   Local:    http://localhost:{port}")
  print(f"   Cloudtop: http://{hostname}:{port}")
  print("=" * 60)

  socketserver.TCPServer.allow_reuse_address = True
  with socketserver.TCPServer(("", port), Handler) as httpd:
    try:
      httpd.serve_forever()
    except KeyboardInterrupt:
      print("\nShutting down server.")


if __name__ == "__main__":
  main()
