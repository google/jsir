# JSIR in the Browser (WebAssembly)

An interactive page: type JavaScript on the left, see the corresponding JSIR
(JSHIR) on the right, updated as you type. Hovering over an op highlights the
JavaScript it came from; hovering over JavaScript highlights the ops converted
from it.

```
 ┌──────────── browser ───────────────────────────────────────────────┐
 │  JS source                                                          │
 │     │  Babel standalone + JSIR's Babel glue (quickjs_babel/native.js)│
 │     ▼                                                               │
 │  Babel AST JSON (with scopes, comment UIDs, …)                      │
 │     │  jsir_wasm.cc  (C++ → WebAssembly via emscripten)             │
 │     ▼    JsFile::FromJson → AstToJshirFile → print                  │
 │  JSHIR text                                                         │
 └─────────────────────────────────────────────────────────────────────┘
```

Babel is already JavaScript, so it runs natively in the page. Only the C++ half
of JSIR (AST deserialization, AST → JSIR conversion, MLIR printing) is compiled
with emscripten. The page reuses the exact glue code JSIR runs inside QuickJS,
so the AST JSON is identical to what the native pipeline sees.

## Running

```bash
bazel run //maldoca/js/ir/wasm:server -- 8080
```

Then open `http://localhost:8080`.

The first build downloads the Emscripten SDK and compiles LLVM/MLIR to
WebAssembly, which takes a while. The demo's targets are tagged `manual`, so
`bazel build //...` skips them.

The Monaco editor (npm package `monaco-editor`) is served by `server.py` along
with everything else. If Monaco can't load, the page falls back to plain
`<textarea>`s.

## GitHub Pages

The demo is a static site, deployed to https://google.github.io/jsir/ by the
`deploy_demo.yml` workflow on every push to `main`. To build it locally:

```bash
bazel run //maldoca/js/ir/wasm:stage_site -- /tmp/jsir_site
```

## Files

-   `jsir_wasm.cc`: Embind entry point `astJsonToJsir(astJson, printLocations)`.
    Built with `MODULARIZE`, so the page creates it via `loadJsirWasm()`.
    Besides the printed JSHIR, it returns a source map: where each op starts in
    the output, and the source range of the AST node it was converted from.
-   `index.html`: Page layout and styles; loads the scripts below.
-   `main.ts`: UI logic (Monaco, examples, source-type/output toggles),
    bundled by esbuild into `app.js`.
-   `source_map.ts`: Turns the source map into IR/source ranges, and finds the
    ops to highlight on hover.
-   `globals.d.ts`: Declarations for JSIR's Babel glue, which the page loads
    via a `<script>` tag (Monaco has its own typings).
-   `site.py`: The files of the site (the page, the wasm build, Babel,
    `native.js` and Monaco), located in runfiles.
-   `server.py`: Serves the site locally.
-   `stage_site.py`: Copies the site into a directory, e.g. for GitHub Pages.
