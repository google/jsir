/**
 * @license
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @fileoverview Declarations for globals that index.html loads via plain
 * <script> tags before main.ts runs (Monaco has its own typings). Being ambient
 * declarations, these are turned into Closure externs, so property names are
 * never renamed.
 */

/**
 * JSIR's Babel glue (quickjs_babel/native.js), which assigns a global
 * `exports` object.
 */
declare interface JsirBabelGlue {
  /**
   * Parses `source` with Babel. `options` is a JSON-encoded BabelParseRequest.
   * Returns the AST as a JSON string (absent on failure) and a JSON-encoded
   * BabelParseResponse.
   */
  parse(source: string, options: string): {ast?: string; response: string};
}

declare interface Window {
  exports: JsirBabelGlue;
}
