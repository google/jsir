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
 * @fileoverview UI logic of the in-browser JSIR demo: parses the left editor's
 * JavaScript with Babel and shows the corresponding JSHIR (computed by the
 * WebAssembly module) in the right editor.
 */

import {MainModule} from './jsir_wasm/jsir_wasm_bin';
import {
  entriesAtIrOffset,
  entriesAtSourceOffset,
  parseSourceMap,
  SourceMapEntry,
} from './source_map';

/**
 * Emscripten module factory defined by jsir_wasm_bin.js (see EXPORT_NAME in
 * the BUILD file).
 */
declare function loadJsirWasm(): Promise<MainModule>;

const EXAMPLES: {[name: string]: string} = {
  'Hello world': `console.log("Hello, JSIR!");\n`,
  'Functions & closures': `function makeCounter(start) {
  let count = start;
  return function () {
    count += 1;
    return count;
  };
}

const counter = makeCounter(10);
counter();
`,
  'Control flow': `function classify(n) {
  if (n < 0) {
    return "negative";
  } else if (n === 0) {
    return "zero";
  }
  for (let i = 0; i < n; i++) {
    if (i % 2 === 0) continue;
    console.log(i);
  }
  while (n > 100) n = n / 2;
  return n > 50 ? "big" : "small";
}
`,
  'Objects & classes': `class Point {
  constructor(x, y) {
    this.x = x;
    this.y = y;
  }
  get norm() {
    return Math.sqrt(this.x ** 2 + this.y ** 2);
  }
}

const { x, ...rest } = { x: 1, y: 2, z: 3 };
const p = new Point(3, 4);
`,
  'Async & generators': `async function fetchAll(urls) {
  const results = [];
  for await (const r of urls.map((u) => fetch(u))) {
    results.push(await r.json());
  }
  return results;
}

function* range(n) {
  for (let i = 0; i < n; i++) yield i;
}
`,
  'Obfuscated-looking': `var _0x1a2b = ["\\x68\\x65\\x6c\\x6c\\x6f", "log"];
(function (a, b) {
  console[_0x1a2b[1]](a + b);
})(_0x1a2b[0], 42);
`,
};

// ----------------------------------------------------------------------------
// Editor abstraction: Monaco if available, <textarea> fallback otherwise.

interface Editor {
  getValue(): string;
  setValue(value: string): void;
  setLanguage(language: string): void;
  onChange(callback: () => void): void;
  /** The underlying Monaco editor, if any. */
  monaco?: monaco.editor.IStandaloneCodeEditor;
}

interface EditorOptions {
  value: string;
  language?: string;
  readOnly?: boolean;
}

function makeTextareaEditor(
  container: HTMLElement,
  {value, readOnly}: EditorOptions,
): Editor {
  const textarea = document.createElement('textarea');
  textarea.value = value;
  textarea.readOnly = !!readOnly;
  textarea.spellcheck = false;
  container.appendChild(textarea);
  return {
    getValue: () => textarea.value,
    setValue: (v) => {
      textarea.value = v;
    },
    setLanguage: () => {},
    onChange: (callback) => {
      textarea.addEventListener('input', callback);
    },
  };
}

function registerMlirLanguage() {
  monaco.languages.register({id: 'mlir'});
  monaco.languages.setMonarchTokensProvider('mlir', {
    tokenizer: {
      root: [
        [/\/\/.*$/, 'comment'],
        [/"([^"\\]|\\.)*"/, 'string'],
        [/%[\w.$-]+(#\d+)?/, 'variable'],
        [/\^[\w.$-]+/, 'type.identifier'],
        [/@[\w.$-]+/, 'annotation'],
        [/#[\w.$-]+/, 'annotation'],
        [/!\w[\w.$-]*/, 'type'],
        [/\b(jsir|jshir|func|cf|builtin)\.[\w.]+/, 'keyword'],
        [/\b(loc|unknown|true|false)\b/, 'keyword'],
        [/\b[if]\d+\b|\bindex\b|\bnone\b/, 'type'],
        [/-?\d+(\.\d+)?(e[+-]?\d+)?/, 'number'],
      ],
    },
  });
}

/**
 * Loads Monaco through its AMD loader (vs/loader.js), which defines the global
 * `monaco` namespace. Monaco is served from the same origin (see server.py), so
 * its web workers need no extra setup.
 */
function loadMonaco(): Promise<void> {
  return new Promise((resolve, reject) => {
    if (typeof require !== 'function') {
      reject(new Error('Monaco loader unavailable'));
      return;
    }
    require.config({paths: {'vs': 'vs'}});
    require(['vs/editor/editor.main'], () => {
      resolve();
    }, reject);
  });
}

function makeMonacoEditor(
  container: HTMLElement,
  {value, language, readOnly}: EditorOptions,
): Editor {
  const editor = monaco.editor.create(container, {
    value,
    language,
    readOnly: !!readOnly,
    theme: 'vs-dark',
    automaticLayout: true,
    minimap: {enabled: false},
    fontSize: 13,
    scrollBeyondLastLine: false,
    wordWrap: readOnly ? 'off' : 'on',
  });
  return {
    getValue: () => editor.getValue(),
    setValue: (v) => {
      if (editor.getValue() === v) return;
      const scrollTop = editor.getScrollTop();
      editor.setValue(v);
      editor.setScrollTop(scrollTop);
    },
    setLanguage: (language) => {
      const model = editor.getModel();
      if (model) monaco.editor.setModelLanguage(model, language);
    },
    onChange: (callback) => {
      editor.onDidChangeModelContent(callback);
    },
    monaco: editor,
  };
}

// ----------------------------------------------------------------------------

/** Returns the element with the given ID, checking that it is a `type`. */
function getElement<T extends HTMLElement>(
  id: string,
  type: new () => T,
): T {
  const element = document.getElementById(id);
  if (!(element instanceof type)) {
    throw new Error(`Missing or unexpected element #${id}`);
  }
  return element;
}

function setStatus(kind: string, text: string, timing = '') {
  getElement('status', HTMLElement).className = kind;
  getElement('status-text', HTMLElement).textContent = text;
  getElement('timing', HTMLElement).textContent = timing;
}

interface BabelParseError {
  name?: string;
  message?: string;
  loc?: {line: number; column: number};
}

/** Parses `source` with Babel and returns the AST as a JSON string. */
function babelParse(source: string, sourceType: string): string {
  const options = {
    'computeScopes': true,
    'createParenthesizedExpressions': true,
    'errorRecovery': false,
    'sourceType': sourceType,
  };
  const {ast, response} = window.exports.parse(source, JSON.stringify(options));
  if (!ast) {
    const errors = (JSON.parse(response) as {errors?: BabelParseError[]})[
      'errors'
    ];
    const err: BabelParseError = (errors && errors[0]) || {};
    const loc = err['loc'] ? ` (${err['loc'].line}:${err['loc'].column})` : '';
    throw new Error(
      `Babel ${err['name'] || 'error'}${loc}: ${err['message']}`,
    );
  }
  return ast;
}

// ----------------------------------------------------------------------------
// Linking IR ops and JavaScript source on hover.

/**
 * Highlights an op and its JavaScript source when the mouse hovers over
 * either of them.
 */
function linkEditorsOnHover(
  input: monaco.editor.IStandaloneCodeEditor,
  output: monaco.editor.IStandaloneCodeEditor,
  getEntries: () => SourceMapEntry[],
) {
  const inputDecorations = input.createDecorationsCollection();
  const outputDecorations = output.createDecorationsCollection();

  function toRange(
    editor: monaco.editor.IStandaloneCodeEditor,
    start: number,
    end: number,
  ): monaco.Range | undefined {
    const model = editor.getModel();
    if (!model) return undefined;
    return monaco.Range.fromPositions(
      model.getPositionAt(start),
      model.getPositionAt(end),
    );
  }

  function decorate(
    editor: monaco.editor.IStandaloneCodeEditor,
    decorations: monaco.editor.IEditorDecorationsCollection,
    spans: Array<[number, number]>,
    reveal: boolean,
  ) {
    const ranges: monaco.Range[] = [];
    for (const [start, end] of spans) {
      const range = toRange(editor, start, end);
      if (range) ranges.push(range);
    }
    decorations.set(
      ranges.map((range) => ({
        range,
        options: {className: 'linked-highlight'},
      })),
    );
    if (reveal && ranges.length > 0) {
      editor.revealRangeInCenterIfOutsideViewport(
        ranges[0],
        monaco.editor.ScrollType.Smooth,
      );
    }
  }

  function highlight(entries: SourceMapEntry[], hovered: 'input' | 'output') {
    decorate(
      input,
      inputDecorations,
      entries.map((e) => [e.srcStart, e.srcEnd]),
      hovered === 'output',
    );
    decorate(
      output,
      outputDecorations,
      entries.map((e) => [e.irStart, e.irEnd]),
      hovered === 'input',
    );
  }

  function clear() {
    inputDecorations.clear();
    outputDecorations.clear();
  }

  function offsetAt(
    editor: monaco.editor.IStandaloneCodeEditor,
    event: monaco.editor.IEditorMouseEvent,
  ): number | undefined {
    const model = editor.getModel();
    const position = event.target.position;
    if (!model || !position) return undefined;
    // Ignore the empty space past the end of a line.
    if (position.column >= model.getLineMaxColumn(position.lineNumber)) {
      return undefined;
    }
    return model.getOffsetAt(position);
  }

  input.onMouseMove((event) => {
    const offset = offsetAt(input, event);
    if (offset === undefined) return clear();
    highlight(entriesAtSourceOffset(getEntries(), offset), 'input');
  });
  output.onMouseMove((event) => {
    const offset = offsetAt(output, event);
    if (offset === undefined) return clear();
    highlight(entriesAtIrOffset(getEntries(), offset), 'output');
  });
  input.onMouseLeave(clear);
  output.onMouseLeave(clear);
  input.onDidChangeModelContent(clear);
  output.onDidChangeModelContent(clear);
}

// ----------------------------------------------------------------------------

async function main() {
  const initialSource = EXAMPLES['Functions & closures'];

  let useMonaco = false;
  try {
    await loadMonaco();
    registerMlirLanguage();
    useMonaco = true;
  } catch (e: unknown) {
    console.warn('Falling back to <textarea> editors:', e);
  }

  const inputContainer = getElement('input', HTMLElement);
  const outputContainer = getElement('output', HTMLElement);
  const input = useMonaco
    ? makeMonacoEditor(inputContainer, {
        value: initialSource,
        language: 'javascript',
      })
    : makeTextareaEditor(inputContainer, {value: initialSource});
  const output = useMonaco
    ? makeMonacoEditor(outputContainer, {
        value: '',
        language: 'mlir',
        readOnly: true,
      })
    : makeTextareaEditor(outputContainer, {value: '', readOnly: true});

  const examplesEl = getElement('examples', HTMLSelectElement);
  for (const name of Object.keys(EXAMPLES)) {
    const option = document.createElement('option');
    option.value = name;
    option.textContent = name;
    examplesEl.appendChild(option);
  }
  examplesEl.addEventListener('change', () => {
    if (examplesEl.value) input.setValue(EXAMPLES[examplesEl.value]);
    examplesEl.value = '';
  });

  const sourceTypeEl = getElement('source-type', HTMLSelectElement);
  const outputKindEl = getElement('output-kind', HTMLSelectElement);
  const printLocationsEl = getElement('print-locations', HTMLInputElement);
  const outputTitleEl = getElement('output-title', HTMLElement);

  const wasm = await loadJsirWasm();

  // Ops in the output and the source ranges they came from. Empty whenever the
  // output does not correspond to the current source.
  let sourceMap: SourceMapEntry[] = [];
  if (input.monaco && output.monaco) {
    linkEditorsOnHover(input.monaco, output.monaco, () => sourceMap);
  }

  function update() {
    sourceMap = [];
    const showAst = outputKindEl.value === 'ast';
    outputTitleEl.textContent = showAst ? 'Babel AST JSON' : 'JSHIR';
    output.setLanguage(showAst ? 'json' : 'mlir');

    let astJson: string;
    const t0 = performance.now();
    try {
      astJson = babelParse(input.getValue(), sourceTypeEl.value);
    } catch (e: unknown) {
      setStatus('error', e instanceof Error ? e.message : String(e));
      return;
    }
    const t1 = performance.now();

    if (showAst) {
      output.setValue(JSON.stringify(JSON.parse(astJson), null, 2));
      setStatus('', 'Parsed with Babel.', `babel ${(t1 - t0).toFixed(1)} ms`);
      return;
    }

    let result;
    try {
      result = wasm.astJsonToJsir(astJson, printLocationsEl.checked);
    } catch (e: unknown) {
      setStatus('error', `WebAssembly error: ${e}`);
      return;
    }
    const t2 = performance.now();
    const timing =
      `babel ${(t1 - t0).toFixed(1)} ms · ` +
      `jsir (wasm) ${(t2 - t1).toFixed(1)} ms`;
    if (result.ok) {
      const ir = String(result.output);
      output.setValue(ir);
      sourceMap = parseSourceMap(ir, String(result.sourceMap));
      setStatus('', 'OK', timing);
    } else {
      setStatus('error', String(result.output), timing);
    }
  }

  let timer: number | undefined;
  input.onChange(() => {
    sourceMap = [];
    clearTimeout(timer);
    timer = setTimeout(update, 250);
  });
  for (const el of [sourceTypeEl, outputKindEl, printLocationsEl]) {
    el.addEventListener('change', update);
  }
  update();
}

main().catch((e: unknown) => {
  console.error(e);
  setStatus('error', `Initialization failed: ${e}`);
});
