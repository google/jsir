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
 * @fileoverview Maps between ops in the printed JSIR and the JavaScript source
 * they were converted from.
 */

/** An op in the printed IR and the JavaScript it was converted from. */
export interface SourceMapEntry {
  /** [irStart, irEnd): offsets of the op's text in the printed IR. */
  irStart: number;
  irEnd: number;
  /** [srcStart, srcEnd): offsets of the op's AST node in the source. */
  srcStart: number;
  srcEnd: number;
}

/**
 * Returns the offset just past the op whose text starts at `start` in `ir`:
 * the first newline outside of brackets (regions, operand lists, attribute
 * dictionaries) and string literals. `<`/`>` are not tracked, since `->`
 * appears in function types.
 */
export function findOpEnd(ir: string, start: number): number {
  let depth = 0;
  for (let i = start; i < ir.length; i++) {
    const c = ir[i];
    if (c === '"') {
      for (i++; i < ir.length && ir[i] !== '"'; i++) {
        if (ir[i] === '\\') i++;
      }
    } else if (c === '(' || c === '[' || c === '{') {
      depth++;
    } else if (c === ')' || c === ']' || c === '}') {
      depth--;
    } else if (c === '\n' && depth <= 0) {
      return i;
    }
  }
  return ir.length;
}

/**
 * Converts the `sourceMap` returned by the wasm module (JSON array of
 * [irLine (1-based), irColumn (0-based), srcStart, srcEnd]) into offsets.
 */
export function parseSourceMap(ir: string, json: string): SourceMapEntry[] {
  const lineStarts = [0];
  for (let i = 0; i < ir.length; i++) {
    if (ir[i] === '\n') lineStarts.push(i + 1);
  }
  const entries: SourceMapEntry[] = [];
  for (const [line, column, srcStart, srcEnd] of JSON.parse(json) as Array<
    [number, number, number, number]
  >) {
    if (line < 1 || line > lineStarts.length) continue;
    const irStart = lineStarts[line - 1] + column;
    entries.push({irStart, irEnd: findOpEnd(ir, irStart), srcStart, srcEnd});
  }
  return entries;
}

/** Returns the innermost (shortest) entry whose `[start, end)` has `offset`. */
function innermost(
  entries: SourceMapEntry[],
  offset: number,
  start: (e: SourceMapEntry) => number,
  end: (e: SourceMapEntry) => number,
): SourceMapEntry | undefined {
  let best: SourceMapEntry | undefined;
  for (const e of entries) {
    if (start(e) <= offset && offset < end(e)) {
      if (!best || end(e) - start(e) < end(best) - start(best)) best = e;
    }
  }
  return best;
}

/** Returns the entries to highlight when hovering over IR offset `offset`. */
export function entriesAtIrOffset(
  entries: SourceMapEntry[],
  offset: number,
): SourceMapEntry[] {
  const e = innermost(
    entries,
    offset,
    (e) => e.irStart,
    (e) => e.irEnd,
  );
  return e ? [e] : [];
}

/**
 * Returns the entries to highlight when hovering over source offset `offset`:
 * all ops converted from the innermost AST node there.
 */
export function entriesAtSourceOffset(
  entries: SourceMapEntry[],
  offset: number,
): SourceMapEntry[] {
  const e = innermost(
    entries,
    offset,
    (e) => e.srcStart,
    (e) => e.srcEnd,
  );
  if (!e) return [];
  return entries.filter(
    (other) => other.srcStart === e.srcStart && other.srcEnd === e.srcEnd,
  );
}
