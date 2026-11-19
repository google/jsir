/**
 * @license
 * Copyright 2024 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 *
 * @fileoverview Entry point of the Babel bundle for embedded JavaScript engines
 * (V8, QuickJS, browsers).
 *
 * The bundle is evaluated as a classic script right after babel-standalone
 * (`babel.min.js`), which defines the global `Babel`. It then exposes
 * `globalThis.jsirBabel = {parse, generate}`, which the C++ side calls with
 * JSON-serialized options and receives JSON-serialized responses, matching the
 * proto3 JSON format of `BabelParseResponse` / `BabelGenerateResponse` in
 * //third_party/maldoca/js/babel/babel_internal.proto.
 */

import type * as babelTraverse from '@babel/traverse'; // from //third_party/javascript/typings/babel__traverse

import {babelGenerate, BabelGenerateOptions} from './babel_generate';
import type {BabelPackages} from './babel_packages';
import {babelParse, BabelParseOptions, ScopeWithUid} from './babel_parse';
import {base64Decode} from './base64_encode_decode_string_values';

/** The global defined by babel-standalone. */
interface BabelStandalone {
  packages: BabelPackages;
}

function getBabel(): BabelPackages {
  const babel =
      (globalThis as unknown as {Babel?: BabelStandalone}).Babel?.packages;
  if (babel === undefined) {
    throw new Error(
        'Babel is not loaded. Evaluate babel.min.js before this bundle.');
  }
  return babel;
}

/** JSON form of the `BabelError` proto. */
interface BabelErrorJson {
  name: string;
  message: string;
  loc: {line: number, column: number}|null;
}

function maybeGetPosition(error: unknown): {line: number, column: number}|
    null {
  const loc = (error as {loc?: unknown} | null)?.loc;
  if (!(loc instanceof Object)) {
    return null;
  }
  const {line, column} = loc as {line?: unknown, column?: unknown};
  if (typeof line !== 'number' || typeof column !== 'number') {
    return null;
  }
  return {line, column};
}

function unknownToBabelError(error: unknown): BabelErrorJson {
  // NOTE: Don't copy `error` into a local `const`/`let`. When QuickJS runs out
  // of stack, the thrown value can be its internal "uninitialized" marker, and
  // reading that through a lexical binding throws a ReferenceError (TDZ check).
  const name = (error as {name?: unknown} | null | undefined)?.name;
  const message = (error as {message?: unknown} | null | undefined)?.message;
  return {
    name: (typeof name === 'string' && name) || '{error}',
    message: (typeof message === 'string' && message) || '',
    loc: typeof error === 'object' ? maybeGetPosition(error) : null,
  };
}

function bindingKindToJson(kind: string): string {
  switch (kind) {
    case 'var':
      return 'KIND_VAR';
    case 'let':
      return 'KIND_LET';
    case 'const':
      return 'KIND_CONST';
    case 'module':
      return 'KIND_MODULE';
    case 'hoisted':
      return 'KIND_HOISTED';
    case 'param':
      return 'KIND_PARAM';
    case 'local':
      return 'KIND_LOCAL';
    default:
      return 'KIND_UNKNOWN';
  }
}

/** JSON form of the `BabelScopes` proto. */
interface BabelScopesJson {
  scopes: {[uid: number]: unknown};
  bindings: {[uid: number]: unknown};
}

function scopesToJson(
    scopes: ScopeWithUid[], bindingToId: Map<babelTraverse.Binding, number>,
    base64EncodedStringValues: boolean): BabelScopesJson {
  const scopesJson: BabelScopesJson = {scopes: {}, bindings: {}};

  for (const [binding, bindingId] of bindingToId.entries()) {
    let name: string|undefined = binding.identifier?.name;
    if (name && base64EncodedStringValues) {
      name = base64Decode(name);
    }
    scopesJson.bindings[bindingId] = {
      kind: bindingKindToJson(binding.kind),
      name,
      uid: bindingId,
    };
  }

  for (const scope of scopes) {
    if (scope === null) continue;

    const bindingUids: {[name: string]: number} = {};
    for (const [name, binding] of Object.entries(scope.bindings)) {
      const bindingId = bindingToId.get(binding as babelTraverse.Binding);
      if (bindingId !== undefined) {
        bindingUids[name] = bindingId;
      }
    }

    const parent = scope.parent as ScopeWithUid | undefined;
    scopesJson.scopes[scope.uid] = {
      uid: scope.uid,
      parentUid: parent ? parent.uid : undefined,
      bindingUids,
    };
  }

  return scopesJson;
}

/**
 * Parses JavaScript source into an AST.
 *
 * @param sourceCode The JavaScript source.
 * @param optionsSerialized JSON-serialized `BabelParseOptions`.
 * @return A JSON-serialized AST (empty on failure) and a JSON-serialized
 *     `BabelParseResponse`.
 */
function parse(sourceCode: string, optionsSerialized?: string|null):
    {ast: string, response: string} {
  try {
    const options = (optionsSerialized ? JSON.parse(optionsSerialized) : {}) as
        BabelParseOptions;
    const {ast, scopes, bindingToId} =
        babelParse(getBabel(), sourceCode, options);
    const response = {
      errors: [],
      scopes: scopesToJson(
          scopes, bindingToId, options.base64EncodeStringValues ?? false),
    };
    return {ast, response: JSON.stringify(response)};
  } catch (error: unknown) {
    const response = {errors: [unknownToBabelError(error)]};
    return {ast: '', response: JSON.stringify(response)};
  }
}

/**
 * Options of `generate`. Differs from `BabelGenerateOptions` in the name of
 * the comments option, which follows @babel/generator.
 */
interface GenerateOptionsJson {
  comments?: boolean;
  compact?: boolean;
  base64DecodeStringValues?: boolean;
  sourceMaps?: boolean;
}

/**
 * Generates JavaScript code from an AST.
 *
 * @param astString JSON-serialized AST.
 * @param optionsSerialized JSON-serialized `GenerateOptionsJson`.
 * @return The generated code (empty on failure) and a JSON-serialized
 *     `BabelGenerateResponse`.
 */
function generate(astString: string, optionsSerialized?: string|null):
    {source: string, response: string} {
  try {
    const optionsJson =
        (optionsSerialized ? JSON.parse(optionsSerialized) : {}) as
        GenerateOptionsJson;
    const options: BabelGenerateOptions = {
      includeComments: optionsJson.comments,
      compact: optionsJson.compact,
      base64DecodeStringValues: optionsJson.base64DecodeStringValues,
      sourceMaps: optionsJson.sourceMaps,
    };
    const {code, sourceMap} = babelGenerate(getBabel(), astString, options);
    return {source: code, response: JSON.stringify({sourceMap})};
  } catch (error: unknown) {
    const response = {error: unknownToBabelError(error)};
    return {source: '', response: JSON.stringify(response)};
  }
}

/** The API exposed to the host as `globalThis.jsirBabel`. */
export interface JsirBabel {
  parse: typeof parse;
  generate: typeof generate;
}

(globalThis as unknown as {jsirBabel: JsirBabel}).jsirBabel = {
  parse,
  generate,
};
