/**
 * @license
 * Copyright 2024 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @fileoverview The top-level file to run in V8, QuickJS and browsers.
 *
 * esbuild bundles this file with babel-standalone into a classic script, which
 * exposes `globalThis.jsirBabel = {parse, generate}`.
 */

import * as Babel from '@babel/standalone'; // from //third_party/javascript/babel_standalone:typings
import type * as t from '@babel/types'; // from //third_party/javascript/node_modules/babel_types:typings

import * as babelGenerate from './babel_generate';
import type {BabelPackages} from './babel_packages';
import * as babelParse from './babel_parse';
import {base64Decode} from './base64_encode_decode_string_values';

const babel: BabelPackages = Babel.packages;

// tslint:disable-next-line:no-any
function maybeGetPosition(error: any): {line: number, column: number}|null {
  if (!(error?.loc instanceof Object)) {
    return null;
  }

  if (typeof error?.loc?.line !== 'number' ||
      typeof error?.loc?.column !== 'number') {
    return null;
  }

  return {line: error.loc.line, column: error.loc.column};
}

// tslint:disable-next-line:no-any
function unknownToBabelError(error: any): {
  name: string,
  message: string,
  loc: {line: number, column: number}|null,
} {
  return {
    name: error?.name || '{error}',
    message: error?.message || '',
    loc: typeof error === 'object' ? maybeGetPosition(error) : null,
  };
}

/**
 * Parses JavaScript source into an AST.
 * @return A JSON-serialized AST and a JSON-serialized `BabelParseResponse`.
 */
function parse(sourceCode: string, optionsSerialized?: string|null):
    {ast: string, response: string} {
  let options: babelParse.ParseOptions|undefined = undefined;
  if (optionsSerialized) {
    options = JSON.parse(optionsSerialized) as babelParse.ParseOptions;
  }

  try {
    const {ast, scopes, bindingToId} =
        babelParse.parse(babel, sourceCode, options);

    const scopesPb: {
      scopes: {[uid: number]: unknown},
      bindings: {[uid: number]: unknown},
    } = {
      scopes: {},
      bindings: {},
    };

    if (bindingToId) {
      for (const [binding, bindingId] of bindingToId.entries()) {
        const bindingKindPb = (() => {
          switch (binding.kind as string) {
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
        })();

        let name = binding.identifier ? binding.identifier.name : undefined;
        if (name && options?.base64EncodeStringValues) {
          name = base64Decode(name);
        }
        const bindingPb = {
          kind: bindingKindPb,
          name,
          uid: bindingId,
        };

        scopesPb.bindings[bindingId] = bindingPb;
      }
    }

    for (const scope of scopes) {
      if (scope === null) continue;
      const uid = scope.uid;

      const scopePb: {
        uid: number,
        parentUid?: number,
        bindingUids: {[name: string]: number},
      } = {
        uid,
        bindingUids: {},
      };

      if (scope.parent) {
        scopePb.parentUid = (scope.parent as babelParse.ScopeWithUid).uid;
      }

      for (const [name, binding] of Object.entries(scope.bindings)) {
        if (bindingToId) {
          const bindingId = bindingToId.get(binding);
          if (bindingId !== undefined) {
            scopePb.bindingUids[name] = bindingId;
          }
        }
      }

      scopesPb.scopes[uid] = scopePb;
    }

    const response = {
      errors: [],
      scopes: scopesPb,
    };

    return {
      ast,
      response: JSON.stringify(response),
    };

  } catch (error) {
    const response: {errors?: unknown[]} = {};

    const babelError = unknownToBabelError(error);
    if (babelError) {
      response.errors = [babelError];
    }

    return {ast: '', response: JSON.stringify(response)};
  }
}

/**
 * Generates JavaScript code from an AST.
 * @return The generated code and a JSON-serialized `BabelGenerateResponse`.
 */
function generate(astString: string, optionsSerialized?: string|null):
    {source: string, response: string} {
  try {
    const ast = JSON.parse(astString) as t.Node;

    let options: babelGenerate.GenerateOptions = {};
    if (optionsSerialized) {
      options = JSON.parse(optionsSerialized) as babelGenerate.GenerateOptions;
    }

    const result = babelGenerate.generate(babel, ast, options);

    const response: {source_map?: string} = {};
    if (result.map) {
      response.source_map = result.map;
    }
    return {source: result.code, response: JSON.stringify(response)};

  } catch (error) {
    const response: {error?: unknown} = {};

    const errorPb = unknownToBabelError(error);
    if (errorPb) {
      response.error = errorPb;
    }

    return {source: '', response: JSON.stringify(response)};
  }
}

/** The API exposed to the host as `globalThis.jsirBabel`. */
export interface JsirBabel {
  parse: typeof parse;
  generate: typeof generate;
}

// Required so that we can use "jsirBabel.parse" from the host.
(globalThis as unknown as {jsirBabel: JsirBabel}).jsirBabel = {
  parse,
  generate,
};
