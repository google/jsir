/**
 * @license
 * Copyright 2024 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @fileoverview A wrapper module for babel - generates JavaScript code from an
 * AST.
 */

import type * as babelGenerator from '@babel/generator'; // from //third_party/javascript/typings/babel__generator
import type * as t from '@babel/types'; // from //third_party/javascript/node_modules/babel_types:typings

import type {BabelPackages} from './babel_packages';
import {base64DecodeStringValues} from './base64_encode_decode_string_values';
import {convertCommentUidsToComments} from './comment_uid';
import {traverseObject} from './traverse_object';

/** @babel/generator options, plus the JSIR-specific ones. */
export interface GenerateOptions extends babelGenerator.GeneratorOptions {
  base64DecodeStringValues?: boolean;
}

/**
 * Restores stringified non-finite `NumericLiteral.value` and
 * `NumericLiteral.extra.rawValue` fields back to `number` values after optional
 * UTF-16LE base64 string decoding. Other `"Infinity"` strings (e.g. a
 * `StringLiteral` value) are left untouched.
 */
function reviveNonFiniteNumbers(ast: object) {
  // tslint:disable-next-line:no-any
  traverseObject(ast, (node: any) => {
    if (node.type !== 'NumericLiteral') {
      return;
    }
    if (typeof node.value === 'string') {
      node.value = Number(node.value);
    }
    if (node.extra && typeof node.extra.rawValue === 'string') {
      node.extra.rawValue = Number(node.extra.rawValue);
    }
  });
}

/**
 * Generates JavaScript code from an AST.
 */
export function generate(
    babel: BabelPackages, ast: t.Node,
    options: GenerateOptions): {code: string, map?: string} {
  if (options && options.base64DecodeStringValues) {
    base64DecodeStringValues(ast);
  }
  reviveNonFiniteNumbers(ast);

  convertCommentUidsToComments(babel.types, ast);

  if (options.sourceMaps) {
    options.sourceFileName = 'source.js';
  }

  const {code, map} = babel.generator.default(ast, options);
  if (map) {
    return {code, map: JSON.stringify(map)};
  }
  return {code};
}
