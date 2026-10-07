/**
 * @license
 * Copyright 2024 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 *
 * @fileoverview Generate JavaScript code from Babel AST.
 */

import type * as babelGenerator from '@babel/generator'; // from //third_party/javascript/typings/babel__generator
import type * as t from '@babel/types'; // from //third_party/javascript/node_modules/babel_types:typings

import type {BabelPackages} from './babel_packages';
import {base64DecodeStringValues} from './base64_encode_decode_string_values';
import {convertCommentUidsToComments} from './comment_uid';

/**
 * Generator options.
 */
export interface BabelGenerateOptions {
  // Whether comments should be kept in the generated source.
  includeComments?: boolean;

  // Whether to base64-decode string values in the AST.
  base64DecodeStringValues?: boolean;

  // Whether to generate compact code.
  compact?: boolean;

  // Whether to generate a source map.
  sourceMaps?: boolean;
}

/**
 * Response from babelGenerate.
 */
export interface BabelGenerateResponse {
  // The generated source code.
  code: string;

  // The generated source map as a JSON string.
  sourceMap?: string;
}

/**
 * Generates a piece of JavaScript source from the AST as stringified JSON.
 */
export function babelGenerate(
    babel: BabelPackages, astString: string,
    options: BabelGenerateOptions): BabelGenerateResponse {
  const ast = JSON.parse(astString) as t.File;

  if (options.base64DecodeStringValues) {
    base64DecodeStringValues(ast);
  }

  convertCommentUidsToComments(babel.types, ast);

  const babelOptions: babelGenerator.GeneratorOptions = {
    comments: options.includeComments,
    compact: options.compact,
    sourceMaps: options.sourceMaps,
    sourceFileName: 'source.js',
  };

  const {code, map} = babel.generator.default(ast, babelOptions);
  return {
    code,
    sourceMap: map ? JSON.stringify(map) : undefined,
  };
}
