/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @fileoverview The subset of Babel that JSIR uses.
 *
 * The Babel glue in this directory runs in different JavaScript runtimes:
 *
 * - Node.js, where Babel comes from the `@babel/*` npm packages.
 * - Embedded engines (V8, QuickJS, browsers), where Babel comes from
 *   babel-standalone (`babel.min.js`), which exposes `Babel.packages.*` as a
 *   global.
 *
 * To keep the glue code identical in both cases, it never imports the Babel
 * runtime directly. Instead, the entry point of each runtime passes in a
 * `BabelPackages`. Only *types* are imported from `@babel/*` here, so these
 * imports are erased at compile time and the bundler never needs to resolve
 * the npm packages.
 */

import type * as babelGenerator from '@babel/generator'; // from //third_party/javascript/typings/babel__generator
import type * as babelParser from '@babel/parser'; // from //third_party/javascript/node_modules/babel_parser:typings
import type * as babelTraverse from '@babel/traverse'; // from //third_party/javascript/typings/babel__traverse
import type * as babelTypes from '@babel/types'; // from //third_party/javascript/node_modules/babel_types:typings

/**
 * The Babel packages that JSIR depends on. Matches both the shape of the npm
 * modules (`import * as parser from '@babel/parser'`) and of babel-standalone
 * (`Babel.packages.parser`).
 */
export interface BabelPackages {
  parser: Pick<typeof babelParser, 'parse'>;
  traverse: Pick<typeof babelTraverse, 'default'>;
  generator: Pick<typeof babelGenerator, 'default'>;
  types: Pick<typeof babelTypes, 'isNode'>;
}
