/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @fileoverview The Babel implementations that the core logic is tested
 * against: the `@babel/*` npm packages (used by Node.js), and babel-standalone
 * (used by V8, QuickJS and browsers via babel_bundle.js).
 *
 * The open-source build only tests against babel-standalone.
 */

import * as BabelStandalone from '@babel/standalone'; // from //third_party/javascript/babel_standalone:typings

import type {BabelPackages} from './babel_packages';

/** A named Babel implementation. */
export interface BabelImplementation {
  name: string;
  babel: BabelPackages;
}

/** All Babel implementations to run the tests against. */
export const BABEL_IMPLEMENTATIONS: readonly BabelImplementation[] = [
  {name: 'standalone', babel: BabelStandalone.packages},
];
