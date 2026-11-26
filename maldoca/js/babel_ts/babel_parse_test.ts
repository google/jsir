/**
 * @license
 * Copyright 2024 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @fileoverview Tests babel_parse.
 */

import * as BabelStandalone from '@babel/standalone'; // from //third_party/javascript/babel_standalone:typings

import * as babelParse from './babel_parse';

const babel = BabelStandalone.packages;

const sourceWithRecoverableError = `
let a = {
  __proto__: x,
  __proto__: y
}
`;

const expectedErrorName = 'SyntaxError';

const expectedErrorMessage = `Redefinition of __proto__ property. (4:2)`;

describe('babelParse', () => {
  it('testParseSimpleSource', () => {
    const {ast} = babelParse.parse(
        babel, 'console.log(\'Hello, world!\');', {errorRecovery: false});
    expect(!!ast).toBeTrue();
  });

  it('testParseVarDef', () => {
    const {ast, scopes} = babelParse.parse(
        babel, 'var a = 1;', {errorRecovery: false, computeScopes: true});
    expect(!!ast).toBeTrue();
    expect(scopes.length).toEqual(1);
  });

  it('testErrorRecovery', () => {
    expect(() => {
      babelParse.parse(
          babel, sourceWithRecoverableError, {errorRecovery: false});
    })
        .toThrowMatching(
            (error) => error.message === expectedErrorMessage &&
                error.name === expectedErrorName);

    expect(() => {
      babelParse.parse(babel, sourceWithRecoverableError, {errorRecovery: true});
    }).not.toThrow();
  });
});
