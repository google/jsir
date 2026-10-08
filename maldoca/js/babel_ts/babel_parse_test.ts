/**
 * @license
 * Copyright 2024 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @fileoverview Tests babel_parse.
 */

import {BABEL_IMPLEMENTATIONS} from './babel_implementations_for_test';
import * as babelParse from './babel_parse';

const sourceWithRecoverableError = `
let a = {
  __proto__: x,
  __proto__: y
}
`;

const expectedErrorName = 'SyntaxError';

const expectedErrorMessage = `Redefinition of __proto__ property. (4:2)`;

for (const {name, babel} of BABEL_IMPLEMENTATIONS) {
  describe(`babelParse (${name})`, () => {
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
        babelParse.parse(
            babel, sourceWithRecoverableError, {errorRecovery: true});
      }).not.toThrow();
    });

    it('parses broken JavaScript', () => {
      const source = '-_-';
      expect(() => babelParse.parse(babel, source, {})).toThrow();
    });

    it('parses class g test', () => {
      const source = `
        const g = 20;

        function baz() {
            {
                class g {}
                console.log(g);
            }

            return g;
        }
      `;
      const result = babelParse.parse(babel, source, {computeScopes: true});
      const ast = JSON.parse(result.ast);

      interface AstNode {
        type?: string;
        id?: {name?: string};
        start?: number;
        [key: string]: unknown;
      }

      // Find the ClassDeclaration for 'g'
      const findClassG = (node: unknown): unknown => {
        if (!node || typeof node !== 'object') return null;
        const obj = node as AstNode;
        if (obj.type === 'ClassDeclaration' && obj.id && obj.id.name === 'g') {
          return obj;
        }
        for (const key of Object.keys(obj)) {
          const found = findClassG(obj[key]);
          if (found) return found;
        }
        return null;
      };

      const classG = findClassG(ast);
      if (classG) {
        const classGObj = classG as AstNode;
        console.log('Class g node:', JSON.stringify(classGObj, null, 2));

        // Confirm that the two entries come from two different bindings
        console.log('Bindings for "g":');
        const gBindings: unknown[] = [];
        (result.scopes as unknown[]).forEach((scopeRaw: unknown) => {
          const scope = scopeRaw as {uid: number, path: {type: string}, bindings: Record<string, unknown>};
          for (const [name, binding] of Object.entries(scope.bindings)) {
            const b = binding as {path?: {node?: {type?: string, start?: number}}};
            if (name === 'g' && b.path && b.path.node) {
              if (b.path.node.type === 'ClassDeclaration' && b.path.node.start === classGObj.start) {
                console.log(`- Found binding for "g" in scope ${scope.uid} (type: ${scope.path.type})`);
                console.log(`  Binding path node type: ${b.path.node.type}, start: ${b.path.node.start}`);
                gBindings.push(b);
              }
            }
          }
        });

        if (gBindings.length >= 2) {
          console.log('Are the two bindings the same object?', gBindings[0] === gBindings[1]);
        } else {
          console.log(`Found only ${gBindings.length} bindings for "g" pointing to the class.`);
        }
      } else {
        console.log('Class g node not found');
      }
    });

    it('tolerates errors', () => {
      const source = `
        let a = {
          __proto__: x,
          __proto__: y
        }
      
        let a = 2;
      `;
      expect(() => babelParse.parse(babel, source, {})).toThrow();
      const ast = babelParse.parse(babel, source, {errorRecovery: true});
      expect(ast).toBeTruthy();
      console.log(ast);
    });
  });
}
