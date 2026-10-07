/**
 * @license
 * Copyright 2024 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @fileoverview A wrapper module for babel - parses JavaScript and returns AST.
 */

import type * as babelParser from '@babel/parser'; // from //third_party/javascript/node_modules/babel_parser:typings
import type * as babelTraverse from '@babel/traverse'; // from //third_party/javascript/typings/babel__traverse

import type {BabelPackages} from './babel_packages';
import {base64EncodeStringValues, mutateStrings} from './base64_encode_decode_string_values';
import {convertCommentsToCommentUids} from './comment_uid';
import {traverseObject} from './traverse_object';

/** @babel/parser options, plus the JSIR-specific ones. */
export interface ParseOptions extends babelParser.ParserOptions {
  replaceInvalidSurrogatePairs?: boolean;
  computeScopes?: boolean;
  base64EncodeStringValues?: boolean;
}

/** A Babel scope, with the (untyped) `uid` field that Babel assigns. */
export type ScopeWithUid = babelTraverse.Scope&{uid: number};

/**
 * Replaces characteres in the range [U+D800, U+DFFF] with '' (U+FFFD).
 */
function replaceInvalidSurrogatePairs(source: string): string {
  return [...source]
      .map(
          (str) => (str.codePointAt(0) ?? 0) >= 0xD800 &&
                  (str.codePointAt(0) ?? 0) <= 0xDFFF ?
              '\ufffd' :
              str)
      .join('');
}

/**
 * Converts non-finite `NumericLiteral.value` and `NumericLiteral.extra.rawValue`
 * fields to `"Infinity"` / `"-Infinity"` / `"NaN"` strings before optional
 * UTF-16LE base64 string encoding and `JSON.stringify` (which would otherwise
 * emit `null` for non-finite numbers such as `2e308`).
 */
function stringifyNonFiniteNumbers(ast: object) {
  // tslint:disable-next-line:no-any
  traverseObject(ast, (node: any) => {
    if (node.type !== 'NumericLiteral') {
      return;
    }
    if (typeof node.value === 'number' && !Number.isFinite(node.value)) {
      node.value = String(node.value);
    }
    if (node.extra && typeof node.extra.rawValue === 'number' &&
        !Number.isFinite(node.extra.rawValue)) {
      node.extra.rawValue = String(node.extra.rawValue);
    }
  });
}

/**
 * Parses JavaScript source and returns a stringified AST.
 */
export function parse(
    babel: BabelPackages, source: string, options?: ParseOptions): {
  ast: string,
  scopes: ScopeWithUid[],
  bindingToId: Map<babelTraverse.Binding, number>,
} {
  const ast = babel.parser.parse(source, options);

  if (options?.replaceInvalidSurrogatePairs) {
    mutateStrings(ast, replaceInvalidSurrogatePairs);
  }

  convertCommentsToCommentUids(babel.types, ast);

  // Store all scopes in a dictionary, and add a scope UID to each AST node.
  //
  // We don't try to get scope information when there are errors in the AST
  // (this only happens when errorRecovery is true), because (1) scope
  // information would be invalid anyway, and (2) babel-traverse would crash
  // with an exception during scope computation.
  const scopes: {[uid: number]: ScopeWithUid} = {};
  const bindingToId = new Map<babelTraverse.Binding, number>();
  if (options?.computeScopes && !('errors' in ast && ast.errors.length > 0)) {
    babel.traverse.default(ast, {
      enter(path) {
        const scope = path.scope;
        if ('uid' in scope && typeof scope.uid === 'number') {
          scopes[scope.uid] = scope as ScopeWithUid;
          // tslint:disable-next-line:no-any
          (path.node as any).scopeUid = scope.uid;
        }
      }
    });

    let nextBindingId = 0;
    const processedBindings = new Set<babelTraverse.Binding>();

    for (const scope of Object.values(scopes)) {
      for (const [name, binding] of Object.entries(scope.bindings)) {
        let bindingId = bindingToId.get(binding);
        if (bindingId === undefined) {
          bindingId = nextBindingId++;
          bindingToId.set(binding, bindingId);
        }

        for (const referencePath of binding.referencePaths) {
          // tslint:disable-next-line:no-any
          (referencePath.node as any).referencedSymbol = {
            name,
            bindingUid: bindingId,
          };
        }

        if (!processedBindings.has(binding)) {
          processedBindings.add(binding);
          // tslint:disable-next-line:no-any
          const def_node = binding.path.node as any;
          if (def_node.definedSymbols === undefined) {
            def_node.definedSymbols = [];
          }
          def_node.definedSymbols.push({
            name,
            bindingUid: bindingId,
          });
        }
      }
    }
  }

  stringifyNonFiniteNumbers(ast);

  if (options?.base64EncodeStringValues) {
    base64EncodeStringValues(ast);
  }

  // We don't serialize to JSON even though it's possible. The reason is that
  // the AST of TSCompiler (the other choice) cannot be directly serialized due
  // to the existence of parent pointers. Therefore, it would not be a fair
  // comparison if we serialize here for Babel.
  return {
    ast: JSON.stringify(ast),
    scopes: Object.values(scopes),
    bindingToId,
  };
}
