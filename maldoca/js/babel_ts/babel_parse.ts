/**
 * @license
 * Copyright 2024 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 *
 * @fileoverview The JavaScript parser that uses babel.
 */

import type * as babelParser from '@babel/parser'; // from //third_party/javascript/node_modules/babel_parser:typings
import type * as babelTraverse from '@babel/traverse'; // from //third_party/javascript/typings/babel__traverse

import type {BabelPackages} from './babel_packages';
import {base64EncodeStringValues, mutateStrings} from './base64_encode_decode_string_values';
import {convertCommentsToCommentUids} from './comment_uid';

/**
 * Parser options.
 */
export interface BabelParseOptions {
  // By default, Babel always throws an error when it finds some invalid code.
  // When this option is set to true, it will store the parsing error and
  // try to continue parsing the invalid input file.
  errorRecovery?: boolean;

  // Replaces characteres in the range [U+D800, U+DFFF] with '�' (U+FFFD).
  replaceInvalidSurrogatePairs?: boolean;

  // Base64-encode all string values.
  base64EncodeStringValues?: boolean;

  // The mode in which source code should be parsed. Can be one of "script",
  // "module", or "unambiguous". Defaults to "script". "unambiguous" will
  // make @babel/parser attempt to guess, based on the presence of ES6 import or
  // export statements. Files with ES6 imports and exports are considered
  // "module" and are otherwise "script".
  sourceType?: 'module'|'script'|'unambiguous'|undefined;

  // Should the parser work in strict mode (i.e. throw more errors).
  // According to Babel's comment, if strictMode is undefined, then it depends
  // on whether sourceType is 'module'. However, the source code has a bug such
  // that even if strictMode is true, the parser still depends on sourceType.
  //
  // +-------------------------------+-------------------------+------+-------+
  // |          strictMode           |        undefined        | true | false |
  // +--------------+----------------+-------------------------+------+-------+
  // | Is parser in |    Comment     | sourceType === 'module' | true | false |
  // |              +----------------+-------------------------+------+-------+
  // | strict mode? | Implementation |    sourceType === 'module'     | false |
  // +--------------+----------------+--------------------------------+-------+
  strictMode?: boolean;

  // Whether to add scope information in the AST.
  // If true:
  // - A separate Scopes proto will be returned
  // - Each AST node will have an additional scopeUid field, specifying which
  //   scope it belongs to
  computeScopes?: boolean;
}

/**
 * Replaces characteres in the range [U+D800, U+DFFF] with '�' (U+FFFD).
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

/** A Babel scope, with the (untyped) `uid` field that Babel assigns. */
export type ScopeWithUid = babelTraverse.Scope&{uid: number};

/** The result of `babelParse`. */
export interface BabelParseResult {
  // The AST, as stringified JSON.
  ast: string;

  // All the scopes in the AST. Empty unless `computeScopes` is set.
  scopes: ScopeWithUid[];

  // Maps each binding to its UID. Empty unless `computeScopes` is set.
  bindingToId: Map<babelTraverse.Binding, number>;
}

/**
 * Parses a piece of JavaScript source, and returns the AST as a stringified
 * JSON.
 */
export function babelParse(
    babel: BabelPackages, source: string,
    options: BabelParseOptions): BabelParseResult {
  // https://babeljs.io/docs/en/babel-parser#options
  const babelOptions: babelParser.ParserOptions = {
    // Create "ParenthesizedExpression" nodes.
    // Otherwise @babel/parser would add an extra.parenthesized field to
    // Expression nodes, which is less type-safe.
    createParenthesizedExpressions: true,

    errorRecovery: options.errorRecovery,
    sourceType: options.sourceType,
    strictMode: options.strictMode,
  };

  const ast = babel.parser.parse(source, babelOptions);

  if (options.replaceInvalidSurrogatePairs) {
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
  if (options.computeScopes &&
      !('errors' in ast && (ast.errors as unknown[]).length > 0)) {
    babel.traverse.default(ast, {
      enter(path: babelTraverse.NodePath) {
        const scope = path.scope;
        if ('uid' in scope && typeof scope.uid === 'number') {
          scopes[scope.uid] = scope as ScopeWithUid;
          // tslint:disable-next-line:no-any
          (path.node as any).scopeUid = scope.uid;
        }
      }
    });

    let nextBindingId = 0;
    // The same binding can be registered in multiple scopes. Only record its
    // definition once.
    const processedBindings = new Set<babelTraverse.Binding>();

    for (const scope of Object.values(scopes)) {
      for (const [name, b] of Object.entries(scope.bindings)) {
        const binding = b as babelTraverse.Binding;

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
          const defNode = binding.path.node as any;
          if (defNode.definedSymbols === undefined) {
            defNode.definedSymbols = [];
          }
          defNode.definedSymbols.push({
            name,
            bindingUid: bindingId,
          });
        }
      }
    }
  }

  if (options.base64EncodeStringValues) {
    base64EncodeStringValues(ast);
  }

  // We don't serialize to JSON even though it's possible. The reason is that
  // the AST of TSCompiler (the other choice) cannot be directly serialized due
  // to the existence of parent pointers. Therefore, it would not be a fair
  // comparison if we serialize here for Babel.
  return {ast: JSON.stringify(ast), scopes: Object.values(scopes), bindingToId};
}
