/**
 * @license
 * Copyright 2024 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 *
 * @fileoverview The top-level file to run in V8.
 */

exports = {};

// =============================================================================

// NOTE: There are two encodings:
// +---------+--------------------+---------------------------+
// | node.js |      'base64'      |        'base64url'        |
// |---------+--------------------+---------------------------+
// |   C++   | absl::Base64Escape | absl::WebSafeBase64Escape |
// +---------+--------------------+---------------------------+
const kBase64Chars =
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
const kBase64Lookup = new Uint8Array(256);
for (let i = 0; i < kBase64Chars.length; i++) {
  kBase64Lookup[kBase64Chars.charCodeAt(i)] = i;
}

/**
 * Base64-encodes a string using UTF-16LE encoding.
 *
 * @param {string} value
 * @return {string}
 */
function base64Encode(value) {
  const binary = new Uint8Array(value.length * 2);
  for (let i = 0; i < value.length; i++) {
    const code = value.charCodeAt(i);
    binary[i * 2] = code & 0xff;
    binary[i * 2 + 1] = (code >> 8) & 0xff;
  }
  const res = [];
  let i = 0;
  for (; i + 2 < binary.length; i += 3) {
    const triplet = (binary[i] << 16) | (binary[i + 1] << 8) | binary[i + 2];
    res.push(
        kBase64Chars[(triplet >> 18) & 0b111111] +
        kBase64Chars[(triplet >> 12) & 0b111111] +
        kBase64Chars[(triplet >> 6) & 0b111111] +
        kBase64Chars[triplet & 0b111111]);
  }
  if (i < binary.length) {
    if (i + 1 < binary.length) {
      const triplet = (binary[i] << 16) | (binary[i + 1] << 8);
      res.push(
          kBase64Chars[(triplet >> 18) & 0b111111] +
          kBase64Chars[(triplet >> 12) & 0b111111] +
          kBase64Chars[(triplet >> 6) & 0b111111] +
          '=');
    } else {
      const triplet = binary[i] << 16;
      res.push(
          kBase64Chars[(triplet >> 18) & 0b111111] +
          kBase64Chars[(triplet >> 12) & 0b111111] +
          '==');
    }
  }
  return res.join('');
}

/**
 * Base64-decodes a string from UTF-16LE.
 *
 * @param {string} value
 * @return {string}
 */
function base64Decode(value) {
  const buffer = [];
  let i = 0;
  while (i < value.length) {
    if (value[i] === '=') break;
    const b0 = kBase64Lookup[value.charCodeAt(i++)];
    if (i >= value.length || value[i] === '=') break;
    const b1 = kBase64Lookup[value.charCodeAt(i++)];
    if (i >= value.length || value[i] === '=') {
      const triplet = (b0 << 18) | (b1 << 12);
      buffer.push((triplet >> 16) & 0xff);
      break;
    }
    const b2 = kBase64Lookup[value.charCodeAt(i++)];
    if (i >= value.length || value[i] === '=') {
      const triplet = (b0 << 18) | (b1 << 12) | (b2 << 6);
      buffer.push((triplet >> 16) & 0xff);
      buffer.push((triplet >> 8) & 0xff);
      break;
    }
    const b3 = kBase64Lookup[value.charCodeAt(i++)];
    const triplet = (b0 << 18) | (b1 << 12) | (b2 << 6) | b3;
    buffer.push((triplet >> 16) & 0xff);
    buffer.push((triplet >> 8) & 0xff);
    buffer.push(triplet & 0xff);
  }
  const res = [];
  for (let j = 0; j + 1 < buffer.length; j += 2) {
    res.push(String.fromCharCode(buffer[j] | (buffer[j + 1] << 8)));
  }
  return res.join('');
}

/**
 * Recursively traverses an object, calling the callback on each object.
 *
 * @param {Object! | null | undefined} node
 * @param {Set<Object!>!} visited
 * @param {function(Object!): void} callback
 */
function traverseObjectInternal(node, visited, callback) {
  if (node === undefined || node === null) {
    return;
  }

  if (typeof (node) !== 'object') {
    return;
  }

  if (visited.has(node)) {
    return;
  }
  visited.add(node);

  callback(node);

  Object.values(node).forEach(field => {
    if (typeof field === 'object') {
      traverseObjectInternal(field, visited, callback);
    }
  });
}

/**
 * Recursively traverses an object, calling the callback on each object.
 *
 * @param {Object! | null | undefined} node
 * @param {function(Object!): void} callback
 */
function traverseObject(node, callback) {
  const visited = new Set();
  traverseObjectInternal(node, visited, callback);
}

/**
 * Base64-encode/decode all string values in the AST.
 *
 * @param {!Object} node
 * @param {function(string): string} mutate
 */
function mutateStrings(node, mutate) {
  traverseObject(node, (n) => {
    for (const key of Object.keys(n)) {
      const val = n[key];
      if (typeof val === 'string') {
        n[key] = mutate(val);
      }
    }
  });
}

/**
 * Base64-encode all string values in the AST.
 *
 * @param {!Object} node
 */
function base64EncodeStringValues(node) {
  mutateStrings(node, base64Encode);
}

/**
 * Base64-decode all string values in the AST.
 *
 * @param {!Object} node
 */
function base64DecodeStringValues(node) {
  mutateStrings(node, base64Decode);
}

// =============================================================================

/**
 * Turns a Comment[] into a number[] representing the comment UIDs, by looking
 * them up in the commentToUid map.
 *
 * @param {Array<Object!>!} comments
 * @param {Map<Object!, number>!} commentToUid
 * @return {Array<number>!}
 */
function commentsToCommentUids(comments, commentToUid) {
  return comments.flatMap((comment) => {
    const uid = commentToUid.get(comment);
    if (uid !== undefined) {
      return [uid];
    } else {
      return [];
    }
  });
}

/**
 * In each AST node, replaces {leading,trailing,inner}Comments with
 * {leading,trailing,inner}CommentUids.
 *
 * @param {Object!} ast
 */
function convertCommentsToCommentUids(ast) {
  const commentToUid = new Map();
  if (ast.comments) {
    ast.comments.forEach((comment, index) => {
      commentToUid.set(comment, index);
    });
  }

  traverseObject(ast, obj => {
    if (!Babel.packages.types.isNode(obj)) {
      return;
    }

    const node = obj;
    if (node.leadingComments) {
      node.leadingCommentUids =
          commentsToCommentUids(node.leadingComments, commentToUid);
      delete node.leadingComments;
    }
    if (node.innerComments) {
      node.innerCommentUids =
          commentsToCommentUids(node.innerComments, commentToUid);
      delete node.innerComments;
    }
    if (node.trailingComments) {
      node.trailingCommentUids =
          commentsToCommentUids(node.trailingComments, commentToUid);
      delete node.trailingComments;
    }
  });
}

/**
 * Turns a number[] representing the comment UIDs into a Comment[], by looking
 * them up in the commentPool.
 *
 * @param {Array<number>!} commentUids
 * @param {Array<Object!>!} commentPool
 * @return {Array<Object!>!}
 */
function commentUidsToComments(commentUids, commentPool) {
  return commentUids.flatMap((uid) => {
    const comment = commentPool[uid];
    if (comment) {
      return [comment];
    } else {
      return [];
    }
  });
}

/**
 * In each AST node, replaces {leading,trailing,inner}CommentUids with
 * {leading,trailing,inner}Comments.
 *
 * @param {Object!} ast
 */
function convertCommentUidsToComments(ast) {
  if (ast.comments) {
    const commentPool = ast.comments;

    traverseObject(ast, obj => {
      if (!Babel.packages.types.isNode(obj)) {
        return;
      }

      const node = obj;
      if (node.leadingCommentUids) {
        node.leadingComments =
            commentUidsToComments(node.leadingCommentUids, commentPool);
        delete node.leadingCommentUids;
      }
      if (node.innerCommentUids) {
        node.innerComments =
            commentUidsToComments(node.innerCommentUids, commentPool);
        delete node.innerCommentUids;
      }
      if (node.trailingCommentUids) {
        node.trailingComments =
            commentUidsToComments(node.trailingCommentUids, commentPool);
        delete node.trailingCommentUids;
      }
    });
  }
}

// =============================================================================

/**
 * Replaces characters in the range [U+D800, U+DFFF] with '�' (U+FFFD).
 *
 // copybara:strip_begin(internal comment)
 * See: b/235090893.
 // copybara:strip_end
 *
 * @param {string} source
 * @return {string}
 */
function replaceInvalidSurrogatePairs(source) {
  return [...source]
      .map(
          (str) => (str.codePointAt(0) ?? 0) >= 0xD800 &&
                  (str.codePointAt(0) ?? 0) <= 0xDFFF ?
              '\ufffd' :
              str)
      .join('');
}

/**
 * Parses JavaScript source and returns a stringified AST.
 * @param {string} source
 * @param {object?} options
 * @return {{ast: string, scopes: !Array<!Object>}}
 */
function parseInternal(source, options) {
  const ast = Babel.packages.parser.parse(source, options);

  if (options?.replaceInvalidSurrogatePairs) {
    mutateStrings(ast, replaceInvalidSurrogatePairs);
  }

  convertCommentsToCommentUids(ast);

  // Store all scopes in a dictionary, and add a scope UID to each AST node.
  //
  // We don't try to get scope information when there are errors in the AST
  // (this only happens when errorRecovery is true), because (1) scope
  // information would be invalid anyway, and (2) babel-traverse would crash
  // with an exception during scope computation.
  const scopes = {};
  const bindingToId = new Map();
  if (options?.computeScopes && !('errors' in ast && ast.errors.length > 0)) {
    Babel.packages.traverse.default(ast, {
      enter(path) {
        const scope = path.scope;
        if ('uid' in scope && typeof scope.uid === 'number') {
          scopes[scope.uid] = scope;
          path.node.scopeUid = scope.uid;
        }
      }
    });

    let nextBindingId = 0;
    const processedBindings = new Set();

    for (const scope of Object.values(scopes)) {
      for (const [name, binding] of Object.entries(scope.bindings)) {
        let bindingId = bindingToId.get(binding);
        if (bindingId === undefined) {
          bindingId = nextBindingId++;
          bindingToId.set(binding, bindingId);
        }

        for (const referencePath of binding.referencePaths) {
          referencePath.node.referencedSymbol = {
            name: name,
            bindingUid: bindingId,
          };
        }

        if (!processedBindings.has(binding)) {
          processedBindings.add(binding);
          const def_node = binding.path.node;
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

  if (options && options.base64EncodeStringValues) {
    base64EncodeStringValues(ast);
  }

  // We don't serialize to JSON even though it's possible. The reason is that
  // the AST of TSCompiler (the other choice) cannot be directly serialized due
  // to the existence of parent pointers. Therefore, it would not be a fair
  // comparison if we serialize here for Babel.
  return {ast: JSON.stringify(ast), scopes: Object.values(scopes), bindingToId};
}
exports.parseInternal = parseInternal;

// =============================================================================

/**
 * Generates JavaScript code from an AST.
 * @param {object!} ast
 * @param {object!} options
 * @return {string}
 */
function generateInternal(ast, options) {
  if (options && options.base64DecodeStringValues) {
    base64DecodeStringValues(ast);
  }

  convertCommentUidsToComments(ast);

  if (options.sourceMaps) {
    options.sourceFileName = 'source.js';
  }

  const {code, map} = Babel.packages.generator.default(ast, options);
  return {code, map};
}
exports.generateInternal = generateInternal;

/**
 * @param {!Object} error
 * @return {?{line: number, column: number}}
 */
function maybeGetPosition(error) {
  if (!(error?.loc instanceof Object)) {
    return null;
  }

  if (typeof error?.loc?.line != 'number' ||
      typeof error?.loc?.column != 'number') {
    return null;
  }

  return {line: error.loc.line, column: error.loc.column};
}

/**
 * @param {!Object} error
 * @return {{
 *    name: string,
 *    message: string,
 *    loc: ?{line: number, column: number}
 * }}
 */
function unknownToBabelError(error) {
  return {
    name: error?.name || '{error}',
    message: error?.message || '',
    loc: typeof error === 'object' ? maybeGetPosition(error) : null,
  };
}

/**
 * Parses JavaScript source into an AST.
 * @param {string} sourceCode
 * @param {string?} optionsSerialized
 * @return {{ast: string, response: string}} A JSON-serialized AST and a
 *     JSON-serialized `BabelParseResponse`.
 */
exports.parse = function(sourceCode, optionsSerialized) {
  let options = undefined;
  if (optionsSerialized) {
    options = JSON.parse(optionsSerialized);
  }

  try {
    let {ast, scopes, bindingToId} = parseInternal(sourceCode, options);

    const scopesPb = {
      scopes: {},
      bindings: {},
    };

    if (bindingToId) {
      for (const [binding, bindingId] of bindingToId.entries()) {
        const bindingKindPb = (() => {
          switch (binding.kind) {
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
          name: name,
          uid: bindingId,
        };

        scopesPb.bindings[bindingId] = bindingPb;
      }
    }

    for (const scope of scopes) {
      if (scope === null) continue;
      const uid = scope.uid;

      const scopePb = {
        uid: uid,
        bindingUids: {},
      };

      if (scope.parent) {
        scopePb.parentUid = scope.parent.uid;
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
      ast: ast,
      response: JSON.stringify(response),
    };

  } catch (error) {
    const response = {};

    const babelError = unknownToBabelError(error);
    if (babelError) {
      response.errors = [babelError];
    }

    return {ast: '', response: JSON.stringify(response)};
  }
};

/**
 * Generates JavaScript code from an AST.
 * @param {string} astString
 * @param {string?} optionsSerialized
 * @return {{source: string, response: string}} The generated code and a
 *     JSON-serialized `BabelGenerateResponse`.
 */
exports.generate = function(astString, optionsSerialized) {
  try {
    let ast = JSON.parse(astString);

    let options = {};
    if (optionsSerialized) {
      options = JSON.parse(optionsSerialized);
    }

    const {code, map} = generateInternal(ast, options);
    const response = {};
    if (map) {
      response.sourceMap = JSON.stringify(map);
    }
    return {source: code, response: JSON.stringify(response)};

  } catch (error) {
    const response = {};

    const errorPb = unknownToBabelError(error);
    if (errorPb) {
      response.error = errorPb;
    }

    return {source: '', response: JSON.stringify(response)};
  }
};
