/**
 * @license
 * Copyright 2024 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @fileoverview Utils for base64-encode/decode string values.
 *
 * This is pure JavaScript (no Node.js `Buffer`) so that it also runs in
 * embedded engines such as V8 and QuickJS.
 */

import {traverseObject} from './traverse_object';

const BASE64_CHARS =
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';

const BASE64_LOOKUP = (() => {
  const lookup = new Uint8Array(256);
  for (let i = 0; i < BASE64_CHARS.length; i++) {
    lookup[BASE64_CHARS.charCodeAt(i)] = i;
  }
  return lookup;
})();

// NOTE: There are two encodings:
// +---------+--------------------+---------------------------+
// | node.js |      'base64'      |        'base64url'        |
// |---------+--------------------+---------------------------+
// |   C++   | absl::Base64Escape | absl::WebSafeBase64Escape |
// +---------+--------------------+---------------------------+
// We use 'base64' (absl::Base64Escape) on the UTF-16LE bytes of the string.
// LINT.IfChange

/**
 * Base64-encodes a string using UTF-16LE encoding. Equivalent to Node.js
 * `Buffer.from(value, 'utf16le').toString('base64')`.
 */
export function base64Encode(value: string): string {
  const bytes = new Uint8Array(value.length * 2);
  for (let i = 0; i < value.length; i++) {
    const code = value.charCodeAt(i);
    bytes[i * 2] = code & 0xff;
    bytes[i * 2 + 1] = (code >> 8) & 0xff;
  }

  const res: string[] = [];
  let i = 0;
  for (; i + 2 < bytes.length; i += 3) {
    const triplet = (bytes[i] << 16) | (bytes[i + 1] << 8) | bytes[i + 2];
    res.push(
        BASE64_CHARS[(triplet >> 18) & 0b111111] +
        BASE64_CHARS[(triplet >> 12) & 0b111111] +
        BASE64_CHARS[(triplet >> 6) & 0b111111] +
        BASE64_CHARS[triplet & 0b111111]);
  }
  if (i + 1 < bytes.length) {
    const triplet = (bytes[i] << 16) | (bytes[i + 1] << 8);
    res.push(
        BASE64_CHARS[(triplet >> 18) & 0b111111] +
        BASE64_CHARS[(triplet >> 12) & 0b111111] +
        BASE64_CHARS[(triplet >> 6) & 0b111111] + '=');
  } else if (i < bytes.length) {
    const triplet = bytes[i] << 16;
    res.push(
        BASE64_CHARS[(triplet >> 18) & 0b111111] +
        BASE64_CHARS[(triplet >> 12) & 0b111111] + '==');
  }
  return res.join('');
}

/**
 * Base64-decodes a string from UTF-16LE. Equivalent to Node.js
 * `Buffer.from(value, 'base64').toString('utf16le')`.
 */
export function base64Decode(value: string): string {
  const bytes: number[] = [];
  let i = 0;
  while (i < value.length) {
    if (value[i] === '=') break;
    const b0 = BASE64_LOOKUP[value.charCodeAt(i++)];
    if (i >= value.length || value[i] === '=') break;
    const b1 = BASE64_LOOKUP[value.charCodeAt(i++)];
    if (i >= value.length || value[i] === '=') {
      const triplet = (b0 << 18) | (b1 << 12);
      bytes.push((triplet >> 16) & 0xff);
      break;
    }
    const b2 = BASE64_LOOKUP[value.charCodeAt(i++)];
    if (i >= value.length || value[i] === '=') {
      const triplet = (b0 << 18) | (b1 << 12) | (b2 << 6);
      bytes.push((triplet >> 16) & 0xff, (triplet >> 8) & 0xff);
      break;
    }
    const b3 = BASE64_LOOKUP[value.charCodeAt(i++)];
    const triplet = (b0 << 18) | (b1 << 12) | (b2 << 6) | b3;
    bytes.push((triplet >> 16) & 0xff, (triplet >> 8) & 0xff, triplet & 0xff);
  }

  const res: string[] = [];
  for (let j = 0; j + 1 < bytes.length; j += 2) {
    res.push(String.fromCharCode(bytes[j] | (bytes[j + 1] << 8)));
  }
  return res.join('');
}
// LINT.ThenChange(//depot/google3/third_party/maldoca/js/ast/ast_util.cc)

/**
 * Applies `mutate` to all string values in the AST.
 */
// tslint:disable-next-line:no-any
export function mutateStrings(node: any, mutate: (value: string) => string) {
  // tslint:disable-next-line:no-any
  traverseObject(node, (n: any) => {
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
 */
// tslint:disable-next-line:no-any
export function base64EncodeStringValues(node: any) {
  mutateStrings(node, base64Encode);
}

/**
 * Base64-decode all string values in the AST.
 */
// tslint:disable-next-line:no-any
export function base64DecodeStringValues(node: any) {
  mutateStrings(node, base64Decode);
}
