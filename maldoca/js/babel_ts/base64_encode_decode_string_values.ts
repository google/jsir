/**
 * @license
 * Copyright 2024 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @fileoverview Utils for base64-encode/decode string values.
 */

import {traverseObject} from './traverse_object';

const kBase64Chars =
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
const kBase64Lookup = new Uint8Array(256);
for (let i = 0; i < kBase64Chars.length; i++) {
  kBase64Lookup[kBase64Chars.charCodeAt(i)] = i;
}

// NOTE: There are two encodings:
// +---------+--------------------+---------------------------+
// | node.js |      'base64'      |        'base64url'        |
// |---------+--------------------+---------------------------+
// |   C++   | absl::Base64Escape | absl::WebSafeBase64Escape |
// +---------+--------------------+---------------------------+
// LINT.IfChange
/**
 * Base64-encodes a string using UTF-16LE encoding.
 */
function base64Encode(value: string): string {
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
 */
function base64Decode(value: string): string {
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
// LINT.ThenChange(//depot/google3/third_party/maldoca/js/ast/ast_util.cc)

/**
 * Base64-encode/decode all string values in the AST.
 */
// tslint:disable-next-line:no-any
function mutateStrings(node: any, mutate: (value: string) => string) {
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
function base64EncodeStringValues(node: any) {
  mutateStrings(node, base64Encode);
}

/**
 * Base64-decode all string values in the AST.
 */
// tslint:disable-next-line:no-any
function base64DecodeStringValues(node: any) {
  mutateStrings(node, base64Decode);
}

export {
  base64Encode,
  base64Decode,
  base64EncodeStringValues,
  base64DecodeStringValues,
  mutateStrings,
};
