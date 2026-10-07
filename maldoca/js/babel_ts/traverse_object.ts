/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Recursively traverses an object, calling the callback on each object.
 */
function traverseObjectInternal(
    node: object|null|undefined, visited: Set<object>,
    callback: (obj: object) => void) {
  if (node === undefined || node === null) {
    return;
  }

  if (typeof(node) !== 'object') {
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
 */
export function traverseObject(
    node: object|null|undefined, callback: (obj: object) => void) {
  const visited = new Set<object>();
  traverseObjectInternal(node, visited, callback);
}
