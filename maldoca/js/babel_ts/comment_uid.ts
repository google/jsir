/**
 * @fileoverview Replaces comment objects in AST nodes with indices into the
 * top-level `comments` array, and back.
 *
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type * as t from '@babel/types'; // from //third_party/javascript/node_modules/babel_types:typings

import type {BabelPackages} from './babel_packages';
import {traverseObject} from './traverse_object';

type NodeWithCommentUids = t.Node&{
  leadingCommentUids?: number[];
  innerCommentUids?: number[];
  trailingCommentUids?: number[];
};

/**
 * Turns a Comment[] into a number[] representing the comment UIDs, by looking
 * them up in the commentToUid map.
 */
function commentsToCommentUids(
    comments: t.Comment[], commentToUid: Map<t.Comment, number>): number[] {
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
 */
export function convertCommentsToCommentUids(
    types: BabelPackages['types'], ast: t.File) {
  const commentToUid: Map<t.Comment, number> = new Map();
  if (ast.comments) {
    ast.comments.forEach((comment, index) => {
      commentToUid.set(comment, index);
    });
  }

  traverseObject(ast, obj => {
    if (!types.isNode(obj)) {
      return;
    }

    const node = obj as NodeWithCommentUids;
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
 */
function commentUidsToComments(
    commentUids: number[], commentPool: t.Comment[]): t.Comment[] {
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
 */
export function convertCommentUidsToComments(
    types: BabelPackages['types'], ast: t.File) {
  if (ast.comments) {
    const commentPool = ast.comments;

    traverseObject(ast, obj => {
      if (!types.isNode(obj)) {
        return;
      }

      const node = obj as NodeWithCommentUids;
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
