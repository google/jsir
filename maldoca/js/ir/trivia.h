// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MALDOCA_JS_IR_CONVERSION_JSIR_TO_AST_UTILS_H_
#define MALDOCA_JS_IR_CONVERSION_JSIR_TO_AST_UTILS_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {

std::unique_ptr<JsPosition> JsirPositionAttr2JsPosition(JsirPositionAttr attr);

JsirLocationAttr GetJsirLocationAttr(mlir::MLIRContext *context,
                                     const JsSourceLocation *loc,
                                     std::optional<int64_t> start_index,
                                     std::optional<int64_t> end_index,
                                     std::optional<int64_t> scope_uid);

// Contains metadata in each AST node.
// This is equivalent to `JsirTriviaAttr` in JSIR.
struct JsTrivia {
  std::optional<std::unique_ptr<JsSourceLocation>> loc;
  std::optional<int64_t> start;
  std::optional<int64_t> end;
  std::optional<int64_t> scope_uid;
  std::optional<std::unique_ptr<JsSymbolId>> referenced_symbol;
  std::optional<std::vector<std::unique_ptr<JsSymbolId>>> defined_symbols;
  std::optional<std::vector<int64_t>> leading_comment_uids;
  std::optional<std::vector<int64_t>> trailing_comment_uids;
  std::optional<std::vector<int64_t>> inner_comment_uids;
};

JsTrivia JsirTriviaAttr2JsTrivia(JsirTriviaAttr attr);

JsirTriviaAttr GetJsirTriviaAttr(mlir::MLIRContext *context,
                                 const JsNode &node);

JsTrivia GetJsTrivia(mlir::Operation *op);

JsTrivia GetJsTrivia(mlir::Attribute attr);

template <typename NodeT,
          typename = std::enable_if_t<std::is_base_of_v<JsNode, NodeT>>,
          typename... Args>
std::unique_ptr<NodeT> CreateJsNodeWithTrivia(JsTrivia trivia, Args &&...args) {
  return absl::make_unique<NodeT>(
      /*loc=*/std::move(trivia.loc),
      /*start=*/trivia.start,
      /*end=*/trivia.end,
      /*leading_comments=*/std::move(trivia.leading_comment_uids),
      /*trailing_comments=*/std::move(trivia.trailing_comment_uids),
      /*inner_comments=*/std::move(trivia.inner_comment_uids),
      /*scope_uid=*/trivia.scope_uid,
      /*referenced_symbol=*/std::move(trivia.referenced_symbol),
      /*defined_symbols=*/std::move(trivia.defined_symbols),
      std::forward<Args>(args)...);
}

template <typename NodeT,
          typename = std::enable_if_t<std::is_base_of_v<JsNode, NodeT>>,
          typename... Args>
std::unique_ptr<NodeT> CreateJsNodeWithTrivia(JsirTriviaAttr trivia_attr,
                                              Args &&...args) {
  JsTrivia trivia;
  if (trivia_attr != nullptr) {
    trivia = JsirTriviaAttr2JsTrivia(trivia_attr);
  }

  return CreateJsNodeWithTrivia<NodeT>(std::move(trivia),
                                       std::forward<Args>(args)...);
}

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_CONVERSION_JSIR_TO_AST_UTILS_H_
