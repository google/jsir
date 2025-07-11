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

#include "maldoca/js/ast/transforms/erase_comments/pass.h"

#include <optional>

#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/ast/ast_visitor.h"
#include "maldoca/js/ast/ast_walker.h"

namespace maldoca {

void EraseCommentsInAst(JsFile& ast) {
  DefaultMutableJsAstVisitorLambdaWrapper pre_visitor([](JsNode& node) {
    node.set_leading_comment_uids(std::nullopt);
    node.set_trailing_comment_uids(std::nullopt);
    node.set_inner_comment_uids(std::nullopt);
  });
  MutableJsAstWalker walker(&pre_visitor, /*postorder_callback=*/nullptr);
  walker.VisitFile(ast);
  ast.set_comments(std::nullopt);
}

}  // namespace maldoca
