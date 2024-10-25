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

#ifndef MALDOCA_JS_AST_TRANSFORMS_ERASE_COMMENTS_PASS_H_
#define MALDOCA_JS_AST_TRANSFORMS_ERASE_COMMENTS_PASS_H_

#include "maldoca/js/ast/ast.generated.h"

namespace maldoca {

// Erases the comments in `ast`. Specifically, all leading, trailing and inner
// comments in each `JsNode` in `ast`.
void EraseCommentsInAst(JsFile& ast);

}  // namespace maldoca

#endif  // MALDOCA_JS_AST_TRANSFORMS_ERASE_COMMENTS_PASS_H_
