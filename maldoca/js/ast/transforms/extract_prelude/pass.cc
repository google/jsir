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

#include "maldoca/js/ast/transforms/extract_prelude/pass.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/variant.h"
#include "maldoca/js/ast/ast.generated.h"

namespace maldoca {

template <class T, class Alloc, class Pred>
typename std::vector<T, Alloc>::size_type erase_if_replacement(
    std::vector<T, Alloc> &c, Pred pred) {
  auto it = std::remove_if(c.begin(), c.end(), pred);
  auto num_erased = std::distance(it, c.end());
  c.erase(it, c.end());
  return num_erased;
}

JsirAnalysisConfig::DynamicConstantPropagation ExtractPrelude(
    absl::string_view original_source, JsFile &ast) {
  std::optional<uint64_t> global_scope_uid = ast.program()->scope_uid();

  std::string prelude;
  bool is_extracting = false;

  erase_if_replacement(*ast.program()->body(), [&](const auto &body) {
    JsNode *node = nullptr;
    switch (body.index()) {
      case 0: {
        node = &*absl::get<0>(body);
        break;
      }
      case 1: {
        node = &*absl::get<1>(body);
        break;
      }
      default:
        LOG(FATAL) << "Unreachable code.";
    }

    // Match "// exec:begin", start extraction;
    // Match "// exec:end", pause extraction.
    if (ast.comments().has_value() &&
        node->leading_comment_uids().has_value()) {
      const auto &comments = **ast.comments();
      for (int64_t comment_uid : **node->leading_comment_uids()) {
        if (!(comment_uid >= 0 && comment_uid < comments.size())) {
          continue;
        }
        const auto &comment = comments[comment_uid];
        std::string comment_text{comment->value()};
        comment_text = absl::StripAsciiWhitespace(comment_text);
        comment_text = absl::AsciiStrToLower(comment_text);
        if (comment_text == "exec:begin") {
          is_extracting = true;
        } else if (comment_text == "exec:end") {
          is_extracting = false;
        }
      }
    }

    if (is_extracting) {
      if (node->start().has_value() && node->end().has_value()) {
        int64_t start = *node->start();
        int64_t end = *node->end();
        absl::StrAppend(&prelude, original_source.substr(start, end - start));

        // The source range in the AST does not contain '\n'.
        //
        // For example:
        //
        // ```
        // let a = 1;            <-- There's a '\n' here.
        // function foo() {
        //   console.log("foo");
        // }                     <-- There's a '\n' here.
        // let b = 2;
        // function bar() {
        //   console.log("bar");
        // }
        // ```
        //
        // In the code above, the source range of the JsFunctionDeclaration node
        // for `foo` does not cover either the '\n' before or after the function
        // declaration (marked above).
        //
        // As a result, if we extract the AST nodes for `foo` and `bar` and
        // concatenate them, we will get the following code:
        //
        // ```
        // function foo() {
        //   console.log("foo");
        // }function bar() {
        //   console.log("bar");
        // }
        // ```
        //
        // To prevent this, we need to manually add a '\n' after every extracted
        // part.
        //
        // This doesn't cause the code to be invalid, because we are only
        // extracting at the level of **top-level statements**.
        absl::StrAppend(&prelude, "\n");
      }
      return true;
    }

    return false;
  });

  JsirAnalysisConfig::DynamicConstantPropagation config;
  config.set_prelude_source(prelude);
  if (global_scope_uid.has_value()) {
    config.set_extracted_from_scope_uid(*global_scope_uid);
  }

  return config;
}

}  // namespace maldoca
