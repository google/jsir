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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/variant.h"
#include "maldoca/base/status_macros.h"
#include "maldoca/js/ast/ast.generated.h"

namespace maldoca {

absl::Status ForEachTopLevelNode(
    const JsFile& ast,
    absl::FunctionRef<absl::Status(const JsNode&)> callback) {
  for (const auto& directive : *ast.program()->directives()) {
    MALDOCA_RETURN_IF_ERROR(callback(*directive));
  }
  for (const auto& body : *ast.program()->body()) {
    MALDOCA_RETURN_IF_ERROR(callback(*body));
  }
  return absl::OkStatus();
}

template <class T, class Alloc, class Pred>
typename std::vector<T, Alloc>::size_type erase_if_replacement(
    std::vector<T, Alloc>& c, Pred pred) {
  auto it = std::remove_if(c.begin(), c.end(), pred);
  auto num_erased = std::distance(it, c.end());
  c.erase(it, c.end());
  return num_erased;
}

absl::flat_hash_set<size_t> GetPreludeIndicesFromAnnotations(
    const JsFile& ast) {
  bool is_extracting = false;
  size_t index = 0;
  absl::flat_hash_set<size_t> prelude_indices;
  CHECK_OK(ForEachTopLevelNode(ast, [&](const JsNode& node) {
    absl::Cleanup increment_index = [&] { index++; };

    if (!ast.comments().has_value()) {
      return absl::OkStatus();
    }
    const auto& comments = **ast.comments();

    // Match "// exec:begin", start extraction;
    // Match "// exec:end", pause extraction.
    if (node.leading_comment_uids().has_value()) {
      const auto& leading_comment_uids = **node.leading_comment_uids();

      for (int64_t comment_uid : leading_comment_uids) {
        if (!(comment_uid >= 0 && comment_uid < comments.size())) {
          continue;
        }
        const auto& comment = comments[comment_uid];
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
      prelude_indices.insert(index);
    }

    return absl::OkStatus();
  }));

  return prelude_indices;
}

// Appends the source code corresponding to `node` to `prelude`.
//
// If `node` does not have a source range, then `prelude` is not modified.
void AppendPrelude(absl::string_view original_source, const JsNode& node,
                   std::string& prelude) {
  if (!node.start().has_value() || !node.end().has_value()) {
    return;
  }

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
  int64_t start = *node.start();
  int64_t end = *node.end();
  absl::StrAppend(&prelude, original_source.substr(start, end - start), "\n");
}

JsirAnalysisConfig::DynamicConstantPropagation ExtractPreludeByIndices(
    absl::string_view original_source,
    const absl::flat_hash_set<size_t>& indices, JsFile& ast) {
  std::optional<uint64_t> global_scope_uid = ast.program()->scope_uid();

  std::string prelude;
  size_t current_index = 0;
  erase_if_replacement(
      *ast.program()->directives(), [&](const auto& directive) {
        absl::Cleanup increment_index = [&] { current_index++; };

        if (indices.contains(current_index)) {
          AppendPrelude(original_source, *directive, prelude);
          return true;
        }

        return false;
      });
  erase_if_replacement(*ast.program()->body(), [&](const auto& body) {
    absl::Cleanup increment_index = [&] { current_index++; };

    if (indices.contains(current_index)) {
      AppendPrelude(original_source, *body, prelude);
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

JsirAnalysisConfig::DynamicConstantPropagation ExtractPreludeByAnnotations(
    absl::string_view original_source, JsFile& ast) {
  auto prelude_indices = GetPreludeIndicesFromAnnotations(ast);
  return ExtractPreludeByIndices(original_source, prelude_indices, ast);
}

JsirAnalysisConfig::DynamicConstantPropagation
ExtractPreludeByIndicesAndAnnotations(
    absl::string_view original_source,
    const absl::flat_hash_set<size_t>& indices, JsFile& ast) {
  auto prelude_indices = GetPreludeIndicesFromAnnotations(ast);
  prelude_indices.insert(indices.begin(), indices.end());
  return ExtractPreludeByIndices(original_source, prelude_indices, ast);
}

}  // namespace maldoca
