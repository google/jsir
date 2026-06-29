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

#include "maldoca/js/ast/ast_util.h"

#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/ast/ast_visitor.h"
#include "maldoca/js/ast/ast_walker.h"
#include "maldoca/js/babel/babel.h"
#include "maldoca/js/babel/babel.pb.h"

namespace maldoca {

absl::StatusOr<BabelAstString> GetAstStringFromSource(
    Babel& babel, std::string_view source, const BabelParseRequest& request,
    absl::Duration timeout) {
  ABSL_ASSIGN_OR_RETURN(auto babel_parse_result,
                        babel.Parse(source, request, timeout));
  if (!babel_parse_result.errors.errors().empty()) {
    std::vector<std::string> error_strings;
    error_strings.reserve(babel_parse_result.errors.errors().size());
    for (const auto& error : babel_parse_result.errors.errors()) {
      error_strings.push_back(absl::StrCat(error));
    }
    return absl::InvalidArgumentError(absl::StrJoin(error_strings, "\n"));
  }
  return babel_parse_result.ast_string;
}

absl::StatusOr<std::tuple<std::unique_ptr<JsFile>, BabelScopes>>
GetFileAstFromSource(Babel& babel, std::string_view source,
                     const BabelParseRequest& request, absl::Duration timeout,
                     std::optional<const int> recursion_depth_limit_op) {
  ABSL_ASSIGN_OR_RETURN(
      BabelAstString babel_ast_string,
      GetAstStringFromSource(babel, source, request, timeout));
  ABSL_ASSIGN_OR_RETURN(
      auto file_ast,
      GetFileAstFromAstString(babel_ast_string, recursion_depth_limit_op));
  return std::tuple{std::move(file_ast), babel_ast_string.scopes()};
}

absl::StatusOr<std::tuple<std::unique_ptr<JsFile>, BabelScopes>>
GetFileAstFromSource(Babel& babel, std::string_view source,
                     absl::Duration timeout,
                     std::optional<const int> recursion_depth_limit_op) {
  return GetFileAstFromSource(babel, source, BabelParseRequest(), timeout,
                              recursion_depth_limit_op);
}

absl::StatusOr<std::unique_ptr<JsFile>> GetFileAstFromAstString(
    const BabelAstString& babel_ast_string,
    std::optional<const int> recursion_depth_limit_op) {
  ABSL_ASSIGN_OR_RETURN(
      auto json_ast,
      GetAstJsonFromAstString(babel_ast_string, recursion_depth_limit_op));
  return JsFile::FromJson(json_ast);
}

void CUnescapeStringLiteral(JsStringLiteral& string_literal) {
  std::string unescaped_string_literal;
  absl::CUnescape(string_literal.value(), &unescaped_string_literal);
  string_literal.set_value(unescaped_string_literal);

  if (string_literal.extra().has_value()) {
    JsStringLiteralExtra* extra = string_literal.extra().value();

    std::string raw_value;
    absl::CUnescape(extra->raw_value(), &raw_value);
    extra->set_raw_value(raw_value);

    std::string raw;
    absl::CUnescape(extra->raw(), &raw);
    extra->set_raw(raw);
  }
}

void CUnescapeStringLiteralsInFile(JsFile& file) {
  class CUnescapeStringLiteralsVisitor : public EmptyMutableJsAstVisitor {
    void VisitStringLiteral(JsStringLiteral& string_literal) override {
      return CUnescapeStringLiteral(string_literal);
    }
  };

  CUnescapeStringLiteralsVisitor visitor;
  MutableJsAstWalker walker{&visitor, /*postorder_callback=*/nullptr};

  walker.VisitFile(file);
}

BabelAstString GetAstStringFromFileAst(const JsFile& file) {
  std::stringstream ss;
  file.Serialize(ss);
  std::string ast_json_str = ss.str();

  BabelAstString result;
  result.set_value(ast_json_str);
  result.set_string_literals_base64_encoded(false);
  return result;
}

absl::StatusOr<std::string> PrettyPrintSourceFromAstString(
    Babel& babel, const BabelAstString& babel_ast_string,
    absl::Duration timeout) {
  ABSL_ASSIGN_OR_RETURN(
      auto result,
      babel.Generate(babel_ast_string, BabelGenerateOptions{}, timeout));
  return result.source_code;
}

absl::StatusOr<std::string> PrettyPrintSourceFromFileAst(
    Babel& babel, const JsFile& file, absl::Duration timeout) {
  BabelAstString ast_string = GetAstStringFromFileAst(file);
  return PrettyPrintSourceFromAstString(babel, ast_string, timeout);
}

absl::StatusOr<std::string> PrettyPrintSourceFromSourceString(
    Babel& babel, std::string_view source_string,
    const BabelParseRequest& request, absl::Duration timeout) {
  ABSL_ASSIGN_OR_RETURN(
      BabelAstString babel_ast_string,
      GetAstStringFromSource(babel, source_string, request, timeout));
  return PrettyPrintSourceFromAstString(babel, babel_ast_string, timeout);
}

absl::StatusOr<std::string> PrettyPrintSourceFromSourceString(
    Babel& babel, absl::string_view source_string, absl::Duration timeout) {
  return PrettyPrintSourceFromSourceString(babel, source_string,
                                           BabelParseRequest(), timeout);
}

}  // namespace maldoca
