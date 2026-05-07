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

#include "maldoca/js/babel/babel_internal.h"

#include <optional>
#include <string>

#include "nlohmann/json.hpp"
#include "maldoca/js/babel/babel.pb.h"

namespace maldoca {

nlohmann::json BabelParseOptionsToJson(const BabelParseRequest &request) {
  auto source_type = [&]() -> std::optional<std::string> {
    switch (request.source_type()) {
      case BabelParseRequest::SOURCE_TYPE_UNSPECIFIED:
        return std::nullopt;
      case BabelParseRequest::SOURCE_TYPE_UNAMBIGUOUS:
        return "unambiguous";
      case BabelParseRequest::SOURCE_TYPE_SCRIPT:
        return "script";
      case BabelParseRequest::SOURCE_TYPE_MODULE:
        return "module";
    }
  }();

  auto strict_mode = [&]() -> std::optional<bool> {
    switch (request.strict_mode()) {
      case BabelParseRequest::STRICT_MODE_UNSPECIFIED:
        return std::nullopt;
      case BabelParseRequest::STRICT_MODE_YES:
        return true;
      case BabelParseRequest::STRICT_MODE_NO:
        return false;
    }
  }();

  nlohmann::json options{
      // Our custom options:
      {"base64EncodeStringLiterals", request.base64_encode_string_literals()},
      {"replaceInvalidSurrogatePairs",
       request.replace_invalid_surrogate_pairs()},
      {"computeScopes", request.compute_scopes()},

      // @babel/parser options:
      {"createParenthesizedExpressions", true},
      {"errorRecovery", request.error_recovery()},
  };

  if (source_type.has_value()) {
    options["sourceType"] = *source_type;
  }
  if (strict_mode.has_value()) {
    options["strictMode"] = *strict_mode;
  }

  return options;
}

nlohmann::json BabelGenerateOptionsToJson(const BabelGenerateOptions &options,
                                          bool string_literals_base64_encoded) {
  nlohmann::json json{
      {"comments", options.include_comments()},
      {"compact", options.compact()},
      {"base64DecodeStringLiterals", string_literals_base64_encoded},
      {"sourceMaps", options.source_maps()},
  };

  return json;
}

}  // namespace maldoca
