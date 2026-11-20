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
#include <utility>

#include "nlohmann/json.hpp"
#include "maldoca/js/babel/babel.pb.h"

namespace maldoca {
namespace {

// Returns the @babel/parser `plugins` entry for `plugin`, or std::nullopt for
// PARSER_PLUGIN_UNSPECIFIED.
std::optional<nlohmann::json> ParserPluginToJson(
    BabelParseRequest::ParserPlugin plugin) {
  switch (plugin) {
    case BabelParseRequest::PARSER_PLUGIN_UNSPECIFIED:
      return std::nullopt;
    case BabelParseRequest::PARSER_PLUGIN_BIG_INT:
      return "bigInt";
    case BabelParseRequest::PARSER_PLUGIN_CLASS_PROPERTIES:
      return "classProperties";
    case BabelParseRequest::PARSER_PLUGIN_CLASS_PRIVATE_PROPERTIES:
      return "classPrivateProperties";
    case BabelParseRequest::PARSER_PLUGIN_CLASS_PRIVATE_METHODS:
      return "classPrivateMethods";
    case BabelParseRequest::PARSER_PLUGIN_CLASS_STATIC_BLOCK:
      return "classStaticBlock";
    case BabelParseRequest::PARSER_PLUGIN_DYNAMIC_IMPORT:
      return "dynamicImport";
    case BabelParseRequest::PARSER_PLUGIN_EXPORT_NAMESPACE_FROM:
      return "exportNamespaceFrom";
    case BabelParseRequest::PARSER_PLUGIN_IMPORT_META:
      return "importMeta";
    case BabelParseRequest::PARSER_PLUGIN_LOGICAL_ASSIGNMENT:
      return "logicalAssignment";
    case BabelParseRequest::PARSER_PLUGIN_NULLISH_COALESCING_OPERATOR:
      return "nullishCoalescingOperator";
    case BabelParseRequest::PARSER_PLUGIN_NUMERIC_SEPARATOR:
      return "numericSeparator";
    case BabelParseRequest::PARSER_PLUGIN_OBJECT_REST_SPREAD:
      return "objectRestSpread";
    case BabelParseRequest::PARSER_PLUGIN_OPTIONAL_CATCH_BINDING:
      return "optionalCatchBinding";
    case BabelParseRequest::PARSER_PLUGIN_OPTIONAL_CHAINING:
      return "optionalChaining";
    case BabelParseRequest::PARSER_PLUGIN_PRIVATE_IN:
      return "privateIn";
    case BabelParseRequest::PARSER_PLUGIN_TOP_LEVEL_AWAIT:
      return "topLevelAwait";
    case BabelParseRequest::PARSER_PLUGIN_EXPORT_DEFAULT_FROM:
      return "exportDefaultFrom";
  }
  return std::nullopt;
}

}  // namespace

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
      {"base64EncodeStringValues", request.base64_encode_string_values()},
      {"replaceInvalidSurrogatePairs",
       request.replace_invalid_surrogate_pairs()},
      {"computeScopes", request.compute_scopes()},
  };

  // @babel/parser options. Babel's default for each of these is false. An
  // option is passed if the request sets it, or if its proto default is true
  // (`create_parenthesized_expressions`), so that Babel uses the proto default.
  auto set_if_present = [&](bool has_option, bool value, const char* name) {
    if (has_option || value) {
      options[name] = value;
    }
  };
  set_if_present(request.has_create_parenthesized_expressions(),
                 request.create_parenthesized_expressions(),
                 "createParenthesizedExpressions");
  set_if_present(request.has_error_recovery(), request.error_recovery(),
                 "errorRecovery");
  set_if_present(request.has_allow_return_outside_function(),
                 request.allow_return_outside_function(),
                 "allowReturnOutsideFunction");
  set_if_present(request.has_allow_await_outside_function(),
                 request.allow_await_outside_function(),
                 "allowAwaitOutsideFunction");
  set_if_present(request.has_allow_import_export_everywhere(),
                 request.allow_import_export_everywhere(),
                 "allowImportExportEverywhere");
  set_if_present(request.has_allow_super_outside_method(),
                 request.allow_super_outside_method(),
                 "allowSuperOutsideMethod");
  set_if_present(request.has_allow_undeclared_exports(),
                 request.allow_undeclared_exports(), "allowUndeclaredExports");

  if (request.plugins_size() > 0) {
    nlohmann::json plugins = nlohmann::json::array();
    for (int plugin : request.plugins()) {
      std::optional<nlohmann::json> plugin_json = ParserPluginToJson(
          static_cast<BabelParseRequest::ParserPlugin>(plugin));
      if (plugin_json.has_value()) {
        plugins.push_back(*std::move(plugin_json));
      }
    }
    options["plugins"] = std::move(plugins);
  }

  if (source_type.has_value()) {
    options["sourceType"] = *source_type;
  }
  if (strict_mode.has_value()) {
    options["strictMode"] = *strict_mode;
  }

  return options;
}

nlohmann::json BabelGenerateOptionsToJson(const BabelGenerateOptions &options,
                                          bool string_values_base64_encoded) {
  nlohmann::json json{
      {"comments", options.include_comments()},
      {"compact", options.compact()},
      {"base64DecodeStringValues", string_values_base64_encoded},
      {"sourceMaps", options.source_maps()},
  };

  return json;
}

}  // namespace maldoca
