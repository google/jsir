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

#include "maldoca/js/quickjs_babel/quickjs_babel.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "nlohmann/json.hpp"
#include "maldoca/base/status_macros.h"
#include "maldoca/js/babel/babel.h"
#include "maldoca/js/babel/babel_internal.h"
#include "maldoca/js/babel/babel_internal.pb.h"
#include "maldoca/js/quickjs/quickjs.h"
#include "maldoca/js/quickjs_babel/babel_standalone_cc_embed_data.cc.inc"
#include "maldoca/js/quickjs_babel/native_cc_embed_data.cc.inc"
#include "google/protobuf/json/json.h"
#include "quickjs/quickjs-libc.h"
#include "quickjs/quickjs.h"

namespace maldoca {

QuickJsBabel::QuickJsBabel()
    : qjs_runtime_(JS_NewRuntime()),
      qjs_context_(JS_NewContext(qjs_runtime_.get())),
      parse_{qjs_context_.get(), JS_NULL},
      generate_{qjs_context_.get(), JS_NULL} {
  JS_SetMaxStackSize(qjs_runtime_.get(), 4 * 1024 * 1024);
  js_std_init_handlers(qjs_runtime_.get());

  // This ensures that console.log(...) gets printed to stdout.
  {
    int argc = 0;
    char* argv[] = {nullptr};
    js_std_add_helpers(qjs_context_.get(), argc, argv);
  }

  {
    std::string babel_standalone{kBabelStandalone, sizeof(kBabelStandalone)};

    QjsValue ignored{
        qjs_context_.get(),
        JS_Eval(qjs_context_.get(), babel_standalone.data(),
                babel_standalone.size(), "babel.js", JS_EVAL_TYPE_GLOBAL),
    };

    CHECK(!JS_IsException(ignored.get()));
  }

  {
    std::string native{kNative, sizeof(kNative)};

    QjsValue ignored{
        qjs_context_.get(),
        JS_Eval(qjs_context_.get(), native.data(), native.size(), "native.js",
                JS_EVAL_TYPE_GLOBAL),
    };

    CHECK(!JS_IsException(ignored.get()));
  }

  constexpr absl::string_view kParse = "exports.parse";
  constexpr absl::string_view kGenerate = "exports.generate";

  parse_ = QjsValue{
      qjs_context_.get(),
      JS_Eval(qjs_context_.get(), kParse.data(), kParse.size(), "parse.js",
              JS_EVAL_TYPE_GLOBAL),
  };
  CHECK(JS_IsFunction(qjs_context_.get(), parse_.get()));

  generate_ = QjsValue{
      qjs_context_.get(),
      JS_Eval(qjs_context_.get(), kGenerate.data(), kGenerate.size(),
              "generate.js", JS_EVAL_TYPE_GLOBAL),
  };
  CHECK(JS_IsFunction(qjs_context_.get(), generate_.get()));
}

QuickJsBabel::~QuickJsBabel() { js_std_free_handlers(qjs_runtime_.get()); }

absl::StatusOr<BabelParseResult> QuickJsBabel::Parse(
    absl::string_view source_code, const BabelParseRequest& request,
    absl::Duration timeout) {
  nlohmann::json options = BabelParseOptionsToJson(request);

  std::string options_string = options.dump(/*indent=*/2);

  QjsValue qjs_options_string = QjsValue{
      qjs_context_.get(),
      JS_NewStringLen(qjs_context_.get(), options_string.data(),
                      options_string.size()),
  };

  QjsValue qjs_source_code{
      qjs_context_.get(),
      JS_NewStringLen(qjs_context_.get(), source_code.data(),
                      source_code.size()),
  };

  std::vector<JSValue> args = {qjs_source_code.get(), qjs_options_string.get()};
  QjsValue result{
      qjs_context_.get(),
      JS_Call(qjs_context_.get(), parse_.get(),
              /*this_obj=*/JS_NULL, args.size(), args.data()),
  };

  if (!JS_IsObject(result.get())) {
    return absl::InternalError("Result is not an object.");
  }

  QjsValue qjs_ast_string{
      qjs_context_.get(),
      JS_GetPropertyStr(qjs_context_.get(), result.get(), "ast"),
  };

  QjsValue qjs_response{
      qjs_context_.get(),
      JS_GetPropertyStr(qjs_context_.get(), result.get(), "response"),
  };

  std::optional<std::string> ast_json_string = qjs_ast_string.ToString();
  if (!ast_json_string.has_value()) {
    return absl::InternalError("Failed to get ast string.");
  }

  std::optional<std::string> response_string = qjs_response.ToString();
  if (!response_string.has_value()) {
    return absl::InternalError("Failed to get response string.");
  }

  BabelParseResponse response;
  MALDOCA_RETURN_IF_ERROR(
      google::protobuf::json::JsonStringToMessage(*response_string, &response));

  BabelAstString ast_string;
  ast_string.set_value(std::move(*ast_json_string));
  ast_string.set_string_literals_base64_encoded(
      request.base64_encode_string_literals());
  *ast_string.mutable_scopes() = std::move(*response.mutable_scopes());

  BabelErrors errors;
  *errors.mutable_errors() = std::move(*response.mutable_errors());

  return BabelParseResult{
      .ast_string = std::move(ast_string),
      .errors = std::move(errors),
  };
}

// Not implemented. The version of @babel/standalone in //third_party doesn't
// provide @babel/generator APIs.
absl::StatusOr<BabelGenerateResult> QuickJsBabel::Generate(
    const BabelAstString& ast_string, const BabelGenerateOptions& opts,
    absl::Duration timeout) {
  nlohmann::json options_json = BabelGenerateOptionsToJson(
      opts, ast_string.string_literals_base64_encoded());

  std::string options_string = options_json.dump(/*indent=*/2);

  QjsValue qjs_options_string = QjsValue{
      qjs_context_.get(),
      JS_NewStringLen(qjs_context_.get(), options_string.data(),
                      options_string.size()),
  };

  QjsValue qjs_ast_string = QjsValue{
      qjs_context_.get(),
      JS_NewStringLen(qjs_context_.get(), ast_string.value().data(),
                      ast_string.value().size()),
  };

  std::vector<JSValue> args = {qjs_ast_string.get(), qjs_options_string.get()};
  QjsValue result{
      qjs_context_.get(),
      JS_Call(qjs_context_.get(), generate_.get(),
              /*this_obj=*/JS_NULL, args.size(), args.data()),
  };

  if (!JS_IsObject(result.get())) {
    return absl::InternalError("Result is not an object.");
  }

  QjsValue qjs_source{
      qjs_context_.get(),
      JS_GetPropertyStr(qjs_context_.get(), result.get(), "source"),
  };

  QjsValue qjs_response{
      qjs_context_.get(),
      JS_GetPropertyStr(qjs_context_.get(), result.get(), "response"),
  };

  std::optional<std::string> source_code = qjs_source.ToString();
  if (!source_code.has_value()) {
    return absl::InternalError("Failed to get source string.");
  }

  std::optional<std::string> response_string = qjs_response.ToString();
  if (!response_string.has_value()) {
    return absl::InternalError("Failed to get response string.");
  }

  BabelGenerateResponse response;
  MALDOCA_RETURN_IF_ERROR(
      google::protobuf::json::JsonStringToMessage(*response_string, &response));

  std::optional<BabelError> error;
  if (response.has_error()) {
    error = response.error();
  }

  std::optional<std::string> source_map;
  if (response.has_source_map()) {
    source_map = response.source_map();
  }

  return BabelGenerateResult{
      .source_code = std::move(*source_code),
      .error = error,
      .source_map = source_map,
  };
}

}  // namespace maldoca
