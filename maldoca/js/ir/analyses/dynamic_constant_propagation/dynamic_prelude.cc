// Copyright 2025 Google LLC
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

#include "maldoca/js/ir/analyses/dynamic_constant_propagation/dynamic_prelude.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "maldoca/base/status_macros.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/babel/babel.h"
#include "maldoca/js/quickjs/quickjs.h"
#include "quickjs/quickjs.h"

namespace maldoca {

absl::StatusOr<DynamicPrelude> DynamicPrelude::Create(
    const JsirAnalysisConfig::DynamicConstantPropagation &config,
    Babel &babel) {
  std::unique_ptr<JSRuntime, QjsRuntimeDeleter> qjs_runtime{JS_NewRuntime()};

  std::unique_ptr<JSContext, QjsContextDeleter> qjs_context{
      JS_NewContext(qjs_runtime.get())};

  MALDOCA_ASSIGN_OR_RETURN(
      BabelParseResult parse_result,
      babel.Parse(config.prelude_source(), BabelParseRequest{},
                  absl::InfiniteDuration()));

  BabelGenerateOptions generate_options;
  generate_options.set_compact(true);
  MALDOCA_ASSIGN_OR_RETURN(
      BabelGenerateResult generate_result,
      babel.Generate(parse_result.ast_string, generate_options,
                     absl::InfiniteDuration()));

  QjsValue qjs_eval_result{
      qjs_context.get(),
      JS_Eval(qjs_context.get(), generate_result.source_code.data(),
              generate_result.source_code.size(), "prelude.js",
              JS_EVAL_TYPE_GLOBAL),
  };
  if (JS_IsException(qjs_eval_result.get())) {
    return absl::InternalError("Failed to evaluate prelude.");
  }

  QjsValue qjs_global{
      qjs_context.get(),
      JS_GetGlobalObject(qjs_context.get()),
  };

  absl::flat_hash_set<std::string> prelude_symbols;
  {
    // Note: Very important to open the scope here, because we need to free the
    // property enums before the runtime and context are std::move()d.

    JSPropertyEnum *property_enums = nullptr;
    uint32_t num_properties = 0;
    absl::Cleanup property_enums_cleanup = [&] {
      if (property_enums == nullptr) {
        return;
      }

      for (uint32_t i = 0; i != num_properties; ++i) {
        JSPropertyEnum *property_enum = &property_enums[i];
        JS_FreeAtom(qjs_context.get(), property_enum->atom);
      }

      js_free(qjs_context.get(), property_enums);
    };

    int flags = JS_GPN_STRING_MASK | JS_GPN_SET_ENUM;
    if (JS_GetOwnPropertyNames(qjs_context.get(), &property_enums,
                               &num_properties, qjs_global.get(), flags) != 0) {
      return absl::InternalError("Failed to get property names.");
    }

    for (uint32_t i = 0; i != num_properties; ++i) {
      JSPropertyEnum *property_enum = &property_enums[i];

      if (!property_enum->is_enumerable) {
        // Skip builtin global symbols like `Object`, `Function`, `Error` ...
        continue;
      }

      const char *property_name =
          JS_AtomToCString(qjs_context.get(), property_enum->atom);
      if (property_name == nullptr) {
        continue;
      }

      prelude_symbols.insert(property_name);
    }
  }

  std::optional<int64_t> extracted_from_scope_uid;
  if (config.has_extracted_from_scope_uid()) {
    extracted_from_scope_uid = config.extracted_from_scope_uid();
  }

  return DynamicPrelude(std::move(qjs_runtime), std::move(qjs_context),
                        std::move(prelude_symbols), extracted_from_scope_uid);
}

std::optional<QjsValue> DynamicPrelude::GetFunction(absl::string_view name) {
  if (!prelude_symbols_.contains(name)) {
    return std::nullopt;
  }

  QjsValue qjs_global{
      qjs_context_.get(),
      JS_GetGlobalObject(qjs_context_.get()),
  };

  std::string name_str{name};
  QjsValue property{
      qjs_context_.get(),
      JS_GetPropertyStr(qjs_context_.get(), qjs_global.get(), name_str.c_str()),
  };

  if (!JS_IsFunction(qjs_context_.get(), property.get())) {
    return std::nullopt;
  }

  return property;
}

std::optional<QjsValue> DynamicPrelude::GetFunction(JsSymbolId symbol_id) {
  if (symbol_id.def_scope_uid() != extracted_from_scope_uid_) {
    return std::nullopt;
  }

  return GetFunction(symbol_id.name());
}

}  // namespace maldoca
