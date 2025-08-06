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

#ifndef MALDOCA_JS_IR_ANALYSES_DYNAMIC_CONSTANT_PROPAGATION_DYNAMIC_PRELUDE_H_
#define MALDOCA_JS_IR_ANALYSES_DYNAMIC_CONSTANT_PROPAGATION_DYNAMIC_PRELUDE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/babel/babel.h"
#include "maldoca/js/driver/driver.pb.h"
#include "maldoca/js/quickjs/quickjs.h"
#include "quickjs/quickjs.h"

namespace maldoca {

// Stores the code that we treat as "builtin". The code stored here is executed
// during dynamic constant propagation.
class DynamicPrelude {
 public:
  static absl::StatusOr<DynamicPrelude> Create(
      const JsirAnalysisConfig::DynamicConstantPropagation &config,
      Babel &babel);

  JSContext *GetQjsContext() { return qjs_context_.get(); }

  // Returns a handle to the global function with the given name. Returns
  // std::nullopt if the function is not found.
  std::optional<QjsValue> GetFunction(absl::string_view name);

  std::optional<QjsValue> GetFunction(JsSymbolId symbol_id);

  // Returns the uid of the scope where the prelude functions are extracted
  // from.
  std::optional<int64_t> GetExtractedFromScopeUid() const {
    return extracted_from_scope_uid_;
  }

  absl::flat_hash_set<std::string> GetPreludeSymbols() const {
    return prelude_symbols_;
  }

 private:
  explicit DynamicPrelude(
      std::unique_ptr<JSRuntime, QjsRuntimeDeleter> qjs_runtime,
      std::unique_ptr<JSContext, QjsContextDeleter> qjs_context,
      absl::flat_hash_set<std::string> prelude_symbols,
      std::optional<int64_t> extracted_from_scope_uid)
      : qjs_runtime_(std::move(qjs_runtime)),
        qjs_context_(std::move(qjs_context)),
        prelude_symbols_(std::move(prelude_symbols)),
        extracted_from_scope_uid_(extracted_from_scope_uid) {}

  std::unique_ptr<JSRuntime, QjsRuntimeDeleter> qjs_runtime_;
  std::unique_ptr<JSContext, QjsContextDeleter> qjs_context_;
  absl::flat_hash_set<std::string> prelude_symbols_;
  std::optional<int64_t> extracted_from_scope_uid_;
};

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_ANALYSES_DYNAMIC_CONSTANT_PROPAGATION_DYNAMIC_PRELUDE_H_
