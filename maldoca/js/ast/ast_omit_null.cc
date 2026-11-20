// Copyright 2026 Google LLC
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

// Handwritten ShouldOmitNull<FieldName>() functions of the fields with
// `missing_means_null` in ast_def.textproto. Serialize() calls them when the
// field is null, and leaves the key out if they return true. They tell when
// @babel/parser leaves the key undefined, so that the JSON that Serialize()
// writes matches the JSON of the parser.

#include <memory>
#include <variant>

#include "maldoca/js/ast/ast.generated.h"

namespace maldoca {

// @babel/parser `parseExport` sets `declaration` only in
// `maybeParseExportNamedSpecifiers` (to null, when a `{...}` list follows) and
// in `maybeParseExportDeclaration`. For `export * as ns from "m"`
// (`ExportNamespaceSpecifier`) and `export x from "m"`
// (`ExportDefaultSpecifier`), it sets neither, so `declaration` stays
// undefined and JSON.stringify drops the key (verified in Babel 7.20.5,
// 7.25.3 and 7.29.11). That is the case whenever `specifiers` is non-empty and
// contains no `ExportSpecifier`.
//
// Known limit: only Babel 7.20.5 also accepts the non-standard
// `export * as ns, {} from "m"`, and writes `declaration: null` there. This
// rule leaves the key out for it. That changes only the JSON, not the printed
// code.
bool JsExportNamedDeclaration::ShouldOmitNullDeclaration() const {
  if (specifiers_.empty()) {
    return false;
  }
  for (const auto& specifier : specifiers_) {
    if (std::holds_alternative<std::unique_ptr<JsExportSpecifier>>(
            specifier)) {
      return false;
    }
  }
  return true;
}

}  // namespace maldoca
