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

#include "maldoca/js/ir/transforms/normalize_object_properties/pass.h"

#include <optional>

#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "absl/strings/str_cat.h"
#include "maldoca/js/ir/ir.h"
#include "maldoca/js/ir/trivia.h"

namespace maldoca {

std::optional<JsirStringLiteralAttr> MaybeNormalizeLiteralKey(
    mlir::OpBuilder &builder, std::optional<mlir::Attribute> literal_key) {
  if (!literal_key.has_value()) {
    return std::nullopt;
  }

  auto identifier_key = llvm::dyn_cast<JsirIdentifierAttr>(*literal_key);
  if (identifier_key == nullptr) {
    return std::nullopt;
  }

  // Babel's `extra` of the new string literal.
  mlir::DictionaryAttr extra = builder.getDictionaryAttr({
      builder.getNamedAttr(
          "raw", builder.getStringAttr(
                     absl::StrCat("\"", identifier_key.getName().str(), "\""))),
      builder.getNamedAttr("rawValue", identifier_key.getName()),
  });

  auto string_literal_key = builder.getAttr<JsirStringLiteralAttr>(
      // loc:
      GetJsirTriviaAttrWithExtra(builder.getContext(), identifier_key.getLoc(),
                                 extra),
      // value:
      identifier_key.getName());

  return string_literal_key;
}

void NormalizeObjectProperties(mlir::Operation *root) {
  mlir::MLIRContext *context = root->getContext();
  mlir::OpBuilder builder{context};

  root->walk([&](JsirObjectPropertyOp op) {
    auto string_literal_key =
        MaybeNormalizeLiteralKey(builder, op.getLiteralKey());
    if (string_literal_key.has_value()) {
      op.setLiteralKeyAttr(*string_literal_key);
      op.setShorthand(false);
    }
  });

  root->walk([&](JsirObjectPropertyRefOp op) {
    auto string_literal_key =
        MaybeNormalizeLiteralKey(builder, op.getLiteralKey());
    if (string_literal_key.has_value()) {
      op.setLiteralKeyAttr(*string_literal_key);
      op.setShorthand(false);
    }
  });
}

}  // namespace maldoca
