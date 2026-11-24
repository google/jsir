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

#include "maldoca/js/ir/trivia.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "nlohmann/json.hpp"
#include "maldoca/astgen/mlir_json_utils.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {
namespace {

JsirPositionAttr JsPosition2JsirPositionAttr(mlir::MLIRContext* context,
                                             const JsPosition& pos) {
  return JsirPositionAttr::get(context, pos.line(), pos.column());
}

JsirSymbolIdAttr GetJsirSymbolIdAttr(mlir::MLIRContext* context,
                                     const JsSymbolId& symbol_id) {
  return JsirSymbolIdAttr::get(context,
                               mlir::StringAttr::get(context, symbol_id.name()),
                               symbol_id.binding_uid());
}

}  // namespace

JsirTriviaAttr GetJsirTriviaAttr(mlir::Attribute attr) {
  return llvm::TypeSwitch<mlir::Attribute, JsirTriviaAttr>(attr)
      .Case([&](JsirStringLiteralAttr attr) { return attr.getLoc(); })
      .Case([&](JsirNumericLiteralAttr attr) { return attr.getLoc(); })
      .Case([&](JsirBigIntLiteralAttr attr) { return attr.getLoc(); })
      .Case([&](JsirIdentifierAttr attr) { return attr.getLoc(); })
      .Case([&](JsirPrivateNameAttr attr) { return attr.getLoc(); })
      .Case([&](JsirImportSpecifierAttr attr) { return attr.getLoc(); })
      .Case([&](JsirImportDefaultSpecifierAttr attr) { return attr.getLoc(); })
      .Case(
          [&](JsirImportNamespaceSpecifierAttr attr) { return attr.getLoc(); })
      .Case([&](JsirImportAttributeAttr attr) { return attr.getLoc(); })
      .Case([&](JsirInterpreterDirectiveAttr attr) { return attr.getLoc(); })
      .Default([&](mlir::Attribute attr) {
        LOG(INFO) << "Unexpected mlir::Attribute to get source location from. "
                  << "Maybe we missed a type cast here!";
        return nullptr;
      });
}

std::unique_ptr<JsPosition> JsirPositionAttr2JsPosition(JsirPositionAttr attr) {
  const auto loc_start_line = attr.getLine();
  const auto loc_start_column = attr.getColumn();
  return absl::make_unique<JsPosition>(loc_start_line, loc_start_column);
}

JsirLocationAttr GetJsirLocationAttr(mlir::MLIRContext* context,
                                     const JsSourceLocation* loc,
                                     std::optional<int64_t> start_index,
                                     std::optional<int64_t> end_index,
                                     std::optional<int64_t> scope_uid) {
  JsirPositionAttr start = JsPosition2JsirPositionAttr(context, *loc->start());
  JsirPositionAttr end = JsPosition2JsirPositionAttr(context, *loc->end());
  const auto identifier_name = loc->identifier_name();
  mlir::StringAttr jsir_identifier_name = nullptr;
  if (identifier_name.has_value()) {
    jsir_identifier_name = mlir::StringAttr::get(context, *identifier_name);
  }

  return JsirLocationAttr::get(context, start, end, jsir_identifier_name,
                               start_index, end_index, scope_uid);
}

JsirTriviaAttr GetJsirTriviaAttr(mlir::MLIRContext* context,
                                 const JsNode& node) {
  JsirLocationAttr jsir_location = nullptr;
  if (node.loc().has_value()) {
    jsir_location =
        GetJsirLocationAttr(context, node.loc().value(), node.start(),
                            node.end(), node.scope_uid());
  }

  llvm::ArrayRef<int64_t> leading_comment_uids;
  if (node.leading_comment_uids().has_value()) {
    leading_comment_uids = **node.leading_comment_uids();
  }

  llvm::ArrayRef<int64_t> trailing_comment_uids;
  if (node.trailing_comment_uids().has_value()) {
    trailing_comment_uids = **node.trailing_comment_uids();
  }

  llvm::ArrayRef<int64_t> inner_comment_uids;
  if (node.inner_comment_uids().has_value()) {
    inner_comment_uids = **node.inner_comment_uids();
  }

  JsirSymbolIdAttr mlir_referenced_symbol = nullptr;
  if (node.referenced_symbol().has_value()) {
    mlir_referenced_symbol =
        GetJsirSymbolIdAttr(context, **node.referenced_symbol());
  }

  std::vector<JsirSymbolIdAttr> mlir_defined_symbols;
  if (node.defined_symbols().has_value()) {
    for (const auto& defined_symbol : **node.defined_symbols()) {
      mlir_defined_symbols.push_back(
          GetJsirSymbolIdAttr(context, *defined_symbol));
    }
  }

  mlir::DictionaryAttr mlir_extra = nullptr;
  if (node.extra().has_value()) {
    mlir_extra = JsonToMlirDictionaryAttr(context, **node.extra());
  }

  return JsirTriviaAttr::get(context, jsir_location, leading_comment_uids,
                             trailing_comment_uids, inner_comment_uids,
                             mlir_referenced_symbol, mlir_defined_symbols,
                             mlir_extra);
}

mlir::DictionaryAttr GetJsirLiteralExtraAttr(mlir::MLIRContext* context,
                                             mlir::StringAttr raw,
                                             mlir::Attribute raw_value) {
  mlir::Builder builder(context);
  return builder.getDictionaryAttr({
      builder.getNamedAttr("raw", raw),
      builder.getNamedAttr("rawValue", raw_value),
  });
}

JsirTriviaAttr GetJsirTriviaAttrWithExtra(mlir::MLIRContext* context,
                                          JsirTriviaAttr trivia,
                                          mlir::DictionaryAttr extra) {
  if (trivia == nullptr) {
    return JsirTriviaAttr::get(context, /*loc=*/nullptr,
                               /*leading_comment_uids=*/{},
                               /*trailing_comment_uids=*/{},
                               /*inner_comment_uids=*/{},
                               /*referenced_symbol=*/nullptr,
                               /*defined_symbols=*/{}, extra);
  }
  return JsirTriviaAttr::get(
      context, trivia.getLoc(), trivia.getLeadingCommentUids(),
      trivia.getTrailingCommentUids(), trivia.getInnerCommentUids(),
      trivia.getReferencedSymbol(), trivia.getDefinedSymbols(), extra);
}

JsTrivia GetJsTrivia(mlir::Operation* op) {
  const auto mlir_trivia = llvm::dyn_cast<JsirTriviaAttr>(op->getLoc());
  if (mlir_trivia == nullptr) {
    return JsTrivia{};
  }
  return JsirTriviaAttr2JsTrivia(mlir_trivia);
}

JsTrivia JsirTriviaAttr2JsTrivia(JsirTriviaAttr attr) {
  // A node without a source location can still have trivia, e.g. an `extra`.
  std::optional<std::unique_ptr<JsSourceLocation>> loc;
  std::optional<int64_t> start_index;
  std::optional<int64_t> end_index;
  std::optional<int64_t> scope_uid;
  if (attr.getLoc() != nullptr) {
    start_index = attr.getLoc().getStartIndex();
    end_index = attr.getLoc().getEndIndex();
    scope_uid = attr.getLoc().getScopeUid();

    std::optional<std::unique_ptr<JsPosition>> start;
    if (JsirPositionAttr mlir_start = attr.getLoc().getStart();
        mlir_start != nullptr) {
      start = JsirPositionAttr2JsPosition(mlir_start);
    }

    std::optional<std::unique_ptr<JsPosition>> end;
    if (JsirPositionAttr mlir_end = attr.getLoc().getEnd();
        mlir_end != nullptr) {
      end = JsirPositionAttr2JsPosition(mlir_end);
    }

    std::optional<std::string> identifier_name;
    if (mlir::StringAttr mlir_identifier_name =
            attr.getLoc().getIdentifierName();
        mlir_identifier_name != nullptr) {
      identifier_name = mlir_identifier_name.str();
    }

    if (start.has_value() && end.has_value()) {
      loc = absl::make_unique<JsSourceLocation>(
          /*start=*/std::move(*start),
          /*end=*/std::move(*end),
          /*identifier_name=*/std::move(identifier_name));
    }
  }

  std::optional<std::vector<int64_t>> leading_comment_uids;
  if (llvm::ArrayRef<int64_t> mlir_leading_comment_uids =
          attr.getLeadingCommentUids();
      !mlir_leading_comment_uids.empty()) {
    leading_comment_uids = mlir_leading_comment_uids;
  }

  std::optional<std::vector<int64_t>> trailing_comment_uids;
  if (llvm::ArrayRef<int64_t> mlir_trailing_comment_uids =
          attr.getTrailingCommentUids();
      !mlir_trailing_comment_uids.empty()) {
    trailing_comment_uids = mlir_trailing_comment_uids;
  }

  std::optional<std::vector<int64_t>> inner_comment_uids;
  if (llvm::ArrayRef<int64_t> mlir_inner_comment_uids =
          attr.getInnerCommentUids();
      !mlir_inner_comment_uids.empty()) {
    inner_comment_uids = mlir_inner_comment_uids;
  }

  std::optional<std::unique_ptr<JsSymbolId>> referenced_symbol;
  if (JsirSymbolIdAttr mlir_referenced_symbol = attr.getReferencedSymbol()) {
    referenced_symbol =
        std::make_unique<JsSymbolId>(mlir_referenced_symbol.getName().str(),
                                     mlir_referenced_symbol.getBindingUid());
  }

  std::optional<std::vector<std::unique_ptr<JsSymbolId>>> defined_symbols;
  if (llvm::ArrayRef<JsirSymbolIdAttr> mlir_defined_symbols =
          attr.getDefinedSymbols();
      !mlir_defined_symbols.empty()) {
    std::vector<std::unique_ptr<JsSymbolId>> defined_symbols_vec;
    for (JsirSymbolIdAttr mlir_defined_symbol : mlir_defined_symbols) {
      defined_symbols_vec.push_back(std::make_unique<JsSymbolId>(
          mlir_defined_symbol.getName().str(),
          mlir_defined_symbol.getBindingUid()));
    }
    defined_symbols = std::move(defined_symbols_vec);
  }

  std::optional<nlohmann::json> extra;
  if (mlir::DictionaryAttr mlir_extra = attr.getExtra();
      mlir_extra != nullptr) {
    absl::StatusOr<nlohmann::json> json_extra = MlirAttrToJson(mlir_extra);
    if (json_extra.ok()) {
      extra = *std::move(json_extra);
    } else {
      LOG(ERROR) << "Invalid `extra` in JsirTriviaAttr: "
                 << json_extra.status();
    }
  }

  return JsTrivia{
      .loc = std::move(loc),
      .start = start_index,
      .end = end_index,
      .scope_uid = scope_uid,
      .referenced_symbol = std::move(referenced_symbol),
      .defined_symbols = std::move(defined_symbols),
      .leading_comment_uids = std::move(leading_comment_uids),
      .trailing_comment_uids = std::move(trailing_comment_uids),
      .inner_comment_uids = std::move(inner_comment_uids),
      .extra = std::move(extra),
  };
}

JsTrivia GetJsTrivia(mlir::Attribute attr) {
  JsirTriviaAttr mlir_trivia = GetJsirTriviaAttr(attr);
  if (mlir_trivia == nullptr) {
    return JsTrivia{};
  }

  return JsirTriviaAttr2JsTrivia(mlir_trivia);
}

}  // namespace maldoca
