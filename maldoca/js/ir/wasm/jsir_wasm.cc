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

// Emscripten bindings that convert a Babel AST (JSON) into JSIR.
//
// Parsing JavaScript source into a Babel AST happens in the browser (Babel is
// a JavaScript library). This module implements everything after that:
//
//   Babel AST JSON --(JsFile::FromJson)--> JsFile --(AstToJshirFile)--> JSHIR
//
// and prints the resulting MLIR.

#include <emscripten/bind.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Visitors.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "nlohmann/json.hpp"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/ir/conversion/utils.h"
#include "maldoca/js/ir/ir.h"

namespace {

struct JsirResult {
  bool ok;
  std::string output;
  // JSON array of [ir_line, ir_column, source_start, source_end] tuples, one
  // per op that has a source location:
  // - ir_line (1-based) and ir_column (0-based) are where the op starts in
  //   `output`;
  // - source_start and source_end are Babel's offsets (UTF-16 code units) of
  //   the corresponding AST node in the JavaScript source.
  std::string source_map;
};

JsirResult Error(absl::string_view stage, const absl::Status& status) {
  return JsirResult{
      .ok = false,
      .output = absl::StrCat(stage, ": ", status.ToString()),
      .source_map = "[]",
  };
}

// Returns the [start, end) source offsets recorded in the op's location.
std::optional<std::pair<int64_t, int64_t>> GetSourceRange(mlir::Operation* op) {
  std::optional<std::pair<int64_t, int64_t>> range;
  op->getLoc()->walk([&](mlir::Location loc) {
    auto trivia = mlir::dyn_cast<maldoca::JsirTriviaAttr>(loc);
    if (trivia == nullptr || trivia.getLoc() == nullptr) {
      return mlir::WalkResult::advance();
    }
    std::optional<int64_t> start = trivia.getLoc().getStartIndex();
    std::optional<int64_t> end = trivia.getLoc().getEndIndex();
    if (!start.has_value() || !end.has_value()) {
      return mlir::WalkResult::advance();
    }
    range = {*start, *end};
    return mlir::WalkResult::interrupt();
  });
  return range;
}

// Converts a Babel AST (as a JSON string) into JSHIR and prints it.
JsirResult AstJsonToJsir(const std::string& ast_json, bool print_locations) {
  nlohmann::json json = nlohmann::json::parse(ast_json, /*cb=*/nullptr,
                                              /*allow_exceptions=*/false);
  if (json.is_discarded()) {
    return Error("JSON parsing",
                 absl::InvalidArgumentError("Invalid AST JSON."));
  }

  absl::StatusOr<std::unique_ptr<maldoca::JsFile>> ast =
      maldoca::JsFile::FromJson(json);
  if (!ast.ok()) {
    return Error("AST deserialization", ast.status());
  }

  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  maldoca::LoadNecessaryDialects(context);

  absl::StatusOr<mlir::OwningOpRef<maldoca::JsirFileOp>> hir =
      maldoca::AstToJshirFile(**ast, context);
  if (!hir.ok()) {
    return Error("AST to JSHIR conversion", hir.status());
  }

  mlir::OpPrintingFlags flags;
  if (print_locations) {
    flags.enableDebugInfo(/*enable=*/true, /*prettyForm=*/false);
  }

  std::string output;
  llvm::raw_string_ostream os(output);
  mlir::AsmState::LocationMap location_map;
  mlir::AsmState state((*hir)->getOperation(), flags, &location_map);
  (*hir)->print(os, state);
  os.flush();

  nlohmann::json source_map = nlohmann::json::array();
  (*hir)->walk([&](mlir::Operation* op) {
    auto it = location_map.find(op);
    if (it == location_map.end()) return;
    std::optional<std::pair<int64_t, int64_t>> range = GetSourceRange(op);
    if (!range.has_value()) return;
    const auto& [line, column] = it->second;
    source_map.push_back({line, column, range->first, range->second});
  });

  return JsirResult{
      .ok = true,
      .output = std::move(output),
      .source_map = source_map.dump(),
  };
}

}  // namespace

EMSCRIPTEN_BINDINGS(jsir_wasm_module) {
  emscripten::value_object<JsirResult>("JsirResult")
      .field("ok", &JsirResult::ok)
      .field("output", &JsirResult::output)
      .field("sourceMap", &JsirResult::source_map);

  emscripten::function("astJsonToJsir", &AstJsonToJsir);
}
