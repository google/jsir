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

#ifndef MALDOCA_ASTGEN_TEST_CONVERSION_TEST_UTIL_H_
#define MALDOCA_ASTGEN_TEST_CONVERSION_TEST_UTIL_H_

#include <memory>
#include <sstream>
#include <string>

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "nlohmann/json.hpp"
#include "maldoca/base/testing/status_matchers.h"

namespace maldoca {

class DummyIrToAst {};

template <typename NodeT, typename OpT, typename AstToIr, typename IrToAst>
struct ConversionTestCase {
  std::string ast_json_string;
  std::unique_ptr<NodeT> ast;
  OpT (AstToIr::*ast_to_ir_visit)(const NodeT *);
  absl::StatusOr<std::unique_ptr<NodeT>> (IrToAst::*ir_to_ast_visit)(OpT);
  std::string expected_ir_dump;
};

template <typename NodeT, typename OpT, typename Dialect, typename AstToIr,
          typename IrToAst = DummyIrToAst>
void TestIrConversion(
    ConversionTestCase<NodeT, OpT, AstToIr, IrToAst> &&test_case) {
  if (!test_case.ast_json_string.empty()) {
    auto ast_json = nlohmann::json::parse(test_case.ast_json_string,
                                          /*callback=*/nullptr,
                                          /*allow_exceptions=*/false,
                                          /*ignore_comments=*/false);
    ASSERT_FALSE(ast_json.is_discarded())
        << "Failed to parse AST: Invalid JSON.";

    MALDOCA_ASSERT_OK_AND_ASSIGN(test_case.ast, NodeT::FromJson(ast_json));
  }

  mlir::MLIRContext context;
  context.getOrLoadDialect<Dialect>();
  mlir::OpBuilder builder(&context);

  // A file is modeled as a "module" in MLIR.
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(builder.getUnknownLoc());

  mlir::Block *block = &module->getBodyRegion().front();
  builder.setInsertionPointToStart(block);

  AstToIr ast_to_ir(builder);
  OpT op = (ast_to_ir.*(test_case.ast_to_ir_visit))(test_case.ast.get());

  std::string ir_dump;
  llvm::raw_string_ostream os(ir_dump);
  module->print(os);

  EXPECT_EQ(absl::StripAsciiWhitespace(ir_dump),
            absl::StripAsciiWhitespace(test_case.expected_ir_dump));

  if (test_case.ir_to_ast_visit != nullptr) {
    IrToAst ir_to_ast;
    MALDOCA_ASSERT_OK_AND_ASSIGN(auto raised_ast,
                                 (ir_to_ast.*(test_case.ir_to_ast_visit))(op));

    std::stringstream test_case_ss;
    test_case.ast->Serialize(test_case_ss);

    std::stringstream raised_ast_ss;
    raised_ast->Serialize(raised_ast_ss);

    EXPECT_EQ(test_case_ss.str(), raised_ast_ss.str());
  }
}

}  // namespace maldoca

#endif  // MALDOCA_ASTGEN_TEST_CONVERSION_TEST_UTIL_H_
