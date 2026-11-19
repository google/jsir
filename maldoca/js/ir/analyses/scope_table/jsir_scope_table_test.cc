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

#include "maldoca/js/ir/analyses/scope_table/jsir_scope_table.h"

#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace maldoca {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

class JsirScopeTableTest : public ::testing::Test {
 protected:
  mlir::Operation* f() { return f_->getOperation(); }
  mlir::Operation* g() { return g_->getOperation(); }

  // Ops standing for two functions.
  mlir::MLIRContext context_;
  mlir::OwningOpRef<mlir::ModuleOp> f_ =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context_));
  mlir::OwningOpRef<mlir::ModuleOp> g_ =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context_));
};

TEST_F(JsirScopeTableTest, StartsEmpty) {
  JsirScopeTable table;
  EXPECT_EQ(table.scope(), nullptr);
  EXPECT_TRUE(table.empty());
  EXPECT_THAT(table.symbols(), IsEmpty());
  EXPECT_EQ(table.Lookup("x"), nullptr);
}

TEST_F(JsirScopeTableTest, DeclareAndLookup) {
  JsirScopeTable table(f());
  EXPECT_EQ(table.scope(), f());
  const JsirSymbol& x = table.Declare("x", JsirSymbolKind::kLocal);
  const JsirSymbol& env = table.Declare("env", JsirSymbolKind::kEnv, g());

  // Declare fills in the table's scope.
  EXPECT_EQ(x.name, "x");
  EXPECT_EQ(x.kind, JsirSymbolKind::kLocal);
  EXPECT_EQ(x.scope, f());
  EXPECT_EQ(x.owner, nullptr);
  EXPECT_EQ(env.kind, JsirSymbolKind::kEnv);
  EXPECT_EQ(env.scope, f());
  EXPECT_EQ(env.owner, g());

  EXPECT_EQ(table.Lookup("x"), &x);
  EXPECT_EQ(table.Lookup("env"), &env);
  EXPECT_TRUE(table.Contains("x"));
  EXPECT_FALSE(table.Contains("y"));
  EXPECT_THAT(table.symbols(), ElementsAre(&x, &env));
}

TEST_F(JsirScopeTableTest, ProgramTableHasNoScope) {
  JsirScopeTable program;
  EXPECT_EQ(program.Declare("env", JsirSymbolKind::kEnv, f()).scope, nullptr);
}

TEST_F(JsirScopeTableTest, FirstDeclarationWins) {
  JsirScopeTable table(f());
  const JsirSymbol& first = table.Declare("x", JsirSymbolKind::kLocal);
  const JsirSymbol& second = table.Declare("x", JsirSymbolKind::kEnv, g());

  EXPECT_EQ(&first, &second);
  EXPECT_EQ(table.size(), 1);
  ASSERT_NE(table.Lookup("x"), nullptr);
  EXPECT_EQ(table.Lookup("x")->kind, JsirSymbolKind::kLocal);
  EXPECT_EQ(table.Lookup("x")->owner, nullptr);
}

TEST_F(JsirScopeTableTest, EqualityIsNameAndScope) {
  JsirSymbol x{.name = "x", .kind = JsirSymbolKind::kLocal, .scope = f()};

  // The same variable, whatever its kind or owner.
  EXPECT_EQ(x, (JsirSymbol{.name = "x",
                           .kind = JsirSymbolKind::kParam,
                           .scope = f(),
                           .owner = g()}));
  // Another name in the same scope, or the same name in another scope.
  EXPECT_FALSE(
      x ==
      (JsirSymbol{.name = "y", .kind = JsirSymbolKind::kLocal, .scope = f()}));
  EXPECT_FALSE(
      x ==
      (JsirSymbol{.name = "x", .kind = JsirSymbolKind::kLocal, .scope = g()}));
  EXPECT_FALSE(x == (JsirSymbol{.name = "x", .kind = JsirSymbolKind::kLocal}));
}

TEST_F(JsirScopeTableTest, SymbolsStayValidAsTheTableGrows) {
  JsirScopeTable table(f());
  const JsirSymbol* first = &table.Declare("tmp_0", JsirSymbolKind::kLocal);
  for (int i = 1; i < 1000; ++i) {
    table.Declare("tmp_" + std::to_string(i), JsirSymbolKind::kLocal);
  }
  EXPECT_EQ(table.Lookup("tmp_0"), first);
  EXPECT_EQ(first->name, "tmp_0");
  EXPECT_EQ(table.size(), 1000);
}

}  // namespace
}  // namespace maldoca
