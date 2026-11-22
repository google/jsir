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

#include "maldoca/js/ir/analyses/scope_table/jsir_scope_registry.h"

#include <atomic>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/WalkResult.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "maldoca/js/ir/analyses/scope_table/jsir_scope_table.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {

// Test passes. They are not in an anonymous namespace because MLIR's implicit
// TypeID fallback rejects types in anonymous namespaces.
namespace scope_registry_test {

// The passes below treat each `builtin.module` nested in the top-level module
// as a function: unlike JSIR functions, nested modules are isolated from
// above, so a pass manager can run per-"function" passes on them in parallel.

// Creates the registry and a table per function, and declares `t` in each.
struct InitPass
    : public mlir::PassWrapper<InitPass, mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    auto& registry = getAnalysis<JsirScopeRegistry>();
    for (auto function : getOperation().getOps<mlir::ModuleOp>()) {
      registry.AddScope(function).Declare("t", JsirSymbolKind::kVar);
    }
  }
};

// Preserves no analyses.
struct PreserveNothingPass
    : public mlir::PassWrapper<PreserveNothingPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {}
};

// A per-function pass: counts the functions that can reach their table, and
// declares a name in it.
struct PerFunctionPass
    : public mlir::PassWrapper<PerFunctionPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  explicit PerFunctionPass(std::atomic<int>* num_found)
      : num_found(num_found) {}

  void runOnOperation() override {
    mlir::Operation* parent = getOperation()->getParentOp();
    auto registry = getCachedParentAnalysis<JsirScopeRegistry>(parent);
    if (!registry.has_value()) {
      return;
    }
    JsirScopeTable* table = registry->get().GetScopeTableFor(getOperation());
    if (table == nullptr) {
      return;
    }
    table->Declare("from_nested_pass", JsirSymbolKind::kVar);
    num_found->fetch_add(1, std::memory_order_relaxed);
  }

  std::atomic<int>* num_found;
};

// Counts the functions whose tables have the declarations made by both
// earlier passes.
struct CheckPass
    : public mlir::PassWrapper<CheckPass, mlir::OperationPass<mlir::ModuleOp>> {
  explicit CheckPass(int* num_complete) : num_complete(num_complete) {}

  void runOnOperation() override {
    auto registry = getCachedAnalysis<JsirScopeRegistry>();
    if (!registry.has_value()) {
      return;
    }
    // The functions InitPass created tables for.
    for (auto function : getOperation().getOps<mlir::ModuleOp>()) {
      const JsirScopeTable* table = registry->get().GetScopeTable(function);
      if (table != nullptr && table->Contains("t") &&
          table->Contains("from_nested_pass")) {
        ++*num_complete;
      }
    }
  }

  int* num_complete;
};

}  // namespace scope_registry_test

namespace {

using ::maldoca::scope_registry_test::CheckPass;
using ::maldoca::scope_registry_test::InitPass;
using ::maldoca::scope_registry_test::PerFunctionPass;
using ::maldoca::scope_registry_test::PreserveNothingPass;
using ::testing::ElementsAre;
using ::testing::UnorderedElementsAre;

// Modeled on decompiled Hermes JSIR. `f` and `g` are functions with tables;
// `bb1` and `bb2` are pseudo-functions nested in `f`, without tables.
//
//   function f(p0) {
//     function bb1($arg0) {
//       $reg0 = $arg0;
//       bb2;
//       return env_0_0_var_0;
//     }
//     function bb2() {
//       return tmp_0;
//     }
//     tmp_0 = p0;
//     env_0_0_var_0;
//     return bb1;
//   }
//   function g(p0) {
//     env_0_0_var_0 = g;
//     return $arg0;
//   }
constexpr char kInput[] = R"mlir(
#locF = #jsir<trivia <L 0 C 0>, <L 0 C 0>, "f", 1, 1>
#locG = #jsir<trivia <L 0 C 0>, <L 0 C 0>, "g", 1, 1>
#locB1 = #jsir<trivia <L 0 C 0>, <L 0 C 0>, "bb1", 0, 0, 0>
#locB2 = #jsir<trivia <L 0 C 0>, <L 0 C 0>, "bb2", 0, 0, 0>
module {
  "jsir.program"() <{source_type = "script"}> ({
    "jsir.function_declaration"() <{async = false, generator = false, id = #jsir<identifier #locF, "f">}> ({
      %fp = "jsir.identifier_ref"() <{name = "p0"}> : () -> !jsir.any
      "jsir.exprs_region_end"(%fp) : (!jsir.any) -> ()
    }, {
      "jshir.block_statement"() ({
        "jsir.function_declaration"() <{async = false, generator = false, id = #jsir<identifier #locB1, "bb1">}> ({
          %ap = "jsir.identifier_ref"() <{name = "$arg0"}> : () -> !jsir.any
          "jsir.exprs_region_end"(%ap) : (!jsir.any) -> ()
        }, {
          "jshir.block_statement"() ({
            %a0 = "jsir.identifier"() <{name = "$arg0"}> : () -> !jsir.any
            %a1 = "jsir.identifier_ref"() <{name = "$reg0"}> : () -> !jsir.any
            %a2 = "jsir.assignment_expression"(%a1, %a0) <{operator_ = "="}> : (!jsir.any, !jsir.any) -> !jsir.any
            "jsir.expression_statement"(%a2) : (!jsir.any) -> ()
            %a3 = "jsir.identifier"() <{name = "bb2"}> : () -> !jsir.any
            "jsir.expression_statement"(%a3) : (!jsir.any) -> ()
            %a4 = "jsir.identifier"() <{name = "env_0_0_var_0"}> : () -> !jsir.any
            "jsir.return_statement"(%a4) : (!jsir.any) -> ()
          }, {
          ^bb0:
          }) : () -> ()
        }) : () -> ()
        "jsir.function_declaration"() <{async = false, generator = false, id = #jsir<identifier #locB2, "bb2">}> ({
          "jsir.exprs_region_end"() : () -> ()
        }, {
          "jshir.block_statement"() ({
            %b0 = "jsir.identifier"() <{name = "tmp_0"}> : () -> !jsir.any
            "jsir.return_statement"(%b0) : (!jsir.any) -> ()
          }, {
          ^bb0:
          }) : () -> ()
        }) : () -> ()
        %f0 = "jsir.identifier"() <{name = "p0"}> : () -> !jsir.any
        %f1 = "jsir.identifier_ref"() <{name = "tmp_0"}> : () -> !jsir.any
        %f2 = "jsir.assignment_expression"(%f1, %f0) <{operator_ = "="}> : (!jsir.any, !jsir.any) -> !jsir.any
        "jsir.expression_statement"(%f2) : (!jsir.any) -> ()
        %f3 = "jsir.identifier"() <{name = "env_0_0_var_0"}> : () -> !jsir.any
        "jsir.expression_statement"(%f3) : (!jsir.any) -> ()
        %f4 = "jsir.identifier"() <{name = "bb1"}> : () -> !jsir.any
        "jsir.return_statement"(%f4) : (!jsir.any) -> ()
      }, {
      ^bb0:
      }) : () -> ()
    }) : () -> ()
    "jsir.function_declaration"() <{async = false, generator = false, id = #jsir<identifier #locG, "g">}> ({
      %gp = "jsir.identifier_ref"() <{name = "p0"}> : () -> !jsir.any
      "jsir.exprs_region_end"(%gp) : (!jsir.any) -> ()
    }, {
      "jshir.block_statement"() ({
        %g0 = "jsir.identifier"() <{name = "g"}> : () -> !jsir.any
        %g1 = "jsir.identifier_ref"() <{name = "env_0_0_var_0"}> : () -> !jsir.any
        %g2 = "jsir.assignment_expression"(%g1, %g0) <{operator_ = "="}> : (!jsir.any, !jsir.any) -> !jsir.any
        "jsir.expression_statement"(%g2) : (!jsir.any) -> ()
        %g3 = "jsir.identifier"() <{name = "$arg0"}> : () -> !jsir.any
        "jsir.return_statement"(%g3) : (!jsir.any) -> ()
      }, {
      ^bb0:
      }) : () -> ()
    }) : () -> ()
  }, {
  ^bb0:
  }) : () -> ()
}
)mlir";

class JsirScopeRegistryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mlir::DialectRegistry registry;
    registry.insert<JsirDialect, JshirDialect>();
    context_.appendDialectRegistry(registry);
    context_.loadAllAvailableDialects();
    module_ = mlir::parseSourceString<mlir::ModuleOp>(kInput, &context_);
    ASSERT_TRUE(module_);
    module_->walk([&](JsirFunctionDeclarationOp op) {
      functions_[op.getIdAttr().getName().str()] = op;
    });
    f_ = functions_["f"];
    g_ = functions_["g"];
    bb1_ = functions_["bb1"];
    bb2_ = functions_["bb2"];
    ASSERT_NE(f_, nullptr);
    ASSERT_NE(g_, nullptr);
    ASSERT_NE(bb1_, nullptr);
    ASSERT_NE(bb2_, nullptr);
  }

  // Registers what the decompiler would: a table per function, the
  // temporaries and registers of `f`, and `f`'s closure variable.
  void Register(JsirScopeRegistry& registry) {
    registry.AddScope(f_);
    registry.AddScope(g_);
    registry.GetScopeTable(f_)->Declare("tmp_0", JsirSymbolKind::kVar);
    registry.GetScopeTable(f_)->Declare("$reg0", JsirSymbolKind::kVar);
    registry.program().Declare("env_0_0_var_0", JsirSymbolKind::kClosureVar,
                               f_);
  }

  // The first identifier (in pre-order) named `name` directly in the scope of
  // `function`, i.e. not in a function nested in it. `ref` selects
  // `jsir.identifier_ref` (true) or `jsir.identifier` (false).
  mlir::Operation* Find(mlir::Operation* function, llvm::StringRef name,
                        bool ref = false) {
    mlir::Operation* result = nullptr;
    function->walk<mlir::WalkOrder::PreOrder>(
        [&](mlir::Operation* op) -> mlir::WalkResult {
          if (op != function && llvm::isa<JsirFunctionDeclarationOp>(op)) {
            return mlir::WalkResult::skip();
          }
          bool matches =
              ref ? llvm::isa<JsirIdentifierRefOp>(op) &&
                        llvm::cast<JsirIdentifierRefOp>(op).getName() == name
                  : llvm::isa<JsirIdentifierOp>(op) &&
                        llvm::cast<JsirIdentifierOp>(op).getName() == name;
          if (matches) {
            result = op;
            return mlir::WalkResult::interrupt();
          }
          return mlir::WalkResult::advance();
        });
    return result;
  }

  // Resolves the identifier `name` in `function` (see Find).
  std::optional<JsirSymbol> ResolveIn(const JsirScopeRegistry& registry,
                                      mlir::Operation* function,
                                      llvm::StringRef name, bool ref = false) {
    mlir::Operation* identifier = Find(function, name, ref);
    EXPECT_NE(identifier, nullptr) << name.str();
    if (identifier == nullptr) {
      return std::nullopt;
    }
    return registry.Resolve(identifier);
  }

  mlir::MLIRContext context_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  std::map<std::string, mlir::Operation*> functions_;
  mlir::Operation* f_ = nullptr;
  mlir::Operation* g_ = nullptr;
  mlir::Operation* bb1_ = nullptr;
  mlir::Operation* bb2_ = nullptr;
};

TEST_F(JsirScopeRegistryTest, StartsEmpty) {
  JsirScopeRegistry registry(*module_);
  EXPECT_TRUE(registry.program().empty());
  EXPECT_EQ(registry.program().scope(), nullptr);
  EXPECT_EQ(registry.GetScopeTable(f_), nullptr);
  EXPECT_EQ(registry.GetScopeTable(g_), nullptr);
  EXPECT_EQ(registry.GetScopeTableFor(bb1_), nullptr);
  // Nothing is registered, so only names that ops declare themselves resolve.
  EXPECT_EQ(ResolveIn(registry, f_, "tmp_0", /*ref=*/true), std::nullopt);
  std::optional<JsirSymbol> bb1 = ResolveIn(registry, f_, "bb1");
  ASSERT_TRUE(bb1.has_value());
  EXPECT_EQ(bb1->kind, JsirSymbolKind::kFunction);
  EXPECT_EQ(bb1->scope, f_);
}

TEST_F(JsirScopeRegistryTest, Tables) {
  JsirScopeRegistry registry(*module_);
  JsirScopeTable& f_table = registry.AddScope(f_);
  EXPECT_EQ(f_table.scope(), f_);
  EXPECT_EQ(&registry.AddScope(f_), &f_table);
  registry.AddScope(g_);

  EXPECT_EQ(registry.GetScopeTable(f_), &f_table);
  // Pseudo-functions have no table of their own ...
  EXPECT_EQ(registry.GetScopeTable(bb1_), nullptr);
  // ... and use their enclosing function's.
  EXPECT_EQ(registry.GetScopeTableFor(bb1_), &f_table);
  EXPECT_EQ(registry.GetScopeTableFor(Find(bb2_, "tmp_0")), &f_table);
  EXPECT_EQ(registry.GetScopeTableFor(module_->getOperation()), nullptr);
  EXPECT_NE(registry.GetScopeTable(g_), nullptr);
  EXPECT_NE(registry.GetScopeTable(g_), &f_table);
}

TEST_F(JsirScopeRegistryTest, ResolutionOrder) {
  JsirScopeRegistry registry(*module_);
  Register(registry);
  // Also declared in f's table: shadowed by bb1's parameter inside bb1.
  registry.GetScopeTable(f_)->Declare("$arg0", JsirSymbolKind::kVar);

  // 1. Parameters of each enclosing function op: bb1's `$arg0` (declared by
  // the pseudo-function bb1) ...
  std::optional<JsirSymbol> arg0 = ResolveIn(registry, bb1_, "$arg0");
  ASSERT_TRUE(arg0.has_value());
  EXPECT_EQ(arg0->kind, JsirSymbolKind::kParam);
  EXPECT_EQ(arg0->scope, bb1_);
  EXPECT_EQ(arg0->owner, nullptr);
  // ... its declaration site in bb1's params region ...
  EXPECT_EQ(ResolveIn(registry, bb1_, "$arg0", /*ref=*/true), arg0);
  // ... and f's `p0`.
  std::optional<JsirSymbol> f_p0 = ResolveIn(registry, f_, "p0");
  ASSERT_TRUE(f_p0.has_value());
  EXPECT_EQ(f_p0->kind, JsirSymbolKind::kParam);
  EXPECT_EQ(f_p0->scope, f_);

  // Functions declared directly in an enclosing function op: a pseudo-function
  // referenced from a sibling is declared by f.
  std::optional<JsirSymbol> bb2 = ResolveIn(registry, bb1_, "bb2");
  ASSERT_TRUE(bb2.has_value());
  EXPECT_EQ(bb2->kind, JsirSymbolKind::kFunction);
  EXPECT_EQ(bb2->scope, f_);

  // 2. The enclosing function's table, from a pseudo-function nested in it.
  std::optional<JsirSymbol> tmp_0 = ResolveIn(registry, bb2_, "tmp_0");
  ASSERT_TRUE(tmp_0.has_value());
  EXPECT_EQ(tmp_0->kind, JsirSymbolKind::kVar);
  EXPECT_EQ(tmp_0->scope, f_);
  EXPECT_EQ(ResolveIn(registry, f_, "tmp_0", /*ref=*/true), tmp_0);
  std::optional<JsirSymbol> reg0 =
      ResolveIn(registry, bb1_, "$reg0", /*ref=*/true);
  ASSERT_TRUE(reg0.has_value());
  EXPECT_EQ(reg0->scope, f_);

  // 3. Program-level functions, then the program table.
  std::optional<JsirSymbol> g = ResolveIn(registry, g_, "g");
  ASSERT_TRUE(g.has_value());
  EXPECT_EQ(g->kind, JsirSymbolKind::kFunction);
  EXPECT_EQ(g->scope, nullptr);
  std::optional<JsirSymbol> env =
      ResolveIn(registry, g_, "env_0_0_var_0", /*ref=*/true);
  ASSERT_TRUE(env.has_value());
  EXPECT_EQ(env->kind, JsirSymbolKind::kClosureVar);
  EXPECT_EQ(env->scope, nullptr);

  // 4. Globals: bb1's parameter is not visible in g, and g's table does not
  // have f's temporaries.
  EXPECT_EQ(ResolveIn(registry, g_, "$arg0"), std::nullopt);
  EXPECT_EQ(registry.Resolve(Find(g_, "g"), "tmp_0"), std::nullopt);
}

TEST_F(JsirScopeRegistryTest, SameVariable) {
  JsirScopeRegistry registry(*module_);
  Register(registry);

  // Uses of the same variable resolve to equal symbols ...
  EXPECT_EQ(ResolveIn(registry, bb2_, "tmp_0"),
            ResolveIn(registry, f_, "tmp_0", /*ref=*/true));
  EXPECT_EQ(ResolveIn(registry, bb1_, "env_0_0_var_0"),
            ResolveIn(registry, g_, "env_0_0_var_0", /*ref=*/true));
  // ... and the same name in different scopes does not.
  std::optional<JsirSymbol> f_p0 = ResolveIn(registry, f_, "p0");
  std::optional<JsirSymbol> g_p0 = ResolveIn(registry, g_, "p0", /*ref=*/true);
  ASSERT_TRUE(f_p0.has_value());
  ASSERT_TRUE(g_p0.has_value());
  EXPECT_EQ(g_p0->scope, g_);
  EXPECT_FALSE(*f_p0 == *g_p0);
}

TEST_F(JsirScopeRegistryTest, EnvironmentVariableOwner) {
  JsirScopeRegistry registry(*module_);
  Register(registry);
  registry.program().Declare("env_UNK_0_var_0", JsirSymbolKind::kClosureVar);

  std::optional<JsirSymbol> env = ResolveIn(registry, bb1_, "env_0_0_var_0");
  ASSERT_TRUE(env.has_value());
  EXPECT_EQ(env->owner, f_);

  std::optional<JsirSymbol> unknown =
      registry.Resolve(Find(g_, "g"), "env_UNK_0_var_0");
  ASSERT_TRUE(unknown.has_value());
  EXPECT_EQ(unknown->kind, JsirSymbolKind::kClosureVar);
  EXPECT_EQ(unknown->owner, nullptr);
}

TEST_F(JsirScopeRegistryTest, UsesOnDemand) {
  JsirScopeRegistry registry(*module_);
  Register(registry);

  std::optional<JsirSymbol> tmp_0 = ResolveIn(registry, bb2_, "tmp_0");
  ASSERT_TRUE(tmp_0.has_value());
  EXPECT_THAT(registry.GetUses(*tmp_0),
              UnorderedElementsAre(Find(bb2_, "tmp_0"),
                                   Find(f_, "tmp_0", /*ref=*/true)));

  // Parameter declaration sites are not uses.
  std::optional<JsirSymbol> f_p0 = ResolveIn(registry, f_, "p0");
  ASSERT_TRUE(f_p0.has_value());
  EXPECT_THAT(registry.GetUses(*f_p0), ElementsAre(Find(f_, "p0")));
  std::optional<JsirSymbol> arg0 = ResolveIn(registry, bb1_, "$arg0");
  ASSERT_TRUE(arg0.has_value());
  EXPECT_THAT(registry.GetUses(*arg0), ElementsAre(Find(bb1_, "$arg0")));

  // Closure variables are used across functions.
  std::optional<JsirSymbol> env = ResolveIn(registry, f_, "env_0_0_var_0");
  ASSERT_TRUE(env.has_value());
  EXPECT_THAT(registry.GetUses(*env),
              UnorderedElementsAre(Find(bb1_, "env_0_0_var_0"),
                                   Find(f_, "env_0_0_var_0"),
                                   Find(g_, "env_0_0_var_0", /*ref=*/true)));
}

TEST_F(JsirScopeRegistryTest, CapturedOnDemand) {
  JsirScopeRegistry registry(*module_);
  Register(registry);
  registry.program().Declare("env_UNK_0_var_0", JsirSymbolKind::kClosureVar);

  // Used in a pseudo-function nested in f.
  EXPECT_TRUE(registry.IsCaptured(*ResolveIn(registry, bb2_, "tmp_0")));
  // Owned by f and used in g.
  EXPECT_TRUE(registry.IsCaptured(*ResolveIn(registry, f_, "env_0_0_var_0")));
  // Only used in the function declaring them.
  EXPECT_FALSE(registry.IsCaptured(*ResolveIn(registry, f_, "p0")));
  EXPECT_FALSE(registry.IsCaptured(*ResolveIn(registry, bb1_, "$arg0")));
  // Unused.
  EXPECT_FALSE(
      registry.IsCaptured(*registry.Resolve(Find(g_, "g"), "env_UNK_0_var_0")));
  // Function names are never captured.
  EXPECT_FALSE(registry.IsCaptured(*ResolveIn(registry, bb1_, "bb2")));
}

TEST_F(JsirScopeRegistryTest, TransferScopeFollowsLowering) {
  JsirScopeRegistry registry(*module_);
  Register(registry);
  registry.program().Declare("env_UNK_0_var_0", JsirSymbolKind::kClosureVar);
  JsirScopeTable* f_table = registry.GetScopeTable(f_);

  // Simulate a lowering that replaces f with a new op taking over its body.
  mlir::OpBuilder builder(f_);
  mlir::Operation* lowered = builder.clone(*f_);
  registry.TransferScope(f_, lowered);

  // The table moved to the new op, and its symbols with it ...
  EXPECT_EQ(registry.GetScopeTable(f_), nullptr);
  EXPECT_EQ(registry.GetScopeTable(lowered), f_table);
  EXPECT_EQ(f_table->scope(), lowered);
  EXPECT_EQ(f_table->Lookup("tmp_0")->scope, lowered);
  EXPECT_EQ(f_table->Lookup("$reg0")->scope, lowered);
  EXPECT_NE(registry.GetScopeTable(g_), nullptr);
  // ... and so did the closure variables it owns. Unowned ones stay so.
  EXPECT_EQ(registry.program().Lookup("env_0_0_var_0")->owner, lowered);
  EXPECT_EQ(registry.program().Lookup("env_0_0_var_0")->scope, nullptr);
  EXPECT_EQ(registry.program().Lookup("env_UNK_0_var_0")->owner, nullptr);
  f_->erase();
  f_ = nullptr;

  // What was declared in the table resolves in the new op.
  std::optional<JsirSymbol> tmp_0 =
      ResolveIn(registry, lowered, "tmp_0", /*ref=*/true);
  ASSERT_TRUE(tmp_0.has_value());
  EXPECT_EQ(tmp_0->kind, JsirSymbolKind::kVar);
  EXPECT_EQ(tmp_0->scope, lowered);
  EXPECT_EQ(registry.GetScopeTableFor(Find(lowered, "p0")), f_table);
  std::optional<JsirSymbol> p0 = ResolveIn(registry, lowered, "p0");
  ASSERT_TRUE(p0.has_value());
  EXPECT_EQ(p0->scope, lowered);
  EXPECT_TRUE(
      registry.IsCaptured(*ResolveIn(registry, lowered, "env_0_0_var_0")));

  // Transferring an op without a table does nothing.
  registry.TransferScope(module_->getOperation(), lowered);
  EXPECT_EQ(registry.GetScopeTable(lowered), f_table);
}

TEST_F(JsirScopeRegistryTest, RemoveScope) {
  JsirScopeRegistry registry(*module_);
  Register(registry);

  registry.RemoveScope(f_);
  EXPECT_EQ(registry.GetScopeTable(f_), nullptr);
  EXPECT_EQ(registry.GetScopeTableFor(bb2_), nullptr);
  EXPECT_NE(registry.GetScopeTable(g_), nullptr);
  EXPECT_EQ(ResolveIn(registry, bb2_, "tmp_0"), std::nullopt);
  // The closure variables f owned are kept, without an owner.
  ASSERT_NE(registry.program().Lookup("env_0_0_var_0"), nullptr);
  EXPECT_EQ(registry.program().Lookup("env_0_0_var_0")->owner, nullptr);
}

TEST_F(JsirScopeRegistryTest, MergeScopeInto) {
  JsirScopeRegistry registry(*module_);
  Register(registry);
  registry.GetScopeTable(g_)->Declare("x", JsirSymbolKind::kVar);
  registry.GetScopeTable(g_)->Declare("tmp_0", JsirSymbolKind::kVar);
  registry.program().Declare("env_UNK_0_var_0", JsirSymbolKind::kClosureVar);

  registry.MergeScopeInto(f_, g_);
  EXPECT_EQ(registry.GetScopeTable(f_), nullptr);
  const JsirScopeTable* g_table = registry.GetScopeTable(g_);
  ASSERT_NE(g_table, nullptr);
  std::vector<std::string> names;
  for (const JsirSymbol* symbol : g_table->symbols()) {
    names.push_back(symbol->name);
    // The callee's declarations now belong to the caller.
    EXPECT_EQ(symbol->scope, g_) << symbol->name;
  }
  // `tmp_0`, declared in both, is kept once.
  EXPECT_THAT(names, ElementsAre("x", "tmp_0", "$reg0"));
  // The inlined body runs in the caller's activation, so the callee's
  // closure variables are now owned by the caller. Unowned ones stay so.
  EXPECT_EQ(registry.program().Lookup("env_0_0_var_0")->owner, g_);
  EXPECT_EQ(registry.program().Lookup("env_0_0_var_0")->scope, nullptr);
  EXPECT_EQ(registry.program().Lookup("env_UNK_0_var_0")->owner, nullptr);
}

TEST_F(JsirScopeRegistryTest, RegistrySurvivesPassesThatPreserveNothing) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
        module {
          module {}
          module {}
        }
      )mlir",
                                              &context_);
  ASSERT_TRUE(module);

  std::atomic<int> num_found = 0;
  int num_complete = 0;
  mlir::PassManager pm(&context_);
  pm.addPass(std::make_unique<InitPass>());
  pm.addPass(std::make_unique<PreserveNothingPass>());
  pm.nest<mlir::ModuleOp>().addPass(
      std::make_unique<PerFunctionPass>(&num_found));
  pm.addPass(std::make_unique<PreserveNothingPass>());
  pm.addPass(std::make_unique<CheckPass>(&num_complete));
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  // Both functions found their table after a pass that preserved nothing, and
  // their declarations were still there after another one.
  EXPECT_EQ(num_found, 2);
  EXPECT_EQ(num_complete, 2);
}

// Parameters with default values and destructuring patterns:
//
//   function f(a, b = (c = d), {x, y: [z, ...r]} = d,
//              e = (() => { a = d; })()) {
//     function h() {
//       return e;
//     }
//   }
constexpr char kParamsInput[] = R"mlir(
#locF = #jsir<trivia <L 0 C 0>, <L 0 C 0>, "f", 1, 1>
#locH = #jsir<trivia <L 0 C 0>, <L 0 C 0>, "h", 1, 1>
#locX = #jsir<trivia <L 0 C 0>, <L 0 C 0>, "x", 0, 0, 0>
#locY = #jsir<trivia <L 0 C 0>, <L 0 C 0>, "y", 0, 0, 0>
module {
  "jsir.program"() <{source_type = "script"}> ({
    "jsir.function_declaration"() <{async = false, generator = false, id = #jsir<identifier #locF, "f">}> ({
      %a = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
      %b = "jsir.identifier_ref"() <{name = "b"}> : () -> !jsir.any
      %c = "jsir.identifier_ref"() <{name = "c"}> : () -> !jsir.any
      %d0 = "jsir.identifier"() <{name = "d"}> : () -> !jsir.any
      %c_assign = "jsir.assignment_expression"(%c, %d0) <{operator_ = "="}> : (!jsir.any, !jsir.any) -> !jsir.any
      %b_pattern = "jsir.assignment_pattern_ref"(%b, %c_assign) : (!jsir.any, !jsir.any) -> !jsir.any
      %object = "jsir.object_pattern_ref"() ({
        %x = "jsir.identifier_ref"() <{name = "x"}> : () -> !jsir.any
        %x_property = "jsir.object_property_ref"(%x) <{literal_key = #jsir<identifier #locX, "x">, shorthand = true}> : (!jsir.any) -> !jsir.any
        %z = "jsir.identifier_ref"() <{name = "z"}> : () -> !jsir.any
        %r = "jsir.identifier_ref"() <{name = "r"}> : () -> !jsir.any
        %rest = "jsir.rest_element_ref"(%r) : (!jsir.any) -> !jsir.any
        %array = "jsir.array_pattern_ref"(%z, %rest) : (!jsir.any, !jsir.any) -> !jsir.any
        %y_property = "jsir.object_property_ref"(%array) <{literal_key = #jsir<identifier #locY, "y">, shorthand = false}> : (!jsir.any) -> !jsir.any
        "jsir.exprs_region_end"(%x_property, %y_property) : (!jsir.any, !jsir.any) -> ()
      }) : () -> !jsir.any
      %d1 = "jsir.identifier"() <{name = "d"}> : () -> !jsir.any
      %object_pattern = "jsir.assignment_pattern_ref"(%object, %d1) : (!jsir.any, !jsir.any) -> !jsir.any
      %e = "jsir.identifier_ref"() <{name = "e"}> : () -> !jsir.any
      %arrow = "jsir.arrow_function_expression"() <{async = false, generator = false}> ({
        "jsir.exprs_region_end"() : () -> ()
      }, {
        "jshir.block_statement"() ({
          %w = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
          %d2 = "jsir.identifier"() <{name = "d"}> : () -> !jsir.any
          %w_assign = "jsir.assignment_expression"(%w, %d2) <{operator_ = "="}> : (!jsir.any, !jsir.any) -> !jsir.any
          "jsir.expression_statement"(%w_assign) : (!jsir.any) -> ()
        }, {
        ^bb0:
        }) : () -> ()
      }, {
      }) : () -> !jsir.any
      %call = "jsir.call_expression"(%arrow) : (!jsir.any) -> !jsir.any
      %e_pattern = "jsir.assignment_pattern_ref"(%e, %call) : (!jsir.any, !jsir.any) -> !jsir.any
      "jsir.exprs_region_end"(%a, %b_pattern, %object_pattern, %e_pattern) : (!jsir.any, !jsir.any, !jsir.any, !jsir.any) -> ()
    }, {
      "jshir.block_statement"() ({
        "jsir.function_declaration"() <{async = false, generator = false, id = #jsir<identifier #locH, "h">}> ({
          "jsir.exprs_region_end"() : () -> ()
        }, {
          "jshir.block_statement"() ({
            %h0 = "jsir.identifier"() <{name = "e"}> : () -> !jsir.any
            "jsir.return_statement"(%h0) : (!jsir.any) -> ()
          }, {
          ^bb0:
          }) : () -> ()
        }) : () -> ()
      }, {
      ^bb0:
      }) : () -> ()
    }) : () -> ()
  }, {
  ^bb0:
  }) : () -> ()
}
)mlir";

class JsirScopeRegistryParamsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mlir::DialectRegistry registry;
    registry.insert<JsirDialect, JshirDialect>();
    context_.appendDialectRegistry(registry);
    context_.loadAllAvailableDialects();
    module_ = mlir::parseSourceString<mlir::ModuleOp>(kParamsInput, &context_);
    ASSERT_TRUE(module_);
    module_->walk([&](JsirFunctionDeclarationOp op) {
      if (op.getIdAttr().getName().getValue() == "f") {
        f_ = op;
      } else {
        h_ = op;
      }
    });
    ASSERT_NE(f_, nullptr);
    ASSERT_NE(h_, nullptr);
  }

  // The identifiers named `name` in the module, in pre-order. `ref` selects
  // `jsir.identifier_ref` (true) or `jsir.identifier` (false).
  std::vector<mlir::Operation*> FindAll(llvm::StringRef name, bool ref) {
    std::vector<mlir::Operation*> result;
    module_->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* op) {
      if (ref ? llvm::isa<JsirIdentifierRefOp>(op) &&
                    llvm::cast<JsirIdentifierRefOp>(op).getName() == name
              : llvm::isa<JsirIdentifierOp>(op) &&
                    llvm::cast<JsirIdentifierOp>(op).getName() == name) {
        result.push_back(op);
      }
    });
    return result;
  }

  mlir::MLIRContext context_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  mlir::Operation* f_ = nullptr;
  mlir::Operation* h_ = nullptr;
};

TEST_F(JsirScopeRegistryParamsTest, PatternTargetsAreParams) {
  JsirScopeRegistry registry(*module_);
  for (llvm::StringRef name : {"a", "b", "x", "z", "r", "e"}) {
    std::vector<mlir::Operation*> refs = FindAll(name, /*ref=*/true);
    ASSERT_FALSE(refs.empty()) << name.str();
    std::optional<JsirSymbol> symbol = registry.Resolve(refs.front());
    ASSERT_TRUE(symbol.has_value()) << name.str();
    EXPECT_EQ(symbol->kind, JsirSymbolKind::kParam) << name.str();
    EXPECT_EQ(symbol->scope, f_) << name.str();
  }
}

TEST_F(JsirScopeRegistryParamsTest, DefaultValuesAreNotParams) {
  JsirScopeRegistry registry(*module_);
  // `c` is assigned in a default value, and `d` is read in default values:
  // neither is a parameter of `f`, so both are globals.
  std::vector<mlir::Operation*> c = FindAll("c", /*ref=*/true);
  ASSERT_EQ(c.size(), 1);
  EXPECT_EQ(registry.Resolve(c.front()), std::nullopt);
  for (mlir::Operation* d : FindAll("d", /*ref=*/false)) {
    EXPECT_EQ(registry.Resolve(d), std::nullopt);
  }
}

TEST_F(JsirScopeRegistryParamsTest, WriteInDefaultValueIsAUse) {
  JsirScopeRegistry registry(*module_);
  registry.AddScope(f_);
  // The binding of `a`, then the write `a = d` in the arrow function in `e`'s
  // default value.
  std::vector<mlir::Operation*> a = FindAll("a", /*ref=*/true);
  ASSERT_EQ(a.size(), 2);
  std::optional<JsirSymbol> symbol = registry.Resolve(a[1]);
  ASSERT_TRUE(symbol.has_value());
  EXPECT_EQ(symbol->kind, JsirSymbolKind::kParam);
  EXPECT_EQ(symbol->scope, f_);

  EXPECT_THAT(registry.GetUses(*symbol), ::testing::ElementsAre(a[1]));
  // The write runs in the arrow function, not in `f`.
  EXPECT_TRUE(registry.IsCaptured(*symbol));
}

TEST_F(JsirScopeRegistryParamsTest, OuterParamsAreVisibleAcrossTables) {
  JsirScopeRegistry registry(*module_);
  registry.AddScope(f_);
  registry.AddScope(h_);
  // `e` in `h` is `f`'s parameter, although `h` has a table of its own.
  std::vector<mlir::Operation*> e = FindAll("e", /*ref=*/false);
  ASSERT_EQ(e.size(), 1);
  std::optional<JsirSymbol> symbol = registry.Resolve(e.front());
  ASSERT_TRUE(symbol.has_value());
  EXPECT_EQ(symbol->kind, JsirSymbolKind::kParam);
  EXPECT_EQ(symbol->scope, f_);
}

}  // namespace
}  // namespace maldoca
