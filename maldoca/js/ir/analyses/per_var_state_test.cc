#include "maldoca/js/ir/analyses/per_var_state.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "gtest/gtest.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/ir/analyses/dataflow_analysis.h"

namespace maldoca {

TEST(PerVarStateTest, EraseDefaultOnSet) {
  auto exec_true = JsirExecutable(true);
  auto exec_false = JsirExecutable(false);

  JsSymbolId my_var("my_var", 0);

  auto true_default = JsirPerVarState<JsirExecutable>(exec_true);
  EXPECT_TRUE(true_default.empty());

  // Set to default value
  auto result1 = true_default.Set(my_var, exec_true);
  EXPECT_EQ(result1, mlir::ChangeResult::NoChange);
  EXPECT_TRUE(true_default.empty());
  EXPECT_EQ(true_default.Get(my_var), exec_true);

  // Set to non-default value
  auto result2 = true_default.Set(my_var, exec_false);
  EXPECT_EQ(result2, mlir::ChangeResult::Change);
  EXPECT_FALSE(true_default.empty());
  EXPECT_EQ(true_default.Get(my_var), exec_false);

  // Set to default value again
  auto result3 = true_default.Set(my_var, exec_true);
  EXPECT_EQ(result3, mlir::ChangeResult::Change);
  EXPECT_TRUE(true_default.empty());
  EXPECT_EQ(true_default.Get(my_var), exec_true);
}

TEST(PerVarStateTest, EraseDefaultOnJoin) {
  auto exec_true = JsirExecutable(true);
  auto exec_false = JsirExecutable(false);

  JsSymbolId my_var("my_var", 0);

  auto true_default_1 = JsirPerVarState<JsirExecutable>(exec_true);
  EXPECT_TRUE(true_default_1.empty());

  auto true_default_2 = JsirPerVarState<JsirExecutable>(exec_true);
  EXPECT_TRUE(true_default_2.empty());

  // Set my_var to false in true_default_1
  auto result_set = true_default_1.Set(my_var, exec_false);
  EXPECT_EQ(result_set, mlir::ChangeResult::Change);
  EXPECT_FALSE(true_default_1.empty());
  EXPECT_EQ(true_default_1.Get(my_var), exec_false);

  // Join so that my_var becomes true in true_default_1
  auto result_join = true_default_1.Join(true_default_2);
  EXPECT_EQ(result_join, mlir::ChangeResult::Change);
  EXPECT_TRUE(true_default_1.empty());
  EXPECT_EQ(true_default_1.Get(my_var), exec_true);
}

TEST(PerVarStateTest, EraseDefaultOnJoinDifferent) {
  auto exec_true = JsirExecutable(true);
  auto exec_false = JsirExecutable(false);

  JsSymbolId my_var("my_var", 0);

  auto true_default = JsirPerVarState<JsirExecutable>(exec_true);
  EXPECT_TRUE(true_default.empty());

  auto false_default = JsirPerVarState<JsirExecutable>(exec_false);
  EXPECT_TRUE(false_default.empty());

  // Set my_var to true in false_default
  auto result_set = false_default.Set(my_var, exec_true);
  EXPECT_EQ(result_set, mlir::ChangeResult::Change);
  EXPECT_FALSE(false_default.empty());
  EXPECT_EQ(false_default.Get(my_var), exec_true);

  // Join and ensure my_var is not added to true_default
  auto result_join = true_default.Join(false_default);
  EXPECT_EQ(result_join, mlir::ChangeResult::NoChange);
  EXPECT_TRUE(true_default.empty());
  EXPECT_EQ(true_default.Get(my_var), exec_true);
}

}  // namespace maldoca
