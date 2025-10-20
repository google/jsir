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

#ifndef MALDOCA_JS_AST_AST_WALKER_H_
#define MALDOCA_JS_AST_AST_WALKER_H_

#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "absl/log/log.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/ast/ast_visitor.h"

namespace maldoca {

class JsAstWalker : public JsAstVisitor<void> {
 public:
  explicit JsAstWalker(JsAstVisitor<void> *preorder_callback,
                       JsAstVisitor<void> *postorder_callback)
      : preorder_callback_(preorder_callback),
        postorder_callback_(postorder_callback) {}

  void VisitInterpreterDirective(
      const JsInterpreterDirective &interpreter_directive) override {
    if (preorder_callback_) {
      preorder_callback_->VisitInterpreterDirective(interpreter_directive);
    }
    if (postorder_callback_) {
      postorder_callback_->VisitInterpreterDirective(interpreter_directive);
    }
  }

  void VisitDirectiveLiteral(
      const JsDirectiveLiteral &directive_literal) override {
    if (preorder_callback_) {
      preorder_callback_->VisitDirectiveLiteral(directive_literal);
    }
    if (postorder_callback_) {
      postorder_callback_->VisitDirectiveLiteral(directive_literal);
    }
  }

  void VisitDirective(const JsDirective &directive) override {
    if (preorder_callback_) {
      preorder_callback_->VisitDirective(directive);
    }
    VisitDirectiveLiteral(*directive.value());
    if (postorder_callback_) {
      postorder_callback_->VisitDirective(directive);
    }
  }

  void VisitProgram(const JsProgram &program) override {
    if (preorder_callback_) {
      preorder_callback_->VisitProgram(program);
    }

    if (program.interpreter().has_value()) {
      VisitInterpreterDirective(*program.interpreter().value());
    }

    for (const std::unique_ptr<JsProgramBodyElement>& body_element :
         *program.body()) {
      VisitProgramBodyElement(*body_element);
    }

    for (const std::unique_ptr<JsDirective> &directive :
         *program.directives()) {
      VisitDirective(*directive);
    }

    if (postorder_callback_) {
      postorder_callback_->VisitProgram(program);
    }
  }

  void VisitFile(const JsFile &file) override {
    if (preorder_callback_) {
      preorder_callback_->VisitFile(file);
    }
    VisitProgram(*file.program());
    if (postorder_callback_) {
      postorder_callback_->VisitFile(file);
    }
  }

  void VisitIdentifier(const JsIdentifier &identifier) override {
    if (preorder_callback_) {
      preorder_callback_->VisitIdentifier(identifier);
    }
    if (postorder_callback_) {
      postorder_callback_->VisitIdentifier(identifier);
    }
  }

  void VisitPrivateName(const JsPrivateName &private_name) override {
    if (preorder_callback_) {
      preorder_callback_->VisitPrivateName(private_name);
    }
    VisitIdentifier(*private_name.id());
    if (postorder_callback_) {
      postorder_callback_->VisitPrivateName(private_name);
    }
  }

  void VisitRegExpLiteral(const JsRegExpLiteral &reg_exp_literal) override {
    if (preorder_callback_) {
      preorder_callback_->VisitRegExpLiteral(reg_exp_literal);
    }
    if (postorder_callback_) {
      postorder_callback_->VisitRegExpLiteral(reg_exp_literal);
    }
  }

  void VisitNullLiteral(const JsNullLiteral &null_literal) override {
    if (preorder_callback_) {
      preorder_callback_->VisitNullLiteral(null_literal);
    }
    if (postorder_callback_) {
      postorder_callback_->VisitNullLiteral(null_literal);
    }
  }

  void VisitStringLiteral(const JsStringLiteral &string_literal) override {
    if (preorder_callback_) {
      preorder_callback_->VisitStringLiteral(string_literal);
    }
    if (postorder_callback_) {
      postorder_callback_->VisitStringLiteral(string_literal);
    }
  }

  void VisitBooleanLiteral(const JsBooleanLiteral &boolean_literal) override {
    if (preorder_callback_) {
      preorder_callback_->VisitBooleanLiteral(boolean_literal);
    }
    if (postorder_callback_) {
      postorder_callback_->VisitBooleanLiteral(boolean_literal);
    }
  }

  void VisitNumericLiteral(const JsNumericLiteral &numeric_literal) override {
    if (preorder_callback_) {
      preorder_callback_->VisitNumericLiteral(numeric_literal);
    }
    if (postorder_callback_) {
      postorder_callback_->VisitNumericLiteral(numeric_literal);
    }
  }

  void VisitBigIntLiteral(const JsBigIntLiteral &big_int_literal) override {
    if (preorder_callback_) {
      preorder_callback_->VisitBigIntLiteral(big_int_literal);
    }
    if (postorder_callback_) {
      postorder_callback_->VisitBigIntLiteral(big_int_literal);
    }
  }

  void VisitBlockStatement(const JsBlockStatement &block_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitBlockStatement(block_statement);
    }

    for (const std::unique_ptr<JsStatement> &statement :
         *block_statement.body()) {
      VisitStatement(*statement);
    }

    for (const std::unique_ptr<JsDirective> &directive :
         *block_statement.directives()) {
      VisitDirective(*directive);
    }

    if (postorder_callback_) {
      postorder_callback_->VisitBlockStatement(block_statement);
    }
  }

  void VisitExpressionStatement(
      const JsExpressionStatement &expression_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitExpressionStatement(expression_statement);
    }
    VisitExpression(*expression_statement.expression());
    if (postorder_callback_) {
      postorder_callback_->VisitExpressionStatement(expression_statement);
    }
  }

  void VisitEmptyStatement(const JsEmptyStatement &empty_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitEmptyStatement(empty_statement);
    }
    if (postorder_callback_) {
      postorder_callback_->VisitEmptyStatement(empty_statement);
    }
  }

  void VisitDebuggerStatement(
      const JsDebuggerStatement &debugger_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitDebuggerStatement(debugger_statement);
    }
    if (postorder_callback_) {
      postorder_callback_->VisitDebuggerStatement(debugger_statement);
    }
  }

  void VisitWithStatement(const JsWithStatement &with_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitWithStatement(with_statement);
    }
    VisitExpression(*with_statement.object());
    VisitStatement(*with_statement.body());
    if (postorder_callback_) {
      postorder_callback_->VisitWithStatement(with_statement);
    }
  }

  void VisitReturnStatement(
      const JsReturnStatement &return_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitReturnStatement(return_statement);
    }
    if (return_statement.argument().has_value()) {
      VisitExpression(*return_statement.argument().value());
    }
    if (postorder_callback_) {
      postorder_callback_->VisitReturnStatement(return_statement);
    }
  }

  void VisitLabeledStatement(
      const JsLabeledStatement &labeled_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitLabeledStatement(labeled_statement);
    }
    VisitIdentifier(*labeled_statement.label());
    VisitStatement(*labeled_statement.body());
    if (postorder_callback_) {
      postorder_callback_->VisitLabeledStatement(labeled_statement);
    }
  }

  void VisitBreakStatement(const JsBreakStatement &break_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitBreakStatement(break_statement);
    }
    if (break_statement.label().has_value()) {
      VisitIdentifier(*break_statement.label().value());
    }
    if (postorder_callback_) {
      postorder_callback_->VisitBreakStatement(break_statement);
    }
  }

  void VisitContinueStatement(
      const JsContinueStatement &continue_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitContinueStatement(continue_statement);
    }
    if (continue_statement.label().has_value()) {
      VisitIdentifier(*continue_statement.label().value());
    }
    if (postorder_callback_) {
      postorder_callback_->VisitContinueStatement(continue_statement);
    }
  }

  void VisitIfStatement(const JsIfStatement &if_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitIfStatement(if_statement);
    }
    VisitExpression(*if_statement.test());
    VisitStatement(*if_statement.consequent());
    if (if_statement.alternate().has_value()) {
      VisitStatement(*if_statement.alternate().value());
    }
    if (postorder_callback_) {
      postorder_callback_->VisitIfStatement(if_statement);
    }
  }

  void VisitSwitchCase(const JsSwitchCase &switch_case) override {
    if (preorder_callback_) {
      preorder_callback_->VisitSwitchCase(switch_case);
    }
    if (switch_case.test().has_value()) {
      VisitExpression(*switch_case.test().value());
    }
    for (const std::unique_ptr<JsStatement> &consequent :
         *switch_case.consequent()) {
      VisitStatement(*consequent);
    }
    if (postorder_callback_) {
      postorder_callback_->VisitSwitchCase(switch_case);
    }
  }

  void VisitSwitchStatement(
      const JsSwitchStatement &switch_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitSwitchStatement(switch_statement);
    }
    VisitExpression(*switch_statement.discriminant());
    for (const std::unique_ptr<JsSwitchCase> &switch_case :
         *switch_statement.cases()) {
      VisitSwitchCase(*switch_case);
    }
    if (postorder_callback_) {
      postorder_callback_->VisitSwitchStatement(switch_statement);
    }
  }

  void VisitThrowStatement(const JsThrowStatement &throw_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitThrowStatement(throw_statement);
    }
    VisitExpression(*throw_statement.argument());
    if (postorder_callback_) {
      postorder_callback_->VisitThrowStatement(throw_statement);
    }
  }

  void VisitCatchClause(const JsCatchClause &catch_clause) override {
    if (preorder_callback_) {
      preorder_callback_->VisitCatchClause(catch_clause);
    }

    if (catch_clause.param().has_value()) {
      VisitPattern(*catch_clause.param().value());
    }
    VisitBlockStatement(*catch_clause.body());

    if (postorder_callback_) {
      postorder_callback_->VisitCatchClause(catch_clause);
    }
  }

  void VisitTryStatement(const JsTryStatement &try_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitTryStatement(try_statement);
    }

    VisitBlockStatement(*try_statement.block());
    if (try_statement.handler().has_value()) {
      VisitCatchClause(*try_statement.handler().value());
    }
    if (try_statement.finalizer().has_value()) {
      VisitBlockStatement(*try_statement.finalizer().value());
    }

    if (postorder_callback_) {
      postorder_callback_->VisitTryStatement(try_statement);
    }
  }

  void VisitWhileStatement(const JsWhileStatement &while_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitWhileStatement(while_statement);
    }

    VisitExpression(*while_statement.test());
    VisitStatement(*while_statement.body());

    if (postorder_callback_) {
      postorder_callback_->VisitWhileStatement(while_statement);
    }
  }

  void VisitDoWhileStatement(
      const JsDoWhileStatement &do_while_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitDoWhileStatement(do_while_statement);
    }

    VisitStatement(*do_while_statement.body());
    VisitExpression(*do_while_statement.test());

    if (postorder_callback_) {
      postorder_callback_->VisitDoWhileStatement(do_while_statement);
    }
  }

  void VisitVariableDeclarator(
      const JsVariableDeclarator &variable_declarator) override {
    if (preorder_callback_) {
      preorder_callback_->VisitVariableDeclarator(variable_declarator);
    }

    VisitLVal(*variable_declarator.id());
    if (variable_declarator.init().has_value()) {
      VisitExpression(*variable_declarator.init().value());
    }

    if (postorder_callback_) {
      postorder_callback_->VisitVariableDeclarator(variable_declarator);
    }
  }

  void VisitVariableDeclaration(
      const JsVariableDeclaration &variable_declaration) override {
    if (preorder_callback_) {
      preorder_callback_->VisitVariableDeclaration(variable_declaration);
    }

    for (const std::unique_ptr<JsVariableDeclarator> &declarator :
         *variable_declaration.declarations()) {
      VisitVariableDeclarator(*declarator);
    }

    if (postorder_callback_) {
      postorder_callback_->VisitVariableDeclaration(variable_declaration);
    }
  }

  void VisitForStatement(const JsForStatement &for_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitForStatement(for_statement);
    }

    if (for_statement.init().has_value()) {
      std::variant<const JsVariableDeclaration *, const JsExpression *> init =
          for_statement.init().value();
      switch (init.index()) {
        case 0:
          VisitVariableDeclaration(*std::get<0>(init));
          break;
        case 1:
          VisitExpression(*std::get<1>(init));
          break;
        default:
          LOG(FATAL) << "Unreachable code.";
      }
    }

    if (for_statement.test().has_value()) {
      VisitExpression(*for_statement.test().value());
    }
    if (for_statement.update().has_value()) {
      VisitExpression(*for_statement.update().value());
    }
    VisitStatement(*for_statement.body());

    if (postorder_callback_) {
      postorder_callback_->VisitForStatement(for_statement);
    }
  }

  void VisitForInStatement(const JsForInStatement &for_in_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitForInStatement(for_in_statement);
    }

    switch (for_in_statement.left().index()) {
      case 0:
        VisitVariableDeclaration(*std::get<0>(for_in_statement.left()));
        break;
      case 1:
        VisitLVal(*std::get<1>(for_in_statement.left()));
        break;
      default:
        LOG(FATAL) << "Unreachable code.";
    }
    VisitExpression(*for_in_statement.right());
    VisitStatement(*for_in_statement.body());

    if (postorder_callback_) {
      postorder_callback_->VisitForInStatement(for_in_statement);
    }
  }

  void VisitForOfStatement(const JsForOfStatement &for_of_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitForOfStatement(for_of_statement);
    }

    switch (for_of_statement.left().index()) {
      case 0:
        VisitVariableDeclaration(*std::get<0>(for_of_statement.left()));
        break;
      case 1:
        VisitLVal(*std::get<1>(for_of_statement.left()));
        break;
      default:
        LOG(FATAL) << "Unreachable code.";
    }
    VisitExpression(*for_of_statement.right());
    VisitStatement(*for_of_statement.body());

    if (postorder_callback_) {
      postorder_callback_->VisitForOfStatement(for_of_statement);
    }
  }

  void VisitFunctionDeclaration(
      const JsFunctionDeclaration &function_declaration) override {
    if (preorder_callback_) {
      preorder_callback_->VisitFunctionDeclaration(function_declaration);
    }

    // This is an example of a leaf node calling functions in parent nodes.
    if (function_declaration.id().has_value()) {
      VisitIdentifier(*function_declaration.id().value());
    }
    for (const std::unique_ptr<JsPattern> &param :
         *function_declaration.params()) {
      VisitPattern(*param);
    }
    VisitBlockStatement(*function_declaration.body());

    if (postorder_callback_) {
      postorder_callback_->VisitFunctionDeclaration(function_declaration);
    }
  }

  void VisitSuper(const JsSuper &super) override {
    if (preorder_callback_) {
      preorder_callback_->VisitSuper(super);
    }

    if (postorder_callback_) {
      postorder_callback_->VisitSuper(super);
    }
  }

  void VisitImport(const JsImport &import) override {
    if (preorder_callback_) {
      preorder_callback_->VisitImport(import);
    }

    if (postorder_callback_) {
      postorder_callback_->VisitImport(import);
    }
  }

  void VisitThisExpression(const JsThisExpression &this_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitThisExpression(this_expression);
    }

    if (postorder_callback_) {
      postorder_callback_->VisitThisExpression(this_expression);
    }
  }

  void VisitArrowFunctionExpression(
      const JsArrowFunctionExpression &arrow_function_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitArrowFunctionExpression(
          arrow_function_expression);
    }

    if (arrow_function_expression.id().has_value()) {
      VisitIdentifier(*arrow_function_expression.id().value());
    }
    for (const std::unique_ptr<JsPattern> &param :
         *arrow_function_expression.params()) {
      VisitPattern(*param);
    }
    switch (arrow_function_expression.body().index()) {
      case 0:
        VisitBlockStatement(*std::get<0>(arrow_function_expression.body()));
        break;
      case 1:
        VisitExpression(*std::get<1>(arrow_function_expression.body()));
        break;
      default:
        LOG(FATAL) << "Unreachable code.";
    }

    if (postorder_callback_) {
      postorder_callback_->VisitArrowFunctionExpression(
          arrow_function_expression);
    }
  }

  void VisitYieldExpression(
      const JsYieldExpression &yield_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitYieldExpression(yield_expression);
    }

    if (yield_expression.argument().has_value()) {
      VisitExpression(*yield_expression.argument().value());
    }

    if (postorder_callback_) {
      postorder_callback_->VisitYieldExpression(yield_expression);
    }
  }

  void VisitAwaitExpression(
      const JsAwaitExpression &await_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitAwaitExpression(await_expression);
    }

    if (await_expression.argument().has_value()) {
      VisitExpression(*await_expression.argument().value());
    }

    if (postorder_callback_) {
      postorder_callback_->VisitAwaitExpression(await_expression);
    }
  }

  void VisitSpreadElement(const JsSpreadElement &spread_element) override {
    if (preorder_callback_) {
      preorder_callback_->VisitSpreadElement(spread_element);
    }

    VisitExpression(*spread_element.argument());

    if (postorder_callback_) {
      postorder_callback_->VisitSpreadElement(spread_element);
    }
  }

  void VisitArrayExpression(
      const JsArrayExpression &array_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitArrayExpression(array_expression);
    }

    for (const std::optional<std::variant<std::unique_ptr<JsExpression>,
                                          std::unique_ptr<JsSpreadElement>>>
             &element : *array_expression.elements()) {
      if (element.has_value()) {
        const std::variant<std::unique_ptr<JsExpression>,
                           std::unique_ptr<JsSpreadElement>> &element_variant =
            element.value();
        switch (element_variant.index()) {
          case 0:
            VisitExpression(*std::get<0>(element_variant));
            break;
          case 1:
            VisitSpreadElement(*std::get<1>(element_variant));
            break;
          default:
            LOG(FATAL) << "Unreachable code.";
        }
      }
    }

    if (postorder_callback_) {
      postorder_callback_->VisitArrayExpression(array_expression);
    }
  }

  void VisitObjectProperty(const JsObjectProperty &object_property) override {
    if (preorder_callback_) {
      preorder_callback_->VisitObjectProperty(object_property);
    }

    VisitExpression(*object_property.key());
    switch (object_property.value().index()) {
      case 0:
        VisitExpression(*std::get<0>(object_property.value()));
        break;
      case 1:
        VisitPattern(*std::get<1>(object_property.value()));
        break;
      default:
        LOG(FATAL) << "Unreachable code.";
    }

    if (postorder_callback_) {
      postorder_callback_->VisitObjectProperty(object_property);
    }
  }

  void VisitObjectMethod(const JsObjectMethod &object_method) override {
    if (preorder_callback_) {
      preorder_callback_->VisitObjectMethod(object_method);
    }

    VisitExpression(*object_method.key());
    if (object_method.id().has_value()) {
      VisitIdentifier(*object_method.id().value());
    }
    for (const std::unique_ptr<JsPattern> &param : *object_method.params()) {
      VisitPattern(*param);
    }
    VisitBlockStatement(*object_method.body());

    if (postorder_callback_) {
      postorder_callback_->VisitObjectMethod(object_method);
    }
  }

  void VisitObjectExpression(
      const JsObjectExpression &object_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitObjectExpression(object_expression);
    }

    for (const std::variant<std::unique_ptr<JsObjectProperty>,
                            std::unique_ptr<JsObjectMethod>,
                            std::unique_ptr<JsSpreadElement>> &property :
         *object_expression.properties_()) {
      switch (property.index()) {
        case 0:
          VisitObjectProperty(*std::get<0>(property));
          break;
        case 1:
          VisitObjectMethod(*std::get<1>(property));
          break;
        case 2:
          VisitSpreadElement(*std::get<2>(property));
          break;
        default:
          LOG(FATAL) << "Unreachable code.";
      }
    }

    if (postorder_callback_) {
      postorder_callback_->VisitObjectExpression(object_expression);
    }
  }

  void VisitFunctionExpression(
      const JsFunctionExpression &function_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitFunctionExpression(function_expression);
    }

    if (function_expression.id().has_value()) {
      VisitIdentifier(*function_expression.id().value());
    }
    for (const std::unique_ptr<JsPattern> &param :
         *function_expression.params()) {
      VisitPattern(*param);
    }
    VisitBlockStatement(*function_expression.body());

    if (postorder_callback_) {
      postorder_callback_->VisitFunctionExpression(function_expression);
    }
  }

  void VisitUnaryExpression(
      const JsUnaryExpression &unary_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitUnaryExpression(unary_expression);
    }

    VisitExpression(*unary_expression.argument());

    if (postorder_callback_) {
      postorder_callback_->VisitUnaryExpression(unary_expression);
    }
  }

  void VisitUpdateExpression(
      const JsUpdateExpression &update_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitUpdateExpression(update_expression);
    }

    VisitLVal(*update_expression.argument());

    if (postorder_callback_) {
      postorder_callback_->VisitUpdateExpression(update_expression);
    }
  }

  void VisitBinaryExpression(
      const JsBinaryExpression &binary_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitBinaryExpression(binary_expression);
    }

    switch (binary_expression.left().index()) {
      case 0:
        VisitExpression(*std::get<0>(binary_expression.left()));
        break;
      case 1:
        VisitPrivateName(*std::get<1>(binary_expression.left()));
        break;
      default:
        LOG(FATAL) << "Unreachable code.";
    }
    VisitExpression(*binary_expression.right());

    if (postorder_callback_) {
      postorder_callback_->VisitBinaryExpression(binary_expression);
    }
  }

  void VisitAssignmentExpression(
      const JsAssignmentExpression &assignment_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitAssignmentExpression(assignment_expression);
    }

    VisitLVal(*assignment_expression.left());
    VisitExpression(*assignment_expression.right());

    if (postorder_callback_) {
      postorder_callback_->VisitAssignmentExpression(assignment_expression);
    }
  }

  void VisitLogicalExpression(
      const JsLogicalExpression &logical_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitLogicalExpression(logical_expression);
    }

    VisitExpression(*logical_expression.left());
    VisitExpression(*logical_expression.right());

    if (postorder_callback_) {
      postorder_callback_->VisitLogicalExpression(logical_expression);
    }
  }

  void VisitMemberExpression(
      const JsMemberExpression &member_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitMemberExpression(member_expression);
    }

    switch (member_expression.object().index()) {
      case 0:
        VisitExpression(*std::get<0>(member_expression.object()));
        break;
      case 1:
        VisitSuper(*std::get<1>(member_expression.object()));
        break;
      default:
        LOG(FATAL) << "Unreachable code.";
    }
    switch (member_expression.property().index()) {
      case 0:
        VisitExpression(*std::get<0>(member_expression.property()));
        break;
      case 1:
        VisitPrivateName(*std::get<1>(member_expression.property()));
        break;
      default:
        LOG(FATAL) << "Unreachable code.";
    }

    if (postorder_callback_) {
      postorder_callback_->VisitMemberExpression(member_expression);
    }
  }

  void VisitOptionalMemberExpression(
      const JsOptionalMemberExpression &optional_member_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitOptionalMemberExpression(
          optional_member_expression);
    }

    VisitExpression(*optional_member_expression.object());
    switch (optional_member_expression.property().index()) {
      case 0:
        VisitExpression(*std::get<0>(optional_member_expression.property()));
        break;
      case 1:
        VisitPrivateName(*std::get<1>(optional_member_expression.property()));
        break;
      default:
        LOG(FATAL) << "Unreachable code.";
    }

    if (postorder_callback_) {
      postorder_callback_->VisitOptionalMemberExpression(
          optional_member_expression);
    }
  }

  void VisitConditionalExpression(
      const JsConditionalExpression &conditional_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitConditionalExpression(conditional_expression);
    }

    VisitExpression(*conditional_expression.test());
    VisitExpression(*conditional_expression.alternate());
    VisitExpression(*conditional_expression.consequent());

    if (postorder_callback_) {
      postorder_callback_->VisitConditionalExpression(conditional_expression);
    }
  }

  void VisitCallExpression(const JsCallExpression &call_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitCallExpression(call_expression);
    }

    switch (call_expression.callee().index()) {
      case 0:
        VisitExpression(*std::get<0>(call_expression.callee()));
        break;
      case 1:
        VisitSuper(*std::get<1>(call_expression.callee()));
        break;
      case 2:
        VisitImport(*std::get<2>(call_expression.callee()));
        break;
      default:
        LOG(FATAL) << "Unreachable code.";
    }
    for (const std::variant<std::unique_ptr<JsExpression>,
                            std::unique_ptr<JsSpreadElement>> &argument :
         *call_expression.arguments()) {
      switch (argument.index()) {
        case 0:
          VisitExpression(*std::get<0>(argument));
          break;
        case 1:
          VisitSpreadElement(*std::get<1>(argument));
          break;
        default:
          LOG(FATAL) << "Unreachable code.";
      }
    }

    if (postorder_callback_) {
      postorder_callback_->VisitCallExpression(call_expression);
    }
  }

  void VisitOptionalCallExpression(
      const JsOptionalCallExpression &optional_call_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitOptionalCallExpression(optional_call_expression);
    }

    VisitExpression(*optional_call_expression.callee());
    for (const std::variant<std::unique_ptr<JsExpression>,
                            std::unique_ptr<JsSpreadElement>> &argument :
         *optional_call_expression.arguments()) {
      switch (argument.index()) {
        case 0:
          VisitExpression(*std::get<0>(argument));
          break;
        case 1:
          VisitSpreadElement(*std::get<1>(argument));
          break;
        default:
          LOG(FATAL) << "Unreachable code.";
      }
    }

    if (postorder_callback_) {
      postorder_callback_->VisitOptionalCallExpression(
          optional_call_expression);
    }
  }

  void VisitNewExpression(const JsNewExpression &new_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitNewExpression(new_expression);
    }

    switch (new_expression.callee().index()) {
      case 0:
        VisitExpression(*std::get<0>(new_expression.callee()));
        break;
      case 1:
        VisitSuper(*std::get<1>(new_expression.callee()));
        break;
      case 2:
        VisitImport(*std::get<2>(new_expression.callee()));
        break;
      default:
        LOG(FATAL) << "Unreachable code.";
    }
    for (const std::variant<std::unique_ptr<JsExpression>,
                            std::unique_ptr<JsSpreadElement>> &argument :
         *new_expression.arguments()) {
      switch (argument.index()) {
        case 0:
          VisitExpression(*std::get<0>(argument));
          break;
        case 1:
          VisitSpreadElement(*std::get<1>(argument));
          break;
        default:
          LOG(FATAL) << "Unreachable code.";
      }
    }

    if (postorder_callback_) {
      postorder_callback_->VisitNewExpression(new_expression);
    }
  }

  void VisitSequenceExpression(
      const JsSequenceExpression &sequence_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitSequenceExpression(sequence_expression);
    }

    for (const std::unique_ptr<JsExpression> &expression :
         *sequence_expression.expressions()) {
      VisitExpression(*expression);
    }

    if (postorder_callback_) {
      postorder_callback_->VisitSequenceExpression(sequence_expression);
    }
  }

  void VisitParenthesizedExpression(
      const JsParenthesizedExpression &parenthesized_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitParenthesizedExpression(
          parenthesized_expression);
    }

    VisitExpression(*parenthesized_expression.expression());

    if (postorder_callback_) {
      postorder_callback_->VisitParenthesizedExpression(
          parenthesized_expression);
    }
  }

  void VisitTemplateElement(
      const JsTemplateElement &template_element) override {
    if (preorder_callback_) {
      preorder_callback_->VisitTemplateElement(template_element);
    }

    if (postorder_callback_) {
      postorder_callback_->VisitTemplateElement(template_element);
    }
  }

  void VisitTemplateLiteral(
      const JsTemplateLiteral &template_literal) override {
    if (preorder_callback_) {
      preorder_callback_->VisitTemplateLiteral(template_literal);
    }

    for (const std::unique_ptr<JsTemplateElement> &quasi :
         *template_literal.quasis()) {
      VisitTemplateElement(*quasi);
    }
    for (const std::unique_ptr<JsExpression> &expression :
         *template_literal.expressions()) {
      VisitExpression(*expression);
    }

    if (postorder_callback_) {
      postorder_callback_->VisitTemplateLiteral(template_literal);
    }
  }

  void VisitTaggedTemplateExpression(
      const JsTaggedTemplateExpression &tagged_template_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitTaggedTemplateExpression(
          tagged_template_expression);
    }

    VisitExpression(*tagged_template_expression.tag());
    VisitTemplateLiteral(*tagged_template_expression.quasi());

    if (postorder_callback_) {
      postorder_callback_->VisitTaggedTemplateExpression(
          tagged_template_expression);
    }
  }

  void VisitRestElement(const JsRestElement &rest_element) override {
    if (preorder_callback_) {
      preorder_callback_->VisitRestElement(rest_element);
    }

    VisitLVal(*rest_element.argument());

    if (postorder_callback_) {
      postorder_callback_->VisitRestElement(rest_element);
    }
  }

  void VisitObjectPattern(const JsObjectPattern &object_pattern) override {
    if (preorder_callback_) {
      preorder_callback_->VisitObjectPattern(object_pattern);
    }

    for (const std::variant<std::unique_ptr<JsObjectProperty>,
                            std::unique_ptr<JsRestElement>> &property :
         *object_pattern.properties_()) {
      switch (property.index()) {
        case 0:
          VisitObjectProperty(*std::get<0>(property));
          break;
        case 1:
          VisitRestElement(*std::get<1>(property));
          break;
        default:
          LOG(FATAL) << "Unreachable code.";
      }
    }

    if (postorder_callback_) {
      postorder_callback_->VisitObjectPattern(object_pattern);
    }
  }

  void VisitArrayPattern(const JsArrayPattern &array_pattern) override {
    if (preorder_callback_) {
      preorder_callback_->VisitArrayPattern(array_pattern);
    }

    for (const std::optional<std::unique_ptr<JsPattern>> &element :
         *array_pattern.elements()) {
      if (element.has_value()) {
        VisitPattern(*element.value());
      }
    }

    if (postorder_callback_) {
      postorder_callback_->VisitArrayPattern(array_pattern);
    }
  }

  void VisitAssignmentPattern(
      const JsAssignmentPattern &assignment_pattern) override {
    if (preorder_callback_) {
      preorder_callback_->VisitAssignmentPattern(assignment_pattern);
    }

    VisitPattern(*assignment_pattern.left());
    VisitExpression(*assignment_pattern.right());

    if (postorder_callback_) {
      postorder_callback_->VisitAssignmentPattern(assignment_pattern);
    }
  }

  void VisitClassMethod(const JsClassMethod &class_method) override {
    if (preorder_callback_) {
      preorder_callback_->VisitClassMethod(class_method);
    }

    VisitExpression(*class_method.key());
    if (class_method.id().has_value()) {
      VisitIdentifier(*class_method.id().value());
    }
    for (const std::unique_ptr<JsPattern> &param : *class_method.params()) {
      VisitPattern(*param);
    }
    VisitBlockStatement(*class_method.body());

    if (postorder_callback_) {
      postorder_callback_->VisitClassMethod(class_method);
    }
  }

  void VisitClassPrivateMethod(
      const JsClassPrivateMethod &class_private_method) override {
    if (preorder_callback_) {
      preorder_callback_->VisitClassPrivateMethod(class_private_method);
    }

    VisitPrivateName(*class_private_method.key());
    if (class_private_method.id().has_value()) {
      VisitIdentifier(*class_private_method.id().value());
    }
    for (const std::unique_ptr<JsPattern> &param :
         *class_private_method.params()) {
      VisitPattern(*param);
    }
    VisitBlockStatement(*class_private_method.body());

    if (postorder_callback_) {
      postorder_callback_->VisitClassPrivateMethod(class_private_method);
    }
  }

  void VisitClassProperty(const JsClassProperty &class_property) override {
    if (preorder_callback_) {
      preorder_callback_->VisitClassProperty(class_property);
    }

    VisitExpression(*class_property.key());
    if (class_property.value().has_value()) {
      VisitExpression(*class_property.value().value());
    }

    if (postorder_callback_) {
      postorder_callback_->VisitClassProperty(class_property);
    }
  }

  void VisitClassPrivateProperty(
      const JsClassPrivateProperty &class_private_property) override {
    if (preorder_callback_) {
      preorder_callback_->VisitClassPrivateProperty(class_private_property);
    }

    VisitPrivateName(*class_private_property.key());
    if (class_private_property.value().has_value()) {
      VisitExpression(*class_private_property.value().value());
    }

    if (postorder_callback_) {
      postorder_callback_->VisitClassPrivateProperty(class_private_property);
    }
  }

  void VisitClassBody(const JsClassBody &class_body) override {
    if (preorder_callback_) {
      preorder_callback_->VisitClassBody(class_body);
    }

    for (const std::variant<std::unique_ptr<JsClassMethod>,
                            std::unique_ptr<JsClassPrivateMethod>,
                            std::unique_ptr<JsClassProperty>,
                            std::unique_ptr<JsClassPrivateProperty>>
             &body_element : *class_body.body()) {
      switch (body_element.index()) {
        case 0:
          VisitClassMethod(*std::get<0>(body_element));
          break;
        case 1:
          VisitClassPrivateMethod(*std::get<1>(body_element));
          break;
        case 2:
          VisitClassProperty(*std::get<2>(body_element));
          break;
        case 3:
          VisitClassPrivateProperty(*std::get<3>(body_element));
          break;
        default:
          LOG(FATAL) << "Unreachable code.";
      }
    }

    if (postorder_callback_) {
      postorder_callback_->VisitClassBody(class_body);
    }
  }

  void VisitClassDeclaration(
      const JsClassDeclaration &class_declaration) override {
    if (preorder_callback_) {
      preorder_callback_->VisitClassDeclaration(class_declaration);
    }

    if (class_declaration.id().has_value()) {
      VisitIdentifier(*class_declaration.id().value());
    }
    if (class_declaration.super_class().has_value()) {
      VisitExpression(*class_declaration.super_class().value());
    }
    VisitClassBody(*class_declaration.body());

    if (postorder_callback_) {
      postorder_callback_->VisitClassDeclaration(class_declaration);
    }
  }

  void VisitClassExpression(
      const JsClassExpression &class_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitClassExpression(class_expression);
    }

    if (class_expression.id().has_value()) {
      VisitIdentifier(*class_expression.id().value());
    }
    if (class_expression.super_class().has_value()) {
      VisitExpression(*class_expression.super_class().value());
    }
    VisitClassBody(*class_expression.body());

    if (postorder_callback_) {
      postorder_callback_->VisitClassExpression(class_expression);
    }
  }

  void VisitMetaProperty(const JsMetaProperty &meta_property) override {
    if (preorder_callback_) {
      preorder_callback_->VisitMetaProperty(meta_property);
    }

    VisitIdentifier(*meta_property.meta());
    VisitIdentifier(*meta_property.property());

    if (postorder_callback_) {
      postorder_callback_->VisitMetaProperty(meta_property);
    }
  }

  void VisitImportSpecifier(
      const JsImportSpecifier &import_specifier) override {
    if (preorder_callback_) {
      preorder_callback_->VisitImportSpecifier(import_specifier);
    }

    switch (import_specifier.imported().index()) {
      case 0:
        VisitIdentifier(*std::get<0>(import_specifier.imported()));
        break;
      case 1:
        VisitStringLiteral(*std::get<1>(import_specifier.imported()));
        break;
      default:
        LOG(FATAL) << "Unreachable code.";
    }
    VisitIdentifier(*import_specifier.local());

    if (postorder_callback_) {
      postorder_callback_->VisitImportSpecifier(import_specifier);
    }
  }

  void VisitImportDefaultSpecifier(
      const JsImportDefaultSpecifier &import_default_specifier) override {
    if (preorder_callback_) {
      preorder_callback_->VisitImportDefaultSpecifier(import_default_specifier);
    }

    VisitIdentifier(*import_default_specifier.local());

    if (postorder_callback_) {
      postorder_callback_->VisitImportDefaultSpecifier(
          import_default_specifier);
    }
  }

  void VisitImportNamespaceSpecifier(
      const JsImportNamespaceSpecifier &import_namespace_specifier) override {
    if (preorder_callback_) {
      preorder_callback_->VisitImportNamespaceSpecifier(
          import_namespace_specifier);
    }

    VisitIdentifier(*import_namespace_specifier.local());

    if (postorder_callback_) {
      postorder_callback_->VisitImportNamespaceSpecifier(
          import_namespace_specifier);
    }
  }

  void VisitImportAttribute(
      const JsImportAttribute &import_attribute) override {
    if (preorder_callback_) {
      preorder_callback_->VisitImportAttribute(import_attribute);
    }

    VisitIdentifier(*import_attribute.key());
    VisitStringLiteral(*import_attribute.value());

    if (postorder_callback_) {
      postorder_callback_->VisitImportAttribute(import_attribute);
    }
  }

  void VisitImportDeclaration(
      const JsImportDeclaration &import_declaration) override {
    if (preorder_callback_) {
      preorder_callback_->VisitImportDeclaration(import_declaration);
    }

    for (const std::variant<std::unique_ptr<JsImportSpecifier>,
                            std::unique_ptr<JsImportDefaultSpecifier>,
                            std::unique_ptr<JsImportNamespaceSpecifier>>
             &specifier : *import_declaration.specifiers()) {
      switch (specifier.index()) {
        case 0:
          VisitImportSpecifier(*std::get<0>(specifier));
          break;
        case 1:
          VisitImportDefaultSpecifier(*std::get<1>(specifier));
          break;
        case 2:
          VisitImportNamespaceSpecifier(*std::get<2>(specifier));
          break;
        default:
          LOG(FATAL) << "Unreachable code.";
      }
    }
    VisitStringLiteral(*import_declaration.source());
    if (import_declaration.assertions().has_value()) {
      VisitImportAttribute(*import_declaration.assertions().value());
    }

    if (postorder_callback_) {
      postorder_callback_->VisitImportDeclaration(import_declaration);
    }
  }

  void VisitExportSpecifier(
      const JsExportSpecifier &export_specifier) override {
    if (preorder_callback_) {
      preorder_callback_->VisitExportSpecifier(export_specifier);
    }

    switch (export_specifier.exported().index()) {
      case 0:
        VisitIdentifier(*std::get<0>(export_specifier.exported()));
        break;
      case 1:
        VisitStringLiteral(*std::get<1>(export_specifier.exported()));
        break;
      default:
        LOG(FATAL) << "Unreachable code.";
    }
    if (export_specifier.local().has_value()) {
      switch (export_specifier.local().value().index()) {
        case 0:
          VisitIdentifier(*std::get<0>(export_specifier.local().value()));
          break;
        case 1:
          VisitStringLiteral(*std::get<1>(export_specifier.local().value()));
          break;
        default:
          LOG(FATAL) << "Unreachable code.";
      }
    }

    if (postorder_callback_) {
      postorder_callback_->VisitExportSpecifier(export_specifier);
    }
  }

  void VisitExportNamedDeclaration(
      const JsExportNamedDeclaration &export_named_declaration) override {
    if (preorder_callback_) {
      preorder_callback_->VisitExportNamedDeclaration(export_named_declaration);
    }

    if (export_named_declaration.declaration().has_value()) {
      VisitDeclaration(*export_named_declaration.declaration().value());
    }
    for (const std::unique_ptr<JsExportSpecifier> &specifier :
         *export_named_declaration.specifiers()) {
      VisitExportSpecifier(*specifier);
    }
    if (export_named_declaration.source().has_value()) {
      VisitStringLiteral(*export_named_declaration.source().value());
    }
    if (export_named_declaration.assertions().has_value()) {
      const std::vector<std::unique_ptr<JsImportAttribute>> *assertions =
          export_named_declaration.assertions().value();
      for (const std::unique_ptr<JsImportAttribute> &assertion : *assertions) {
        VisitImportAttribute(*assertion);
      }
    }

    if (postorder_callback_) {
      postorder_callback_->VisitExportNamedDeclaration(
          export_named_declaration);
    }
  }

  void VisitExportDefaultDeclaration(
      const JsExportDefaultDeclaration &export_default_declaration) override {
    if (preorder_callback_) {
      preorder_callback_->VisitExportDefaultDeclaration(
          export_default_declaration);
    }

    switch (export_default_declaration.declaration().index()) {
      case 0:
        VisitFunctionDeclaration(
            *std::get<0>(export_default_declaration.declaration()));
        break;
      case 1:
        VisitClassDeclaration(
            *std::get<1>(export_default_declaration.declaration()));
        break;
      case 2:
        VisitExpression(
            *std::get<2>(export_default_declaration.declaration()));
        break;
      default:
        LOG(FATAL) << "Unreachable code.";
    }

    if (postorder_callback_) {
      postorder_callback_->VisitExportDefaultDeclaration(
          export_default_declaration);
    }
  }

  void VisitExportAllDeclaration(
      const JsExportAllDeclaration &export_all_declaration) override {
    if (preorder_callback_) {
      preorder_callback_->VisitExportAllDeclaration(export_all_declaration);
    }

    VisitStringLiteral(*export_all_declaration.source());
    if (export_all_declaration.assertions().has_value()) {
      const std::vector<std::unique_ptr<JsImportAttribute>> *assertions =
          export_all_declaration.assertions().value();
      for (const std::unique_ptr<JsImportAttribute> &assertion : *assertions) {
        VisitImportAttribute(*assertion);
      }
    }

    if (postorder_callback_) {
      postorder_callback_->VisitExportAllDeclaration(export_all_declaration);
    }
  }

 private:
  JsAstVisitor<void> *preorder_callback_;
  JsAstVisitor<void> *postorder_callback_;
};

class MutableJsAstWalker : public MutableJsAstVisitor<void> {
 public:
  explicit MutableJsAstWalker(MutableJsAstVisitor<void> *preorder_callback,
                              MutableJsAstVisitor<void> *postorder_callback)
      : preorder_callback_(preorder_callback),
        postorder_callback_(postorder_callback) {}

  void VisitInterpreterDirective(
      JsInterpreterDirective &interpreter_directive) override {
    if (preorder_callback_) {
      preorder_callback_->VisitInterpreterDirective(interpreter_directive);
    }
    if (postorder_callback_) {
      postorder_callback_->VisitInterpreterDirective(interpreter_directive);
    }
  }

  void VisitDirectiveLiteral(JsDirectiveLiteral &directive_literal) override {
    if (preorder_callback_) {
      preorder_callback_->VisitDirectiveLiteral(directive_literal);
    }
    if (postorder_callback_) {
      postorder_callback_->VisitDirectiveLiteral(directive_literal);
    }
  }

  void VisitDirective(JsDirective &directive) override {
    if (preorder_callback_) {
      preorder_callback_->VisitDirective(directive);
    }
    VisitDirectiveLiteral(*directive.value());
    if (postorder_callback_) {
      postorder_callback_->VisitDirective(directive);
    }
  }

  void VisitProgram(JsProgram &program) override {
    if (preorder_callback_) {
      preorder_callback_->VisitProgram(program);
    }

    if (program.interpreter().has_value()) {
      VisitInterpreterDirective(*program.interpreter().value());
    }

    for (std::unique_ptr<JsProgramBodyElement>& body_element :
         *program.body()) {
      VisitProgramBodyElement(*body_element);
    }

    for (std::unique_ptr<JsDirective> &directive : *program.directives()) {
      VisitDirective(*directive);
    }

    if (postorder_callback_) {
      postorder_callback_->VisitProgram(program);
    }
  }

  void VisitFile(JsFile &file) override {
    if (preorder_callback_) {
      preorder_callback_->VisitFile(file);
    }
    VisitProgram(*file.program());
    if (postorder_callback_) {
      postorder_callback_->VisitFile(file);
    }
  }

  void VisitIdentifier(JsIdentifier &identifier) override {
    if (preorder_callback_) {
      preorder_callback_->VisitIdentifier(identifier);
    }
    if (postorder_callback_) {
      postorder_callback_->VisitIdentifier(identifier);
    }
  }

  void VisitPrivateName(JsPrivateName &private_name) override {
    if (preorder_callback_) {
      preorder_callback_->VisitPrivateName(private_name);
    }
    VisitIdentifier(*private_name.id());
    if (postorder_callback_) {
      postorder_callback_->VisitPrivateName(private_name);
    }
  }

  void VisitRegExpLiteral(JsRegExpLiteral &reg_exp_literal) override {
    if (preorder_callback_) {
      preorder_callback_->VisitRegExpLiteral(reg_exp_literal);
    }
    if (postorder_callback_) {
      postorder_callback_->VisitRegExpLiteral(reg_exp_literal);
    }
  }

  void VisitNullLiteral(JsNullLiteral &null_literal) override {
    if (preorder_callback_) {
      preorder_callback_->VisitNullLiteral(null_literal);
    }
    if (postorder_callback_) {
      postorder_callback_->VisitNullLiteral(null_literal);
    }
  }

  void VisitStringLiteral(JsStringLiteral &string_literal) override {
    if (preorder_callback_) {
      preorder_callback_->VisitStringLiteral(string_literal);
    }
    if (postorder_callback_) {
      postorder_callback_->VisitStringLiteral(string_literal);
    }
  }

  void VisitBooleanLiteral(JsBooleanLiteral &boolean_literal) override {
    if (preorder_callback_) {
      preorder_callback_->VisitBooleanLiteral(boolean_literal);
    }
    if (postorder_callback_) {
      postorder_callback_->VisitBooleanLiteral(boolean_literal);
    }
  }

  void VisitNumericLiteral(JsNumericLiteral &numeric_literal) override {
    if (preorder_callback_) {
      preorder_callback_->VisitNumericLiteral(numeric_literal);
    }
    if (postorder_callback_) {
      postorder_callback_->VisitNumericLiteral(numeric_literal);
    }
  }

  void VisitBigIntLiteral(JsBigIntLiteral &big_int_literal) override {
    if (preorder_callback_) {
      preorder_callback_->VisitBigIntLiteral(big_int_literal);
    }
    if (postorder_callback_) {
      postorder_callback_->VisitBigIntLiteral(big_int_literal);
    }
  }

  void VisitBlockStatement(JsBlockStatement &block_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitBlockStatement(block_statement);
    }

    for (std::unique_ptr<JsStatement> &statement : *block_statement.body()) {
      VisitStatement(*statement);
    }

    for (std::unique_ptr<JsDirective> &directive :
         *block_statement.directives()) {
      VisitDirective(*directive);
    }

    if (postorder_callback_) {
      postorder_callback_->VisitBlockStatement(block_statement);
    }
  }

  void VisitExpressionStatement(
      JsExpressionStatement &expression_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitExpressionStatement(expression_statement);
    }
    VisitExpression(*expression_statement.expression());
    if (postorder_callback_) {
      postorder_callback_->VisitExpressionStatement(expression_statement);
    }
  }

  void VisitEmptyStatement(JsEmptyStatement &empty_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitEmptyStatement(empty_statement);
    }
    if (postorder_callback_) {
      postorder_callback_->VisitEmptyStatement(empty_statement);
    }
  }

  void VisitDebuggerStatement(
      JsDebuggerStatement &debugger_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitDebuggerStatement(debugger_statement);
    }
    if (postorder_callback_) {
      postorder_callback_->VisitDebuggerStatement(debugger_statement);
    }
  }

  void VisitWithStatement(JsWithStatement &with_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitWithStatement(with_statement);
    }
    VisitExpression(*with_statement.object());
    VisitStatement(*with_statement.body());
    if (postorder_callback_) {
      postorder_callback_->VisitWithStatement(with_statement);
    }
  }

  void VisitReturnStatement(JsReturnStatement &return_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitReturnStatement(return_statement);
    }
    if (return_statement.argument().has_value()) {
      VisitExpression(*return_statement.argument().value());
    }
    if (postorder_callback_) {
      postorder_callback_->VisitReturnStatement(return_statement);
    }
  }

  void VisitLabeledStatement(JsLabeledStatement &labeled_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitLabeledStatement(labeled_statement);
    }
    VisitIdentifier(*labeled_statement.label());
    VisitStatement(*labeled_statement.body());
    if (postorder_callback_) {
      postorder_callback_->VisitLabeledStatement(labeled_statement);
    }
  }

  void VisitBreakStatement(JsBreakStatement &break_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitBreakStatement(break_statement);
    }
    if (break_statement.label().has_value()) {
      VisitIdentifier(*break_statement.label().value());
    }
    if (postorder_callback_) {
      postorder_callback_->VisitBreakStatement(break_statement);
    }
  }

  void VisitContinueStatement(
      JsContinueStatement &continue_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitContinueStatement(continue_statement);
    }
    if (continue_statement.label().has_value()) {
      VisitIdentifier(*continue_statement.label().value());
    }
    if (postorder_callback_) {
      postorder_callback_->VisitContinueStatement(continue_statement);
    }
  }

  void VisitIfStatement(JsIfStatement &if_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitIfStatement(if_statement);
    }
    VisitExpression(*if_statement.test());
    VisitStatement(*if_statement.consequent());
    if (if_statement.alternate().has_value()) {
      VisitStatement(*if_statement.alternate().value());
    }
    if (postorder_callback_) {
      postorder_callback_->VisitIfStatement(if_statement);
    }
  }

  void VisitSwitchCase(JsSwitchCase &switch_case) override {
    if (preorder_callback_) {
      preorder_callback_->VisitSwitchCase(switch_case);
    }
    if (switch_case.test().has_value()) {
      VisitExpression(*switch_case.test().value());
    }
    for (std::unique_ptr<JsStatement> &consequent : *switch_case.consequent()) {
      VisitStatement(*consequent);
    }
    if (postorder_callback_) {
      postorder_callback_->VisitSwitchCase(switch_case);
    }
  }

  void VisitSwitchStatement(JsSwitchStatement &switch_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitSwitchStatement(switch_statement);
    }
    VisitExpression(*switch_statement.discriminant());
    for (std::unique_ptr<JsSwitchCase> &switch_case :
         *switch_statement.cases()) {
      VisitSwitchCase(*switch_case);
    }
    if (postorder_callback_) {
      postorder_callback_->VisitSwitchStatement(switch_statement);
    }
  }

  void VisitThrowStatement(JsThrowStatement &throw_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitThrowStatement(throw_statement);
    }
    VisitExpression(*throw_statement.argument());
    if (postorder_callback_) {
      postorder_callback_->VisitThrowStatement(throw_statement);
    }
  }

  void VisitCatchClause(JsCatchClause &catch_clause) override {
    if (preorder_callback_) {
      preorder_callback_->VisitCatchClause(catch_clause);
    }

    if (catch_clause.param().has_value()) {
      VisitPattern(*catch_clause.param().value());
    }
    VisitBlockStatement(*catch_clause.body());

    if (postorder_callback_) {
      postorder_callback_->VisitCatchClause(catch_clause);
    }
  }

  void VisitTryStatement(JsTryStatement &try_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitTryStatement(try_statement);
    }

    VisitBlockStatement(*try_statement.block());
    if (try_statement.handler().has_value()) {
      VisitCatchClause(*try_statement.handler().value());
    }
    if (try_statement.finalizer().has_value()) {
      VisitBlockStatement(*try_statement.finalizer().value());
    }

    if (postorder_callback_) {
      postorder_callback_->VisitTryStatement(try_statement);
    }
  }

  void VisitWhileStatement(JsWhileStatement &while_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitWhileStatement(while_statement);
    }

    VisitExpression(*while_statement.test());
    VisitStatement(*while_statement.body());

    if (postorder_callback_) {
      postorder_callback_->VisitWhileStatement(while_statement);
    }
  }

  void VisitDoWhileStatement(JsDoWhileStatement &do_while_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitDoWhileStatement(do_while_statement);
    }

    VisitStatement(*do_while_statement.body());
    VisitExpression(*do_while_statement.test());

    if (postorder_callback_) {
      postorder_callback_->VisitDoWhileStatement(do_while_statement);
    }
  }

  void VisitVariableDeclarator(
      JsVariableDeclarator &variable_declarator) override {
    if (preorder_callback_) {
      preorder_callback_->VisitVariableDeclarator(variable_declarator);
    }

    VisitLVal(*variable_declarator.id());
    if (variable_declarator.init().has_value()) {
      VisitExpression(*variable_declarator.init().value());
    }

    if (postorder_callback_) {
      postorder_callback_->VisitVariableDeclarator(variable_declarator);
    }
  }

  void VisitVariableDeclaration(
      JsVariableDeclaration &variable_declaration) override {
    if (preorder_callback_) {
      preorder_callback_->VisitVariableDeclaration(variable_declaration);
    }

    for (std::unique_ptr<JsVariableDeclarator> &declarator :
         *variable_declaration.declarations()) {
      VisitVariableDeclarator(*declarator);
    }

    if (postorder_callback_) {
      postorder_callback_->VisitVariableDeclaration(variable_declaration);
    }
  }

  void VisitForStatement(JsForStatement &for_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitForStatement(for_statement);
    }

    if (for_statement.init().has_value()) {
      std::variant<JsVariableDeclaration *, JsExpression *> init =
          for_statement.init().value();
      switch (init.index()) {
        case 0:
          VisitVariableDeclaration(*std::get<0>(init));
          break;
        case 1:
          VisitExpression(*std::get<1>(init));
          break;
        default:
          LOG(FATAL) << "Unreachable code.";
      }
    }

    if (for_statement.test().has_value()) {
      VisitExpression(*for_statement.test().value());
    }
    if (for_statement.update().has_value()) {
      VisitExpression(*for_statement.update().value());
    }
    VisitStatement(*for_statement.body());

    if (postorder_callback_) {
      postorder_callback_->VisitForStatement(for_statement);
    }
  }

  void VisitForInStatement(JsForInStatement &for_in_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitForInStatement(for_in_statement);
    }

    switch (for_in_statement.left().index()) {
      case 0:
        VisitVariableDeclaration(*std::get<0>(for_in_statement.left()));
        break;
      case 1:
        VisitLVal(*std::get<1>(for_in_statement.left()));
        break;
      default:
        LOG(FATAL) << "Unreachable code.";
    }
    VisitExpression(*for_in_statement.right());
    VisitStatement(*for_in_statement.body());

    if (postorder_callback_) {
      postorder_callback_->VisitForInStatement(for_in_statement);
    }
  }

  void VisitForOfStatement(JsForOfStatement &for_of_statement) override {
    if (preorder_callback_) {
      preorder_callback_->VisitForOfStatement(for_of_statement);
    }

    switch (for_of_statement.left().index()) {
      case 0:
        VisitVariableDeclaration(*std::get<0>(for_of_statement.left()));
        break;
      case 1:
        VisitLVal(*std::get<1>(for_of_statement.left()));
        break;
      default:
        LOG(FATAL) << "Unreachable code.";
    }
    VisitExpression(*for_of_statement.right());
    VisitStatement(*for_of_statement.body());

    if (postorder_callback_) {
      postorder_callback_->VisitForOfStatement(for_of_statement);
    }
  }

  void VisitFunctionDeclaration(
      JsFunctionDeclaration &function_declaration) override {
    if (preorder_callback_) {
      preorder_callback_->VisitFunctionDeclaration(function_declaration);
    }

    // This is an example of a leaf node calling functions in parent nodes.
    if (function_declaration.id().has_value()) {
      VisitIdentifier(*function_declaration.id().value());
    }
    for (std::unique_ptr<JsPattern> &param : *function_declaration.params()) {
      VisitPattern(*param);
    }
    VisitBlockStatement(*function_declaration.body());

    if (postorder_callback_) {
      postorder_callback_->VisitFunctionDeclaration(function_declaration);
    }
  }

  void VisitSuper(JsSuper &super) override {
    if (preorder_callback_) {
      preorder_callback_->VisitSuper(super);
    }

    if (postorder_callback_) {
      postorder_callback_->VisitSuper(super);
    }
  }

  void VisitImport(JsImport &import) override {
    if (preorder_callback_) {
      preorder_callback_->VisitImport(import);
    }

    if (postorder_callback_) {
      postorder_callback_->VisitImport(import);
    }
  }

  void VisitThisExpression(JsThisExpression &this_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitThisExpression(this_expression);
    }

    if (postorder_callback_) {
      postorder_callback_->VisitThisExpression(this_expression);
    }
  }

  void VisitArrowFunctionExpression(
      JsArrowFunctionExpression &arrow_function_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitArrowFunctionExpression(
          arrow_function_expression);
    }

    if (arrow_function_expression.id().has_value()) {
      VisitIdentifier(*arrow_function_expression.id().value());
    }
    for (std::unique_ptr<JsPattern> &param :
         *arrow_function_expression.params()) {
      VisitPattern(*param);
    }
    switch (arrow_function_expression.body().index()) {
      case 0:
        VisitBlockStatement(*std::get<0>(arrow_function_expression.body()));
        break;
      case 1:
        VisitExpression(*std::get<1>(arrow_function_expression.body()));
        break;
      default:
        LOG(FATAL) << "Unreachable code.";
    }

    if (postorder_callback_) {
      postorder_callback_->VisitArrowFunctionExpression(
          arrow_function_expression);
    }
  }

  void VisitYieldExpression(JsYieldExpression &yield_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitYieldExpression(yield_expression);
    }

    if (yield_expression.argument().has_value()) {
      VisitExpression(*yield_expression.argument().value());
    }

    if (postorder_callback_) {
      postorder_callback_->VisitYieldExpression(yield_expression);
    }
  }

  void VisitAwaitExpression(JsAwaitExpression &await_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitAwaitExpression(await_expression);
    }

    if (await_expression.argument().has_value()) {
      VisitExpression(*await_expression.argument().value());
    }

    if (postorder_callback_) {
      postorder_callback_->VisitAwaitExpression(await_expression);
    }
  }

  void VisitSpreadElement(JsSpreadElement &spread_element) override {
    if (preorder_callback_) {
      preorder_callback_->VisitSpreadElement(spread_element);
    }

    VisitExpression(*spread_element.argument());

    if (postorder_callback_) {
      postorder_callback_->VisitSpreadElement(spread_element);
    }
  }

  void VisitArrayExpression(JsArrayExpression &array_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitArrayExpression(array_expression);
    }

    for (std::optional<std::variant<std::unique_ptr<JsExpression>,
                                    std::unique_ptr<JsSpreadElement>>>
             &element : *array_expression.elements()) {
      if (element.has_value()) {
        std::variant<std::unique_ptr<JsExpression>,
                     std::unique_ptr<JsSpreadElement>> &element_variant =
            element.value();
        switch (element_variant.index()) {
          case 0:
            VisitExpression(*std::get<0>(element_variant));
            break;
          case 1:
            VisitSpreadElement(*std::get<1>(element_variant));
            break;
          default:
            LOG(FATAL) << "Unreachable code.";
        }
      }
    }

    if (postorder_callback_) {
      postorder_callback_->VisitArrayExpression(array_expression);
    }
  }

  void VisitObjectProperty(JsObjectProperty &object_property) override {
    if (preorder_callback_) {
      preorder_callback_->VisitObjectProperty(object_property);
    }

    VisitExpression(*object_property.key());
    switch (object_property.value().index()) {
      case 0:
        VisitExpression(*std::get<0>(object_property.value()));
        break;
      case 1:
        VisitPattern(*std::get<1>(object_property.value()));
        break;
      default:
        LOG(FATAL) << "Unreachable code.";
    }

    if (postorder_callback_) {
      postorder_callback_->VisitObjectProperty(object_property);
    }
  }

  void VisitObjectMethod(JsObjectMethod &object_method) override {
    if (preorder_callback_) {
      preorder_callback_->VisitObjectMethod(object_method);
    }

    VisitExpression(*object_method.key());
    if (object_method.id().has_value()) {
      VisitIdentifier(*object_method.id().value());
    }
    for (std::unique_ptr<JsPattern> &param : *object_method.params()) {
      VisitPattern(*param);
    }
    VisitBlockStatement(*object_method.body());

    if (postorder_callback_) {
      postorder_callback_->VisitObjectMethod(object_method);
    }
  }

  void VisitObjectExpression(JsObjectExpression &object_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitObjectExpression(object_expression);
    }

    for (std::variant<std::unique_ptr<JsObjectProperty>,
                      std::unique_ptr<JsObjectMethod>,
                      std::unique_ptr<JsSpreadElement>> &property :
         *object_expression.properties_()) {
      switch (property.index()) {
        case 0:
          VisitObjectProperty(*std::get<0>(property));
          break;
        case 1:
          VisitObjectMethod(*std::get<1>(property));
          break;
        case 2:
          VisitSpreadElement(*std::get<2>(property));
          break;
        default:
          LOG(FATAL) << "Unreachable code.";
      }
    }

    if (postorder_callback_) {
      postorder_callback_->VisitObjectExpression(object_expression);
    }
  }

  void VisitFunctionExpression(
      JsFunctionExpression &function_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitFunctionExpression(function_expression);
    }

    if (function_expression.id().has_value()) {
      VisitIdentifier(*function_expression.id().value());
    }
    for (std::unique_ptr<JsPattern> &param : *function_expression.params()) {
      VisitPattern(*param);
    }
    VisitBlockStatement(*function_expression.body());

    if (postorder_callback_) {
      postorder_callback_->VisitFunctionExpression(function_expression);
    }
  }

  void VisitUnaryExpression(JsUnaryExpression &unary_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitUnaryExpression(unary_expression);
    }

    VisitExpression(*unary_expression.argument());

    if (postorder_callback_) {
      postorder_callback_->VisitUnaryExpression(unary_expression);
    }
  }

  void VisitUpdateExpression(JsUpdateExpression &update_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitUpdateExpression(update_expression);
    }

    VisitLVal(*update_expression.argument());

    if (postorder_callback_) {
      postorder_callback_->VisitUpdateExpression(update_expression);
    }
  }

  void VisitBinaryExpression(JsBinaryExpression &binary_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitBinaryExpression(binary_expression);
    }

    switch (binary_expression.left().index()) {
      case 0:
        VisitExpression(*std::get<0>(binary_expression.left()));
        break;
      case 1:
        VisitPrivateName(*std::get<1>(binary_expression.left()));
        break;
      default:
        LOG(FATAL) << "Unreachable code.";
    }
    VisitExpression(*binary_expression.right());

    if (postorder_callback_) {
      postorder_callback_->VisitBinaryExpression(binary_expression);
    }
  }

  void VisitAssignmentExpression(
      JsAssignmentExpression &assignment_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitAssignmentExpression(assignment_expression);
    }

    VisitLVal(*assignment_expression.left());
    VisitExpression(*assignment_expression.right());

    if (postorder_callback_) {
      postorder_callback_->VisitAssignmentExpression(assignment_expression);
    }
  }

  void VisitLogicalExpression(
      JsLogicalExpression &logical_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitLogicalExpression(logical_expression);
    }

    VisitExpression(*logical_expression.left());
    VisitExpression(*logical_expression.right());

    if (postorder_callback_) {
      postorder_callback_->VisitLogicalExpression(logical_expression);
    }
  }

  void VisitMemberExpression(JsMemberExpression &member_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitMemberExpression(member_expression);
    }

    switch (member_expression.object().index()) {
      case 0:
        VisitExpression(*std::get<0>(member_expression.object()));
        break;
      case 1:
        VisitSuper(*std::get<1>(member_expression.object()));
        break;
      default:
        LOG(FATAL) << "Unreachable code.";
    }
    switch (member_expression.property().index()) {
      case 0:
        VisitExpression(*std::get<0>(member_expression.property()));
        break;
      case 1:
        VisitPrivateName(*std::get<1>(member_expression.property()));
        break;
      default:
        LOG(FATAL) << "Unreachable code.";
    }

    if (postorder_callback_) {
      postorder_callback_->VisitMemberExpression(member_expression);
    }
  }

  void VisitOptionalMemberExpression(
      JsOptionalMemberExpression &optional_member_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitOptionalMemberExpression(
          optional_member_expression);
    }

    VisitExpression(*optional_member_expression.object());
    switch (optional_member_expression.property().index()) {
      case 0:
        VisitExpression(*std::get<0>(optional_member_expression.property()));
        break;
      case 1:
        VisitPrivateName(*std::get<1>(optional_member_expression.property()));
        break;
      default:
        LOG(FATAL) << "Unreachable code.";
    }

    if (postorder_callback_) {
      postorder_callback_->VisitOptionalMemberExpression(
          optional_member_expression);
    }
  }

  void VisitConditionalExpression(
      JsConditionalExpression &conditional_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitConditionalExpression(conditional_expression);
    }

    VisitExpression(*conditional_expression.test());
    VisitExpression(*conditional_expression.alternate());
    VisitExpression(*conditional_expression.consequent());

    if (postorder_callback_) {
      postorder_callback_->VisitConditionalExpression(conditional_expression);
    }
  }

  void VisitCallExpression(JsCallExpression &call_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitCallExpression(call_expression);
    }

    switch (call_expression.callee().index()) {
      case 0:
        VisitExpression(*std::get<0>(call_expression.callee()));
        break;
      case 1:
        VisitSuper(*std::get<1>(call_expression.callee()));
        break;
      case 2:
        VisitImport(*std::get<2>(call_expression.callee()));
        break;
      default:
        LOG(FATAL) << "Unreachable code.";
    }
    for (std::variant<std::unique_ptr<JsExpression>,
                      std::unique_ptr<JsSpreadElement>> &argument :
         *call_expression.arguments()) {
      switch (argument.index()) {
        case 0:
          VisitExpression(*std::get<0>(argument));
          break;
        case 1:
          VisitSpreadElement(*std::get<1>(argument));
          break;
        default:
          LOG(FATAL) << "Unreachable code.";
      }
    }

    if (postorder_callback_) {
      postorder_callback_->VisitCallExpression(call_expression);
    }
  }

  void VisitOptionalCallExpression(
      JsOptionalCallExpression &optional_call_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitOptionalCallExpression(optional_call_expression);
    }

    VisitExpression(*optional_call_expression.callee());
    for (std::variant<std::unique_ptr<JsExpression>,
                      std::unique_ptr<JsSpreadElement>> &argument :
         *optional_call_expression.arguments()) {
      switch (argument.index()) {
        case 0:
          VisitExpression(*std::get<0>(argument));
          break;
        case 1:
          VisitSpreadElement(*std::get<1>(argument));
          break;
        default:
          LOG(FATAL) << "Unreachable code.";
      }
    }

    if (postorder_callback_) {
      postorder_callback_->VisitOptionalCallExpression(
          optional_call_expression);
    }
  }

  void VisitNewExpression(JsNewExpression &new_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitNewExpression(new_expression);
    }

    switch (new_expression.callee().index()) {
      case 0:
        VisitExpression(*std::get<0>(new_expression.callee()));
        break;
      case 1:
        VisitSuper(*std::get<1>(new_expression.callee()));
        break;
      case 2:
        VisitImport(*std::get<2>(new_expression.callee()));
        break;
      default:
        LOG(FATAL) << "Unreachable code.";
    }
    for (std::variant<std::unique_ptr<JsExpression>,
                      std::unique_ptr<JsSpreadElement>> &argument :
         *new_expression.arguments()) {
      switch (argument.index()) {
        case 0:
          VisitExpression(*std::get<0>(argument));
          break;
        case 1:
          VisitSpreadElement(*std::get<1>(argument));
          break;
        default:
          LOG(FATAL) << "Unreachable code.";
      }
    }

    if (postorder_callback_) {
      postorder_callback_->VisitNewExpression(new_expression);
    }
  }

  void VisitSequenceExpression(
      JsSequenceExpression &sequence_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitSequenceExpression(sequence_expression);
    }

    for (std::unique_ptr<JsExpression> &expression :
         *sequence_expression.expressions()) {
      VisitExpression(*expression);
    }

    if (postorder_callback_) {
      postorder_callback_->VisitSequenceExpression(sequence_expression);
    }
  }

  void VisitParenthesizedExpression(
      JsParenthesizedExpression &parenthesized_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitParenthesizedExpression(
          parenthesized_expression);
    }

    VisitExpression(*parenthesized_expression.expression());

    if (postorder_callback_) {
      postorder_callback_->VisitParenthesizedExpression(
          parenthesized_expression);
    }
  }

  void VisitTemplateElement(JsTemplateElement &template_element) override {
    if (preorder_callback_) {
      preorder_callback_->VisitTemplateElement(template_element);
    }

    if (postorder_callback_) {
      postorder_callback_->VisitTemplateElement(template_element);
    }
  }

  void VisitTemplateLiteral(JsTemplateLiteral &template_literal) override {
    if (preorder_callback_) {
      preorder_callback_->VisitTemplateLiteral(template_literal);
    }

    for (std::unique_ptr<JsTemplateElement> &quasi :
         *template_literal.quasis()) {
      VisitTemplateElement(*quasi);
    }
    for (std::unique_ptr<JsExpression> &expression :
         *template_literal.expressions()) {
      VisitExpression(*expression);
    }

    if (postorder_callback_) {
      postorder_callback_->VisitTemplateLiteral(template_literal);
    }
  }

  void VisitTaggedTemplateExpression(
      JsTaggedTemplateExpression &tagged_template_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitTaggedTemplateExpression(
          tagged_template_expression);
    }

    VisitExpression(*tagged_template_expression.tag());
    VisitTemplateLiteral(*tagged_template_expression.quasi());

    if (postorder_callback_) {
      postorder_callback_->VisitTaggedTemplateExpression(
          tagged_template_expression);
    }
  }

  void VisitRestElement(JsRestElement &rest_element) override {
    if (preorder_callback_) {
      preorder_callback_->VisitRestElement(rest_element);
    }

    VisitLVal(*rest_element.argument());

    if (postorder_callback_) {
      postorder_callback_->VisitRestElement(rest_element);
    }
  }

  void VisitObjectPattern(JsObjectPattern &object_pattern) override {
    if (preorder_callback_) {
      preorder_callback_->VisitObjectPattern(object_pattern);
    }

    for (std::variant<std::unique_ptr<JsObjectProperty>,
                      std::unique_ptr<JsRestElement>> &property :
         *object_pattern.properties_()) {
      switch (property.index()) {
        case 0:
          VisitObjectProperty(*std::get<0>(property));
          break;
        case 1:
          VisitRestElement(*std::get<1>(property));
          break;
        default:
          LOG(FATAL) << "Unreachable code.";
      }
    }

    if (postorder_callback_) {
      postorder_callback_->VisitObjectPattern(object_pattern);
    }
  }

  void VisitArrayPattern(JsArrayPattern &array_pattern) override {
    if (preorder_callback_) {
      preorder_callback_->VisitArrayPattern(array_pattern);
    }

    for (std::optional<std::unique_ptr<JsPattern>> &element :
         *array_pattern.elements()) {
      if (element.has_value()) {
        VisitPattern(*element.value());
      }
    }

    if (postorder_callback_) {
      postorder_callback_->VisitArrayPattern(array_pattern);
    }
  }

  void VisitAssignmentPattern(
      JsAssignmentPattern &assignment_pattern) override {
    if (preorder_callback_) {
      preorder_callback_->VisitAssignmentPattern(assignment_pattern);
    }

    VisitPattern(*assignment_pattern.left());
    VisitExpression(*assignment_pattern.right());

    if (postorder_callback_) {
      postorder_callback_->VisitAssignmentPattern(assignment_pattern);
    }
  }

  void VisitClassMethod(JsClassMethod &class_method) override {
    if (preorder_callback_) {
      preorder_callback_->VisitClassMethod(class_method);
    }

    VisitExpression(*class_method.key());
    if (class_method.id().has_value()) {
      VisitIdentifier(*class_method.id().value());
    }
    for (std::unique_ptr<JsPattern> &param : *class_method.params()) {
      VisitPattern(*param);
    }
    VisitBlockStatement(*class_method.body());

    if (postorder_callback_) {
      postorder_callback_->VisitClassMethod(class_method);
    }
  }

  void VisitClassPrivateMethod(
      JsClassPrivateMethod &class_private_method) override {
    if (preorder_callback_) {
      preorder_callback_->VisitClassPrivateMethod(class_private_method);
    }

    VisitPrivateName(*class_private_method.key());
    if (class_private_method.id().has_value()) {
      VisitIdentifier(*class_private_method.id().value());
    }
    for (std::unique_ptr<JsPattern> &param : *class_private_method.params()) {
      VisitPattern(*param);
    }
    VisitBlockStatement(*class_private_method.body());

    if (postorder_callback_) {
      postorder_callback_->VisitClassPrivateMethod(class_private_method);
    }
  }

  void VisitClassProperty(JsClassProperty &class_property) override {
    if (preorder_callback_) {
      preorder_callback_->VisitClassProperty(class_property);
    }

    VisitExpression(*class_property.key());
    if (class_property.value().has_value()) {
      VisitExpression(*class_property.value().value());
    }

    if (postorder_callback_) {
      postorder_callback_->VisitClassProperty(class_property);
    }
  }

  void VisitClassPrivateProperty(
      JsClassPrivateProperty &class_private_property) override {
    if (preorder_callback_) {
      preorder_callback_->VisitClassPrivateProperty(class_private_property);
    }

    VisitPrivateName(*class_private_property.key());
    if (class_private_property.value().has_value()) {
      VisitExpression(*class_private_property.value().value());
    }

    if (postorder_callback_) {
      postorder_callback_->VisitClassPrivateProperty(class_private_property);
    }
  }

  void VisitClassBody(JsClassBody &class_body) override {
    if (preorder_callback_) {
      preorder_callback_->VisitClassBody(class_body);
    }

    for (std::variant<std::unique_ptr<JsClassMethod>,
                      std::unique_ptr<JsClassPrivateMethod>,
                      std::unique_ptr<JsClassProperty>,
                      std::unique_ptr<JsClassPrivateProperty>> &body_element :
         *class_body.body()) {
      switch (body_element.index()) {
        case 0:
          VisitClassMethod(*std::get<0>(body_element));
          break;
        case 1:
          VisitClassPrivateMethod(*std::get<1>(body_element));
          break;
        case 2:
          VisitClassProperty(*std::get<2>(body_element));
          break;
        case 3:
          VisitClassPrivateProperty(*std::get<3>(body_element));
          break;
        default:
          LOG(FATAL) << "Unreachable code.";
      }
    }

    if (postorder_callback_) {
      postorder_callback_->VisitClassBody(class_body);
    }
  }

  void VisitClassDeclaration(JsClassDeclaration &class_declaration) override {
    if (preorder_callback_) {
      preorder_callback_->VisitClassDeclaration(class_declaration);
    }

    if (class_declaration.id().has_value()) {
      VisitIdentifier(*class_declaration.id().value());
    }
    if (class_declaration.super_class().has_value()) {
      VisitExpression(*class_declaration.super_class().value());
    }
    VisitClassBody(*class_declaration.body());

    if (postorder_callback_) {
      postorder_callback_->VisitClassDeclaration(class_declaration);
    }
  }

  void VisitClassExpression(JsClassExpression &class_expression) override {
    if (preorder_callback_) {
      preorder_callback_->VisitClassExpression(class_expression);
    }

    if (class_expression.id().has_value()) {
      VisitIdentifier(*class_expression.id().value());
    }
    if (class_expression.super_class().has_value()) {
      VisitExpression(*class_expression.super_class().value());
    }
    VisitClassBody(*class_expression.body());

    if (postorder_callback_) {
      postorder_callback_->VisitClassExpression(class_expression);
    }
  }

  void VisitMetaProperty(JsMetaProperty &meta_property) override {
    if (preorder_callback_) {
      preorder_callback_->VisitMetaProperty(meta_property);
    }

    VisitIdentifier(*meta_property.meta());
    VisitIdentifier(*meta_property.property());

    if (postorder_callback_) {
      postorder_callback_->VisitMetaProperty(meta_property);
    }
  }

  void VisitImportSpecifier(JsImportSpecifier &import_specifier) override {
    if (preorder_callback_) {
      preorder_callback_->VisitImportSpecifier(import_specifier);
    }

    switch (import_specifier.imported().index()) {
      case 0:
        VisitIdentifier(*std::get<0>(import_specifier.imported()));
        break;
      case 1:
        VisitStringLiteral(*std::get<1>(import_specifier.imported()));
        break;
      default:
        LOG(FATAL) << "Unreachable code.";
    }
    VisitIdentifier(*import_specifier.local());

    if (postorder_callback_) {
      postorder_callback_->VisitImportSpecifier(import_specifier);
    }
  }

  void VisitImportDefaultSpecifier(
      JsImportDefaultSpecifier &import_default_specifier) override {
    if (preorder_callback_) {
      preorder_callback_->VisitImportDefaultSpecifier(import_default_specifier);
    }

    VisitIdentifier(*import_default_specifier.local());

    if (postorder_callback_) {
      postorder_callback_->VisitImportDefaultSpecifier(
          import_default_specifier);
    }
  }

  void VisitImportNamespaceSpecifier(
      JsImportNamespaceSpecifier &import_namespace_specifier) override {
    if (preorder_callback_) {
      preorder_callback_->VisitImportNamespaceSpecifier(
          import_namespace_specifier);
    }

    VisitIdentifier(*import_namespace_specifier.local());

    if (postorder_callback_) {
      postorder_callback_->VisitImportNamespaceSpecifier(
          import_namespace_specifier);
    }
  }

  void VisitImportAttribute(JsImportAttribute &import_attribute) override {
    if (preorder_callback_) {
      preorder_callback_->VisitImportAttribute(import_attribute);
    }

    VisitIdentifier(*import_attribute.key());
    VisitStringLiteral(*import_attribute.value());

    if (postorder_callback_) {
      postorder_callback_->VisitImportAttribute(import_attribute);
    }
  }

  void VisitImportDeclaration(
      JsImportDeclaration &import_declaration) override {
    if (preorder_callback_) {
      preorder_callback_->VisitImportDeclaration(import_declaration);
    }

    for (std::variant<std::unique_ptr<JsImportSpecifier>,
                      std::unique_ptr<JsImportDefaultSpecifier>,
                      std::unique_ptr<JsImportNamespaceSpecifier>> &specifier :
         *import_declaration.specifiers()) {
      switch (specifier.index()) {
        case 0:
          VisitImportSpecifier(*std::get<0>(specifier));
          break;
        case 1:
          VisitImportDefaultSpecifier(*std::get<1>(specifier));
          break;
        case 2:
          VisitImportNamespaceSpecifier(*std::get<2>(specifier));
          break;
        default:
          LOG(FATAL) << "Unreachable code.";
      }
    }
    VisitStringLiteral(*import_declaration.source());
    if (import_declaration.assertions().has_value()) {
      VisitImportAttribute(*import_declaration.assertions().value());
    }

    if (postorder_callback_) {
      postorder_callback_->VisitImportDeclaration(import_declaration);
    }
  }

  void VisitExportSpecifier(JsExportSpecifier &export_specifier) override {
    if (preorder_callback_) {
      preorder_callback_->VisitExportSpecifier(export_specifier);
    }

    switch (export_specifier.exported().index()) {
      case 0:
        VisitIdentifier(*std::get<0>(export_specifier.exported()));
        break;
      case 1:
        VisitStringLiteral(*std::get<1>(export_specifier.exported()));
        break;
      default:
        LOG(FATAL) << "Unreachable code.";
    }
    if (export_specifier.local().has_value()) {
      switch (export_specifier.local().value().index()) {
        case 0:
          VisitIdentifier(*std::get<0>(export_specifier.local().value()));
          break;
        case 1:
          VisitStringLiteral(*std::get<1>(export_specifier.local().value()));
          break;
        default:
          LOG(FATAL) << "Unreachable code.";
      }
    }

    if (postorder_callback_) {
      postorder_callback_->VisitExportSpecifier(export_specifier);
    }
  }

  void VisitExportNamedDeclaration(
      JsExportNamedDeclaration &export_named_declaration) override {
    if (preorder_callback_) {
      preorder_callback_->VisitExportNamedDeclaration(export_named_declaration);
    }

    if (export_named_declaration.declaration().has_value()) {
      VisitDeclaration(*export_named_declaration.declaration().value());
    }
    for (std::unique_ptr<JsExportSpecifier> &specifier :
         *export_named_declaration.specifiers()) {
      VisitExportSpecifier(*specifier);
    }
    if (export_named_declaration.source().has_value()) {
      VisitStringLiteral(*export_named_declaration.source().value());
    }
    if (export_named_declaration.assertions().has_value()) {
      std::vector<std::unique_ptr<JsImportAttribute>> *assertions =
          export_named_declaration.assertions().value();
      for (std::unique_ptr<JsImportAttribute> &assertion : *assertions) {
        VisitImportAttribute(*assertion);
      }
    }

    if (postorder_callback_) {
      postorder_callback_->VisitExportNamedDeclaration(
          export_named_declaration);
    }
  }

  void VisitExportDefaultDeclaration(
      JsExportDefaultDeclaration &export_default_declaration) override {
    if (preorder_callback_) {
      preorder_callback_->VisitExportDefaultDeclaration(
          export_default_declaration);
    }

    switch (export_default_declaration.declaration().index()) {
      case 0:
        VisitFunctionDeclaration(
            *std::get<0>(export_default_declaration.declaration()));
        break;
      case 1:
        VisitClassDeclaration(
            *std::get<1>(export_default_declaration.declaration()));
        break;
      case 2:
        VisitExpression(
            *std::get<2>(export_default_declaration.declaration()));
        break;
      default:
        LOG(FATAL) << "Unreachable code.";
    }

    if (postorder_callback_) {
      postorder_callback_->VisitExportDefaultDeclaration(
          export_default_declaration);
    }
  }

  void VisitExportAllDeclaration(
      JsExportAllDeclaration &export_all_declaration) override {
    if (preorder_callback_) {
      preorder_callback_->VisitExportAllDeclaration(export_all_declaration);
    }

    VisitStringLiteral(*export_all_declaration.source());
    if (export_all_declaration.assertions().has_value()) {
      std::vector<std::unique_ptr<JsImportAttribute>> *assertions =
          export_all_declaration.assertions().value();
      for (std::unique_ptr<JsImportAttribute> &assertion : *assertions) {
        VisitImportAttribute(*assertion);
      }
    }

    if (postorder_callback_) {
      postorder_callback_->VisitExportAllDeclaration(export_all_declaration);
    }
  }

 private:
  MutableJsAstVisitor<void> *preorder_callback_;
  MutableJsAstVisitor<void> *postorder_callback_;
};

}  // namespace maldoca

#endif  // MALDOCA_JS_AST_AST_WALKER_H_
