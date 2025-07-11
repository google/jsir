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

#ifndef MALDOCA_JS_AST_AST_VISITOR_H_
#define MALDOCA_JS_AST_AST_VISITOR_H_

#include <functional>
#include <utility>

#include "absl/log/log.h"
#include "maldoca/js/ast/ast.generated.h"

namespace maldoca {

// The AST visitor is the main API for visiting the AST. Clients are supposed to
// sub-class the JsAstVisitor class and implement all the functions.
// Note that JsAstVisitor only defines functions to visit the leaf nodes in the
// type hierarchy of the AST API.
template <typename R>
class JsPatternVisitor {
 public:
  virtual ~JsPatternVisitor() = default;

  R VisitPattern(const JsPattern &pattern) {
    if (const auto *identifier = dynamic_cast<const JsIdentifier *>(&pattern)) {
      return VisitIdentifier(*identifier);
    }
    if (const auto *member_expression =
            dynamic_cast<const JsMemberExpression *>(&pattern)) {
      return VisitMemberExpression(*member_expression);
    }
    if (const auto *parenthesized_expression =
            dynamic_cast<const JsParenthesizedExpression *>(&pattern)) {
      return VisitParenthesizedExpression(*parenthesized_expression);
    }
    if (const auto *rest_element =
            dynamic_cast<const JsRestElement *>(&pattern)) {
      return VisitRestElement(*rest_element);
    }
    if (const auto *object_pattern =
            dynamic_cast<const JsObjectPattern *>(&pattern)) {
      return VisitObjectPattern(*object_pattern);
    }
    if (const auto *array_pattern =
            dynamic_cast<const JsArrayPattern *>(&pattern)) {
      return VisitArrayPattern(*array_pattern);
    }
    if (const auto *assignment_pattern =
            dynamic_cast<const JsAssignmentPattern *>(&pattern)) {
      return VisitAssignmentPattern(*assignment_pattern);
    }

    LOG(FATAL) << "Unreachable code.";
  }

  virtual R VisitIdentifier(const JsIdentifier &node) = 0;
  virtual R VisitMemberExpression(const JsMemberExpression &node) = 0;
  virtual R VisitParenthesizedExpression(
      const JsParenthesizedExpression &node) = 0;
  virtual R VisitRestElement(const JsRestElement &node) = 0;
  virtual R VisitObjectPattern(const JsObjectPattern &node) = 0;
  virtual R VisitArrayPattern(const JsArrayPattern &node) = 0;
  virtual R VisitAssignmentPattern(const JsAssignmentPattern &node) = 0;
};

template <typename R>
class JsLValVisitor {
 public:
  virtual ~JsLValVisitor() = default;

  R VisitLVal(const JsLVal &lval) {
    if (const JsIdentifier *identifier =
            dynamic_cast<const JsIdentifier *>(&lval)) {
      return VisitIdentifier(*identifier);
    }
    if (const JsMemberExpression *member_expression =
            dynamic_cast<const JsMemberExpression *>(&lval)) {
      return VisitMemberExpression(*member_expression);
    }
    if (const JsParenthesizedExpression *parenthesized_expression =
            dynamic_cast<const JsParenthesizedExpression *>(&lval)) {
      return VisitParenthesizedExpression(*parenthesized_expression);
    }
    if (const JsRestElement *rest_element =
            dynamic_cast<const JsRestElement *>(&lval)) {
      return VisitRestElement(*rest_element);
    }
    if (const JsObjectPattern *object_pattern =
            dynamic_cast<const JsObjectPattern *>(&lval)) {
      return VisitObjectPattern(*object_pattern);
    }
    if (const JsArrayPattern *array_pattern =
            dynamic_cast<const JsArrayPattern *>(&lval)) {
      return VisitArrayPattern(*array_pattern);
    }
    if (const JsAssignmentPattern *assignment_pattern =
            dynamic_cast<const JsAssignmentPattern *>(&lval)) {
      return VisitAssignmentPattern(*assignment_pattern);
    }

    LOG(FATAL) << "Unreachable code.";
  }

  virtual R VisitIdentifier(const JsIdentifier &node) = 0;
  virtual R VisitMemberExpression(const JsMemberExpression &node) = 0;
  virtual R VisitParenthesizedExpression(
      const JsParenthesizedExpression &node) = 0;
  virtual R VisitRestElement(const JsRestElement &node) = 0;
  virtual R VisitObjectPattern(const JsObjectPattern &node) = 0;
  virtual R VisitArrayPattern(const JsArrayPattern &node) = 0;
  virtual R VisitAssignmentPattern(const JsAssignmentPattern &node) = 0;
};

template <typename R>
class JsAstVisitor : public virtual JsPatternVisitor<R>,
                     public virtual JsLValVisitor<R> {
 public:
  virtual ~JsAstVisitor() = default;

  virtual R VisitInterpreterDirective(
      const JsInterpreterDirective &interpreter_directive) = 0;

  virtual R VisitDirectiveLiteral(
      const JsDirectiveLiteral &directive_literal) = 0;

  virtual R VisitDirective(const JsDirective &directive) = 0;

  virtual R VisitProgram(const JsProgram &program) = 0;

  virtual R VisitFile(const JsFile &file) = 0;

  R VisitIdentifier(const JsIdentifier &identifier) override = 0;

  virtual R VisitPrivateName(const JsPrivateName &private_name) = 0;

  virtual R VisitRegExpLiteral(const JsRegExpLiteral &reg_exp_literal) = 0;

  virtual R VisitNullLiteral(const JsNullLiteral &null_literal) = 0;

  virtual R VisitStringLiteral(const JsStringLiteral &string_literal) = 0;

  virtual R VisitBooleanLiteral(const JsBooleanLiteral &boolean_literal) = 0;

  virtual R VisitNumericLiteral(const JsNumericLiteral &numeric_literal) = 0;

  virtual R VisitBigIntLiteral(const JsBigIntLiteral &big_int_literal) = 0;

  virtual R VisitBlockStatement(const JsBlockStatement &block_statement) = 0;

  virtual R VisitExpressionStatement(
      const JsExpressionStatement &expression_statement) = 0;

  virtual R VisitEmptyStatement(const JsEmptyStatement &empty_statement) = 0;

  virtual R VisitDebuggerStatement(
      const JsDebuggerStatement &debugger_statement) = 0;

  virtual R VisitWithStatement(const JsWithStatement &with_statement) = 0;

  virtual R VisitReturnStatement(const JsReturnStatement &return_statement) = 0;

  virtual R VisitLabeledStatement(
      const JsLabeledStatement &labeled_statement) = 0;

  virtual R VisitBreakStatement(const JsBreakStatement &break_statement) = 0;

  virtual R VisitContinueStatement(
      const JsContinueStatement &continue_statement) = 0;

  virtual R VisitIfStatement(const JsIfStatement &if_statement) = 0;

  virtual R VisitSwitchCase(const JsSwitchCase &switch_case) = 0;

  virtual R VisitSwitchStatement(const JsSwitchStatement &switch_statement) = 0;

  virtual R VisitThrowStatement(const JsThrowStatement &throw_statement) = 0;

  virtual R VisitCatchClause(const JsCatchClause &catch_clause) = 0;

  virtual R VisitTryStatement(const JsTryStatement &try_statement) = 0;

  virtual R VisitWhileStatement(const JsWhileStatement &while_statement) = 0;

  virtual R VisitDoWhileStatement(
      const JsDoWhileStatement &do_while_statement) = 0;

  virtual R VisitVariableDeclarator(
      const JsVariableDeclarator &variable_declarator) = 0;

  virtual R VisitVariableDeclaration(
      const JsVariableDeclaration &variable_declaration) = 0;

  virtual R VisitForStatement(const JsForStatement &for_statement) = 0;

  virtual R VisitForInStatement(const JsForInStatement &for_in_statement) = 0;

  virtual R VisitForOfStatement(const JsForOfStatement &for_of_statement) = 0;

  virtual R VisitFunctionDeclaration(
      const JsFunctionDeclaration &function_declaration) = 0;

  virtual R VisitSuper(const JsSuper &super) = 0;

  virtual R VisitImport(const JsImport &import) = 0;

  virtual R VisitThisExpression(const JsThisExpression &this_expression) = 0;

  virtual R VisitArrowFunctionExpression(
      const JsArrowFunctionExpression &arrow_function_expression) = 0;

  virtual R VisitYieldExpression(const JsYieldExpression &yield_expression) = 0;

  virtual R VisitAwaitExpression(const JsAwaitExpression &await_expression) = 0;

  virtual R VisitSpreadElement(const JsSpreadElement &spread_element) = 0;

  virtual R VisitArrayExpression(const JsArrayExpression &array_expression) = 0;

  virtual R VisitObjectProperty(const JsObjectProperty &object_property) = 0;

  virtual R VisitObjectMethod(const JsObjectMethod &object_method) = 0;

  virtual R VisitObjectExpression(
      const JsObjectExpression &object_expression) = 0;

  virtual R VisitFunctionExpression(
      const JsFunctionExpression &function_expression) = 0;

  virtual R VisitUnaryExpression(const JsUnaryExpression &unary_expression) = 0;

  virtual R VisitUpdateExpression(
      const JsUpdateExpression &update_expression) = 0;

  virtual R VisitBinaryExpression(
      const JsBinaryExpression &binary_expression) = 0;

  virtual R VisitAssignmentExpression(
      const JsAssignmentExpression &assignment_expression) = 0;

  virtual R VisitLogicalExpression(
      const JsLogicalExpression &logical_expression) = 0;

  R VisitMemberExpression(
      const JsMemberExpression &member_expression) override = 0;

  virtual R VisitOptionalMemberExpression(
      const JsOptionalMemberExpression &optional_member_expression) = 0;

  virtual R VisitConditionalExpression(
      const JsConditionalExpression &conditional_expression) = 0;

  virtual R VisitCallExpression(const JsCallExpression &call_expression) = 0;

  virtual R VisitOptionalCallExpression(
      const JsOptionalCallExpression &optional_call_expression) = 0;

  virtual R VisitNewExpression(const JsNewExpression &new_expression) = 0;

  virtual R VisitSequenceExpression(
      const JsSequenceExpression &sequence_expression) = 0;

  R VisitParenthesizedExpression(
      const JsParenthesizedExpression &parenthesized_expression) override = 0;

  virtual R VisitTemplateElement(const JsTemplateElement &template_element) = 0;

  virtual R VisitTemplateLiteral(const JsTemplateLiteral &template_literal) = 0;

  virtual R VisitTaggedTemplateExpression(
      const JsTaggedTemplateExpression &tagged_template_expression) = 0;

  R VisitRestElement(const JsRestElement &rest_element) override = 0;

  R VisitObjectPattern(const JsObjectPattern &object_pattern) override = 0;

  R VisitArrayPattern(const JsArrayPattern &array_pattern) override = 0;

  R VisitAssignmentPattern(
      const JsAssignmentPattern &assignment_pattern) override = 0;

  virtual R VisitClassMethod(const JsClassMethod &class_method) = 0;

  virtual R VisitClassPrivateMethod(
      const JsClassPrivateMethod &class_private_method) = 0;

  virtual R VisitClassProperty(const JsClassProperty &class_property) = 0;

  virtual R VisitClassPrivateProperty(
      const JsClassPrivateProperty &class_private_property) = 0;

  virtual R VisitClassBody(const JsClassBody &class_body) = 0;

  virtual R VisitClassDeclaration(
      const JsClassDeclaration &class_declaration) = 0;

  virtual R VisitClassExpression(const JsClassExpression &class_expression) = 0;

  virtual R VisitMetaProperty(const JsMetaProperty &meta_property) = 0;

  virtual R VisitImportSpecifier(const JsImportSpecifier &import_specifier) = 0;

  virtual R VisitImportDefaultSpecifier(
      const JsImportDefaultSpecifier &import_default_specifier) = 0;

  virtual R VisitImportNamespaceSpecifier(
      const JsImportNamespaceSpecifier &import_namespace_specifier) = 0;

  virtual R VisitImportAttribute(const JsImportAttribute &import_attribute) = 0;

  virtual R VisitImportDeclaration(
      const JsImportDeclaration &import_declaration) = 0;

  virtual R VisitExportSpecifier(const JsExportSpecifier &export_specifier) = 0;

  virtual R VisitExportNamedDeclaration(
      const JsExportNamedDeclaration &export_named_declaration) = 0;

  virtual R VisitExportDefaultDeclaration(
      const JsExportDefaultDeclaration &export_default_declaration) = 0;

  virtual R VisitExportAllDeclaration(
      const JsExportAllDeclaration &export_all_declaration) = 0;

  R VisitDeclaration(const JsDeclaration &declaration) {
    const JsDeclaration *declaration_ptr = &declaration;
    if (const JsVariableDeclaration *variable_declaration =
            dynamic_cast<const JsVariableDeclaration *>(declaration_ptr)) {
      return VisitVariableDeclaration(*variable_declaration);
    } else if (const JsFunctionDeclaration *function_declaration =
                   dynamic_cast<const JsFunctionDeclaration *>(
                       declaration_ptr)) {
      return VisitFunctionDeclaration(*function_declaration);
    } else if (const JsClassDeclaration *class_declaration =
                   dynamic_cast<const JsClassDeclaration *>(declaration_ptr)) {
      return VisitClassDeclaration(*class_declaration);
    }
  }

  virtual R VisitStatement(const JsStatement &statement) {
    const JsStatement *statement_ptr = &statement;
    if (const JsBlockStatement *block_statement =
            dynamic_cast<const JsBlockStatement *>(statement_ptr)) {
      return VisitBlockStatement(*block_statement);
    } else if (const JsExpressionStatement *expression_statement =
                   dynamic_cast<const JsExpressionStatement *>(statement_ptr)) {
      return VisitExpressionStatement(*expression_statement);
    } else if (const JsEmptyStatement *empty_statement =
                   dynamic_cast<const JsEmptyStatement *>(statement_ptr)) {
      return VisitEmptyStatement(*empty_statement);
    } else if (const JsDebuggerStatement *debugger_statement =
                   dynamic_cast<const JsDebuggerStatement *>(statement_ptr)) {
      return VisitDebuggerStatement(*debugger_statement);
    } else if (const JsWithStatement *with_statement =
                   dynamic_cast<const JsWithStatement *>(statement_ptr)) {
      return VisitWithStatement(*with_statement);
    } else if (const JsReturnStatement *return_statement =
                   dynamic_cast<const JsReturnStatement *>(statement_ptr)) {
      return VisitReturnStatement(*return_statement);
    } else if (const JsLabeledStatement *labeled_statement =
                   dynamic_cast<const JsLabeledStatement *>(statement_ptr)) {
      return VisitLabeledStatement(*labeled_statement);
    } else if (const JsBreakStatement *break_statement =
                   dynamic_cast<const JsBreakStatement *>(statement_ptr)) {
      return VisitBreakStatement(*break_statement);
    } else if (const JsContinueStatement *continue_statement =
                   dynamic_cast<const JsContinueStatement *>(statement_ptr)) {
      return VisitContinueStatement(*continue_statement);
    } else if (const JsIfStatement *if_statement =
                   dynamic_cast<const JsIfStatement *>(statement_ptr)) {
      return VisitIfStatement(*if_statement);
    } else if (const JsSwitchStatement *switch_statement =
                   dynamic_cast<const JsSwitchStatement *>(statement_ptr)) {
      return VisitSwitchStatement(*switch_statement);
    } else if (const JsThrowStatement *throw_statement =
                   dynamic_cast<const JsThrowStatement *>(statement_ptr)) {
      return VisitThrowStatement(*throw_statement);
    } else if (const JsTryStatement *try_statement =
                   dynamic_cast<const JsTryStatement *>(statement_ptr)) {
      return VisitTryStatement(*try_statement);
    } else if (const JsWhileStatement *while_statement =
                   dynamic_cast<const JsWhileStatement *>(statement_ptr)) {
      return VisitWhileStatement(*while_statement);
    } else if (const JsDoWhileStatement *do_while_statement =
                   dynamic_cast<const JsDoWhileStatement *>(statement_ptr)) {
      return VisitDoWhileStatement(*do_while_statement);
    } else if (const JsDeclaration *declaration =
                   dynamic_cast<const JsDeclaration *>(statement_ptr)) {
      return VisitDeclaration(*declaration);
    } else if (const JsForStatement *for_statement =
                   dynamic_cast<const JsForStatement *>(statement_ptr)) {
      return VisitForStatement(*for_statement);
    } else if (const JsForInStatement *for_in_statement =
                   dynamic_cast<const JsForInStatement *>(statement_ptr)) {
      return VisitForInStatement(*for_in_statement);
    } else if (const JsForOfStatement *for_of_statement =
                   dynamic_cast<const JsForOfStatement *>(statement_ptr)) {
      return VisitForOfStatement(*for_of_statement);
    }
  }

  R VisitModuleDeclaration(const JsModuleDeclaration &module_declaration) {
    const JsModuleDeclaration *module_declaration_ptr = &module_declaration;
    if (const JsImportDeclaration *import_declaration =
            dynamic_cast<const JsImportDeclaration *>(module_declaration_ptr)) {
      return VisitImportDeclaration(*import_declaration);
    } else if (const JsExportNamedDeclaration *export_named_declaration =
                   dynamic_cast<const JsExportNamedDeclaration *>(
                       module_declaration_ptr)) {
      return VisitExportNamedDeclaration(*export_named_declaration);
    } else if (const JsExportDefaultDeclaration *export_default_declaration =
                   dynamic_cast<const JsExportDefaultDeclaration *>(
                       module_declaration_ptr)) {
      return VisitExportDefaultDeclaration(*export_default_declaration);
    } else if (const JsExportAllDeclaration *export_all_declaration =
                   dynamic_cast<const JsExportAllDeclaration *>(
                       module_declaration_ptr)) {
      return VisitExportAllDeclaration(*export_all_declaration);
    }
  }

  R VisitLiteral(const JsLiteral &literal) {
    const JsLiteral *literal_ptr = &literal;
    if (const JsRegExpLiteral *reg_exp_literal =
            dynamic_cast<const JsRegExpLiteral *>(literal_ptr)) {
      return VisitRegExpLiteral(*reg_exp_literal);
    } else if (const JsNullLiteral *null_literal =
                   dynamic_cast<const JsNullLiteral *>(literal_ptr)) {
      return VisitNullLiteral(*null_literal);
    } else if (const JsStringLiteral *string_literal =
                   dynamic_cast<const JsStringLiteral *>(literal_ptr)) {
      return VisitStringLiteral(*string_literal);
    } else if (const JsBooleanLiteral *boolean_literal =
                   dynamic_cast<const JsBooleanLiteral *>(literal_ptr)) {
      return VisitBooleanLiteral(*boolean_literal);
    } else if (const JsNumericLiteral *numeric_literal =
                   dynamic_cast<const JsNumericLiteral *>(literal_ptr)) {
      return VisitNumericLiteral(*numeric_literal);
    } else if (const JsBigIntLiteral *big_int_literal =
                   dynamic_cast<const JsBigIntLiteral *>(literal_ptr)) {
      return VisitBigIntLiteral(*big_int_literal);
    }
  }

  R VisitExpression(const JsExpression &expression) {
    const JsExpression *expression_ptr = &expression;
    if (const JsIdentifier *identifier =
            dynamic_cast<const JsIdentifier *>(expression_ptr)) {
      return VisitIdentifier(*identifier);
    } else if (const JsLiteral *literal =
                   dynamic_cast<const JsLiteral *>(expression_ptr)) {
      return VisitLiteral(*literal);
    } else if (const JsThisExpression *this_expression =
                   dynamic_cast<const JsThisExpression *>(expression_ptr)) {
      return VisitThisExpression(*this_expression);
    } else if (const JsArrowFunctionExpression *arrow_function_expression =
                   dynamic_cast<const JsArrowFunctionExpression *>(
                       expression_ptr)) {
      return VisitArrowFunctionExpression(*arrow_function_expression);
    } else if (const JsYieldExpression *yield_expression =
                   dynamic_cast<const JsYieldExpression *>(expression_ptr)) {
      return VisitYieldExpression(*yield_expression);
    } else if (const JsAwaitExpression *await_expression =
                   dynamic_cast<const JsAwaitExpression *>(expression_ptr)) {
      return VisitAwaitExpression(*await_expression);
    } else if (const JsArrayExpression *array_expression =
                   dynamic_cast<const JsArrayExpression *>(expression_ptr)) {
      return VisitArrayExpression(*array_expression);
    } else if (const JsObjectExpression *object_expression =
                   dynamic_cast<const JsObjectExpression *>(expression_ptr)) {
      return VisitObjectExpression(*object_expression);
    } else if (const JsFunctionExpression *function_expression =
                   dynamic_cast<const JsFunctionExpression *>(expression_ptr)) {
      return VisitFunctionExpression(*function_expression);
    } else if (const JsUnaryExpression *unary_expression =
                   dynamic_cast<const JsUnaryExpression *>(expression_ptr)) {
      return VisitUnaryExpression(*unary_expression);
    } else if (const JsUpdateExpression *update_expression =
                   dynamic_cast<const JsUpdateExpression *>(expression_ptr)) {
      return VisitUpdateExpression(*update_expression);
    } else if (const JsBinaryExpression *binary_expression =
                   dynamic_cast<const JsBinaryExpression *>(expression_ptr)) {
      return VisitBinaryExpression(*binary_expression);
    } else if (const JsAssignmentExpression *assignment_expression =
                   dynamic_cast<const JsAssignmentExpression *>(
                       expression_ptr)) {
      return VisitAssignmentExpression(*assignment_expression);
    } else if (const JsLogicalExpression *logical_expression =
                   dynamic_cast<const JsLogicalExpression *>(expression_ptr)) {
      return VisitLogicalExpression(*logical_expression);
    } else if (const JsMemberExpression *member_expression =
                   dynamic_cast<const JsMemberExpression *>(expression_ptr)) {
      return VisitMemberExpression(*member_expression);
    } else if (const JsOptionalMemberExpression *optional_member_expression =
                   dynamic_cast<const JsOptionalMemberExpression *>(
                       expression_ptr)) {
      return VisitOptionalMemberExpression(*optional_member_expression);
    } else if (const JsConditionalExpression *conditional_expression =
                   dynamic_cast<const JsConditionalExpression *>(
                       expression_ptr)) {
      return VisitConditionalExpression(*conditional_expression);
    } else if (const JsCallExpression *call_expression =
                   dynamic_cast<const JsCallExpression *>(expression_ptr)) {
      return VisitCallExpression(*call_expression);
    } else if (const JsOptionalCallExpression *optional_call_expression =
                   dynamic_cast<const JsOptionalCallExpression *>(
                       expression_ptr)) {
      return VisitOptionalCallExpression(*optional_call_expression);
    } else if (const JsNewExpression *new_expression =
                   dynamic_cast<const JsNewExpression *>(expression_ptr)) {
      return VisitNewExpression(*new_expression);
    } else if (const JsSequenceExpression *sequence_expression =
                   dynamic_cast<const JsSequenceExpression *>(expression_ptr)) {
      return VisitSequenceExpression(*sequence_expression);
    } else if (const JsParenthesizedExpression *parenthesized_expression =
                   dynamic_cast<const JsParenthesizedExpression *>(
                       expression_ptr)) {
      return VisitParenthesizedExpression(*parenthesized_expression);
    } else if (const JsTemplateLiteral *template_literal =
                   dynamic_cast<const JsTemplateLiteral *>(expression_ptr)) {
      return VisitTemplateLiteral(*template_literal);
    } else if (const JsTaggedTemplateExpression *tagged_template_expression =
                   dynamic_cast<const JsTaggedTemplateExpression *>(
                       expression_ptr)) {
      return VisitTaggedTemplateExpression(*tagged_template_expression);
    } else if (const JsClassExpression *class_expression =
                   dynamic_cast<const JsClassExpression *>(expression_ptr)) {
      return VisitClassExpression(*class_expression);
    } else if (const JsMetaProperty *meta_property =
                   dynamic_cast<const JsMetaProperty *>(expression_ptr)) {
      return VisitMetaProperty(*meta_property);
    }
  }
};

// A mutable version of the AST visitor.
// This API can be useful when a client needs to modify the AST.
template <typename R>
class MutableJsAstVisitor {
 public:
  virtual ~MutableJsAstVisitor() = default;

  virtual R VisitInterpreterDirective(
      JsInterpreterDirective &interpreter_directive) = 0;

  virtual R VisitDirectiveLiteral(JsDirectiveLiteral &directive_literal) = 0;

  virtual R VisitDirective(JsDirective &directive) = 0;

  virtual R VisitProgram(JsProgram &program) = 0;

  virtual R VisitFile(JsFile &file) = 0;

  virtual R VisitIdentifier(JsIdentifier &identifier) = 0;

  virtual R VisitPrivateName(JsPrivateName &private_name) = 0;

  virtual R VisitRegExpLiteral(JsRegExpLiteral &reg_exp_literal) = 0;

  virtual R VisitNullLiteral(JsNullLiteral &null_literal) = 0;

  virtual R VisitStringLiteral(JsStringLiteral &string_literal) = 0;

  virtual R VisitBooleanLiteral(JsBooleanLiteral &boolean_literal) = 0;

  virtual R VisitNumericLiteral(JsNumericLiteral &numeric_literal) = 0;

  virtual R VisitBigIntLiteral(JsBigIntLiteral &big_int_literal) = 0;

  virtual R VisitBlockStatement(JsBlockStatement &block_statement) = 0;

  virtual R VisitExpressionStatement(
      JsExpressionStatement &expression_statement) = 0;

  virtual R VisitEmptyStatement(JsEmptyStatement &empty_statement) = 0;

  virtual R VisitDebuggerStatement(JsDebuggerStatement &debugger_statement) = 0;

  virtual R VisitWithStatement(JsWithStatement &with_statement) = 0;

  virtual R VisitReturnStatement(JsReturnStatement &return_statement) = 0;

  virtual R VisitLabeledStatement(JsLabeledStatement &labeled_statement) = 0;

  virtual R VisitBreakStatement(JsBreakStatement &break_statement) = 0;

  virtual R VisitContinueStatement(JsContinueStatement &continue_statement) = 0;

  virtual R VisitIfStatement(JsIfStatement &if_statement) = 0;

  virtual R VisitSwitchCase(JsSwitchCase &switch_case) = 0;

  virtual R VisitSwitchStatement(JsSwitchStatement &switch_statement) = 0;

  virtual R VisitThrowStatement(JsThrowStatement &throw_statement) = 0;

  virtual R VisitCatchClause(JsCatchClause &catch_clause) = 0;

  virtual R VisitTryStatement(JsTryStatement &try_statement) = 0;

  virtual R VisitWhileStatement(JsWhileStatement &while_statement) = 0;

  virtual R VisitDoWhileStatement(JsDoWhileStatement &do_while_statement) = 0;

  virtual R VisitVariableDeclarator(
      JsVariableDeclarator &variable_declarator) = 0;

  virtual R VisitVariableDeclaration(
      JsVariableDeclaration &variable_declaration) = 0;

  virtual R VisitForStatement(JsForStatement &for_statement) = 0;

  virtual R VisitForInStatement(JsForInStatement &for_in_statement) = 0;

  virtual R VisitForOfStatement(JsForOfStatement &for_of_statement) = 0;

  virtual R VisitFunctionDeclaration(
      JsFunctionDeclaration &function_declaration) = 0;

  virtual R VisitSuper(JsSuper &super) = 0;

  virtual R VisitImport(JsImport &import) = 0;

  virtual R VisitThisExpression(JsThisExpression &this_expression) = 0;

  virtual R VisitArrowFunctionExpression(
      JsArrowFunctionExpression &arrow_function_expression) = 0;

  virtual R VisitYieldExpression(JsYieldExpression &yield_expression) = 0;

  virtual R VisitAwaitExpression(JsAwaitExpression &await_expression) = 0;

  virtual R VisitSpreadElement(JsSpreadElement &spread_element) = 0;

  virtual R VisitArrayExpression(JsArrayExpression &array_expression) = 0;

  virtual R VisitObjectProperty(JsObjectProperty &object_property) = 0;

  virtual R VisitObjectMethod(JsObjectMethod &object_method) = 0;

  virtual R VisitObjectExpression(JsObjectExpression &object_expression) = 0;

  virtual R VisitFunctionExpression(
      JsFunctionExpression &function_expression) = 0;

  virtual R VisitUnaryExpression(JsUnaryExpression &unary_expression) = 0;

  virtual R VisitUpdateExpression(JsUpdateExpression &update_expression) = 0;

  virtual R VisitBinaryExpression(JsBinaryExpression &binary_expression) = 0;

  virtual R VisitAssignmentExpression(
      JsAssignmentExpression &assignment_expression) = 0;

  virtual R VisitLogicalExpression(JsLogicalExpression &logical_expression) = 0;

  virtual R VisitMemberExpression(JsMemberExpression &member_expression) = 0;

  virtual R VisitOptionalMemberExpression(
      JsOptionalMemberExpression &optional_member_expression) = 0;

  virtual R VisitConditionalExpression(
      JsConditionalExpression &conditional_expression) = 0;

  virtual R VisitCallExpression(JsCallExpression &call_expression) = 0;

  virtual R VisitOptionalCallExpression(
      JsOptionalCallExpression &optional_call_expression) = 0;

  virtual R VisitNewExpression(JsNewExpression &new_expression) = 0;

  virtual R VisitSequenceExpression(
      JsSequenceExpression &sequence_expression) = 0;

  virtual R VisitParenthesizedExpression(
      JsParenthesizedExpression &parenthesized_expression) = 0;

  virtual R VisitTemplateElement(JsTemplateElement &template_element) = 0;

  virtual R VisitTemplateLiteral(JsTemplateLiteral &template_literal) = 0;

  virtual R VisitTaggedTemplateExpression(
      JsTaggedTemplateExpression &tagged_template_expression) = 0;

  virtual R VisitRestElement(JsRestElement &rest_element) = 0;

  virtual R VisitObjectPattern(JsObjectPattern &object_pattern) = 0;

  virtual R VisitArrayPattern(JsArrayPattern &array_pattern) = 0;

  virtual R VisitAssignmentPattern(JsAssignmentPattern &assignment_pattern) = 0;

  virtual R VisitClassMethod(JsClassMethod &class_method) = 0;

  virtual R VisitClassPrivateMethod(
      JsClassPrivateMethod &class_private_method) = 0;

  virtual R VisitClassProperty(JsClassProperty &class_property) = 0;

  virtual R VisitClassPrivateProperty(
      JsClassPrivateProperty &class_private_property) = 0;

  virtual R VisitClassBody(JsClassBody &class_body) = 0;

  virtual R VisitClassDeclaration(JsClassDeclaration &class_declaration) = 0;

  virtual R VisitClassExpression(JsClassExpression &class_expression) = 0;

  virtual R VisitMetaProperty(JsMetaProperty &meta_property) = 0;

  virtual R VisitImportSpecifier(JsImportSpecifier &import_specifier) = 0;

  virtual R VisitImportDefaultSpecifier(
      JsImportDefaultSpecifier &import_default_specifier) = 0;

  virtual R VisitImportNamespaceSpecifier(
      JsImportNamespaceSpecifier &import_namespace_specifier) = 0;

  virtual R VisitImportAttribute(JsImportAttribute &import_attribute) = 0;

  virtual R VisitImportDeclaration(JsImportDeclaration &import_declaration) = 0;

  virtual R VisitExportSpecifier(JsExportSpecifier &export_specifier) = 0;

  virtual R VisitExportNamedDeclaration(
      JsExportNamedDeclaration &export_named_declaration) = 0;

  virtual R VisitExportDefaultDeclaration(
      JsExportDefaultDeclaration &export_default_declaration) = 0;

  virtual R VisitExportAllDeclaration(
      JsExportAllDeclaration &export_all_declaration) = 0;

  R VisitDeclaration(JsDeclaration &declaration) {
    JsDeclaration *declaration_ptr = &declaration;
    if (JsVariableDeclaration *variable_declaration =
            dynamic_cast<JsVariableDeclaration *>(declaration_ptr)) {
      return VisitVariableDeclaration(*variable_declaration);
    } else if (JsFunctionDeclaration *function_declaration =
                   dynamic_cast<JsFunctionDeclaration *>(declaration_ptr)) {
      return VisitFunctionDeclaration(*function_declaration);
    } else if (JsClassDeclaration *class_declaration =
                   dynamic_cast<JsClassDeclaration *>(declaration_ptr)) {
      return VisitClassDeclaration(*class_declaration);
    }
  }

  virtual R VisitStatement(JsStatement &statement) {
    JsStatement *statement_ptr = &statement;
    if (JsBlockStatement *block_statement =
            dynamic_cast<JsBlockStatement *>(statement_ptr)) {
      return VisitBlockStatement(*block_statement);
    } else if (JsExpressionStatement *expression_statement =
                   dynamic_cast<JsExpressionStatement *>(statement_ptr)) {
      return VisitExpressionStatement(*expression_statement);
    } else if (JsEmptyStatement *empty_statement =
                   dynamic_cast<JsEmptyStatement *>(statement_ptr)) {
      return VisitEmptyStatement(*empty_statement);
    } else if (JsDebuggerStatement *debugger_statement =
                   dynamic_cast<JsDebuggerStatement *>(statement_ptr)) {
      return VisitDebuggerStatement(*debugger_statement);
    } else if (JsWithStatement *with_statement =
                   dynamic_cast<JsWithStatement *>(statement_ptr)) {
      return VisitWithStatement(*with_statement);
    } else if (JsReturnStatement *return_statement =
                   dynamic_cast<JsReturnStatement *>(statement_ptr)) {
      return VisitReturnStatement(*return_statement);
    } else if (JsLabeledStatement *labeled_statement =
                   dynamic_cast<JsLabeledStatement *>(statement_ptr)) {
      return VisitLabeledStatement(*labeled_statement);
    } else if (JsBreakStatement *break_statement =
                   dynamic_cast<JsBreakStatement *>(statement_ptr)) {
      return VisitBreakStatement(*break_statement);
    } else if (JsContinueStatement *continue_statement =
                   dynamic_cast<JsContinueStatement *>(statement_ptr)) {
      return VisitContinueStatement(*continue_statement);
    } else if (JsIfStatement *if_statement =
                   dynamic_cast<JsIfStatement *>(statement_ptr)) {
      return VisitIfStatement(*if_statement);
    } else if (JsSwitchStatement *switch_statement =
                   dynamic_cast<JsSwitchStatement *>(statement_ptr)) {
      return VisitSwitchStatement(*switch_statement);
    } else if (JsThrowStatement *throw_statement =
                   dynamic_cast<JsThrowStatement *>(statement_ptr)) {
      return VisitThrowStatement(*throw_statement);
    } else if (JsTryStatement *try_statement =
                   dynamic_cast<JsTryStatement *>(statement_ptr)) {
      return VisitTryStatement(*try_statement);
    } else if (JsWhileStatement *while_statement =
                   dynamic_cast<JsWhileStatement *>(statement_ptr)) {
      return VisitWhileStatement(*while_statement);
    } else if (JsDoWhileStatement *do_while_statement =
                   dynamic_cast<JsDoWhileStatement *>(statement_ptr)) {
      return VisitDoWhileStatement(*do_while_statement);
    } else if (JsDeclaration *declaration =
                   dynamic_cast<JsDeclaration *>(statement_ptr)) {
      return VisitDeclaration(*declaration);
    } else if (JsForStatement *for_statement =
                   dynamic_cast<JsForStatement *>(statement_ptr)) {
      return VisitForStatement(*for_statement);
    } else if (JsForInStatement *for_in_statement =
                   dynamic_cast<JsForInStatement *>(statement_ptr)) {
      return VisitForInStatement(*for_in_statement);
    } else if (JsForOfStatement *for_of_statement =
                   dynamic_cast<JsForOfStatement *>(statement_ptr)) {
      return VisitForOfStatement(*for_of_statement);
    }
  }

  R VisitModuleDeclaration(JsModuleDeclaration &module_declaration) {
    JsModuleDeclaration *module_declaration_ptr = &module_declaration;
    if (JsImportDeclaration *import_declaration =
            dynamic_cast<JsImportDeclaration *>(module_declaration_ptr)) {
      return VisitImportDeclaration(*import_declaration);
    } else if (JsExportNamedDeclaration *export_named_declaration =
                   dynamic_cast<JsExportNamedDeclaration *>(
                       module_declaration_ptr)) {
      return VisitExportNamedDeclaration(*export_named_declaration);
    } else if (JsExportDefaultDeclaration *export_default_declaration =
                   dynamic_cast<JsExportDefaultDeclaration *>(
                       module_declaration_ptr)) {
      return VisitExportDefaultDeclaration(*export_default_declaration);
    } else if (JsExportAllDeclaration *export_all_declaration =
                   dynamic_cast<JsExportAllDeclaration *>(
                       module_declaration_ptr)) {
      return VisitExportAllDeclaration(*export_all_declaration);
    }
  }

  R VisitLiteral(JsLiteral &literal) {
    JsLiteral *literal_ptr = &literal;
    if (JsRegExpLiteral *reg_exp_literal =
            dynamic_cast<JsRegExpLiteral *>(literal_ptr)) {
      return VisitRegExpLiteral(*reg_exp_literal);
    } else if (JsNullLiteral *null_literal =
                   dynamic_cast<JsNullLiteral *>(literal_ptr)) {
      return VisitNullLiteral(*null_literal);
    } else if (JsStringLiteral *string_literal =
                   dynamic_cast<JsStringLiteral *>(literal_ptr)) {
      return VisitStringLiteral(*string_literal);
    } else if (JsBooleanLiteral *boolean_literal =
                   dynamic_cast<JsBooleanLiteral *>(literal_ptr)) {
      return VisitBooleanLiteral(*boolean_literal);
    } else if (JsNumericLiteral *numeric_literal =
                   dynamic_cast<JsNumericLiteral *>(literal_ptr)) {
      return VisitNumericLiteral(*numeric_literal);
    } else if (JsBigIntLiteral *big_int_literal =
                   dynamic_cast<JsBigIntLiteral *>(literal_ptr)) {
      return VisitBigIntLiteral(*big_int_literal);
    }
  }

  R VisitExpression(JsExpression &expression) {
    JsExpression *expression_ptr = &expression;
    if (JsIdentifier *identifier =
            dynamic_cast<JsIdentifier *>(expression_ptr)) {
      return VisitIdentifier(*identifier);
    } else if (JsLiteral *literal = dynamic_cast<JsLiteral *>(expression_ptr)) {
      return VisitLiteral(*literal);
    } else if (JsThisExpression *this_expression =
                   dynamic_cast<JsThisExpression *>(expression_ptr)) {
      return VisitThisExpression(*this_expression);
    } else if (JsArrowFunctionExpression *arrow_function_expression =
                   dynamic_cast<JsArrowFunctionExpression *>(expression_ptr)) {
      return VisitArrowFunctionExpression(*arrow_function_expression);
    } else if (JsYieldExpression *yield_expression =
                   dynamic_cast<JsYieldExpression *>(expression_ptr)) {
      return VisitYieldExpression(*yield_expression);
    } else if (JsAwaitExpression *await_expression =
                   dynamic_cast<JsAwaitExpression *>(expression_ptr)) {
      return VisitAwaitExpression(*await_expression);
    } else if (JsArrayExpression *array_expression =
                   dynamic_cast<JsArrayExpression *>(expression_ptr)) {
      return VisitArrayExpression(*array_expression);
    } else if (JsObjectExpression *object_expression =
                   dynamic_cast<JsObjectExpression *>(expression_ptr)) {
      return VisitObjectExpression(*object_expression);
    } else if (JsFunctionExpression *function_expression =
                   dynamic_cast<JsFunctionExpression *>(expression_ptr)) {
      return VisitFunctionExpression(*function_expression);
    } else if (JsUnaryExpression *unary_expression =
                   dynamic_cast<JsUnaryExpression *>(expression_ptr)) {
      return VisitUnaryExpression(*unary_expression);
    } else if (JsUpdateExpression *update_expression =
                   dynamic_cast<JsUpdateExpression *>(expression_ptr)) {
      return VisitUpdateExpression(*update_expression);
    } else if (JsBinaryExpression *binary_expression =
                   dynamic_cast<JsBinaryExpression *>(expression_ptr)) {
      return VisitBinaryExpression(*binary_expression);
    } else if (JsAssignmentExpression *assignment_expression =
                   dynamic_cast<JsAssignmentExpression *>(expression_ptr)) {
      return VisitAssignmentExpression(*assignment_expression);
    } else if (JsLogicalExpression *logical_expression =
                   dynamic_cast<JsLogicalExpression *>(expression_ptr)) {
      return VisitLogicalExpression(*logical_expression);
    } else if (JsMemberExpression *member_expression =
                   dynamic_cast<JsMemberExpression *>(expression_ptr)) {
      return VisitMemberExpression(*member_expression);
    } else if (JsOptionalMemberExpression *optional_member_expression =
                   dynamic_cast<JsOptionalMemberExpression *>(expression_ptr)) {
      return VisitOptionalMemberExpression(*optional_member_expression);
    } else if (JsConditionalExpression *conditional_expression =
                   dynamic_cast<JsConditionalExpression *>(expression_ptr)) {
      return VisitConditionalExpression(*conditional_expression);
    } else if (JsCallExpression *call_expression =
                   dynamic_cast<JsCallExpression *>(expression_ptr)) {
      return VisitCallExpression(*call_expression);
    } else if (JsOptionalCallExpression *optional_call_expression =
                   dynamic_cast<JsOptionalCallExpression *>(expression_ptr)) {
      return VisitOptionalCallExpression(*optional_call_expression);
    } else if (JsNewExpression *new_expression =
                   dynamic_cast<JsNewExpression *>(expression_ptr)) {
      return VisitNewExpression(*new_expression);
    } else if (JsSequenceExpression *sequence_expression =
                   dynamic_cast<JsSequenceExpression *>(expression_ptr)) {
      return VisitSequenceExpression(*sequence_expression);
    } else if (JsParenthesizedExpression *parenthesized_expression =
                   dynamic_cast<JsParenthesizedExpression *>(expression_ptr)) {
      return VisitParenthesizedExpression(*parenthesized_expression);
    } else if (JsTemplateLiteral *template_literal =
                   dynamic_cast<JsTemplateLiteral *>(expression_ptr)) {
      return VisitTemplateLiteral(*template_literal);
    } else if (JsTaggedTemplateExpression *tagged_template_expression =
                   dynamic_cast<JsTaggedTemplateExpression *>(expression_ptr)) {
      return VisitTaggedTemplateExpression(*tagged_template_expression);
    } else if (JsClassExpression *class_expression =
                   dynamic_cast<JsClassExpression *>(expression_ptr)) {
      return VisitClassExpression(*class_expression);
    } else if (JsMetaProperty *meta_property =
                   dynamic_cast<JsMetaProperty *>(expression_ptr)) {
      return VisitMetaProperty(*meta_property);
    }
  }

  R VisitPattern(JsPattern &pattern) {
    JsPattern *pattern_ptr = &pattern;
    if (JsIdentifier *identifier = dynamic_cast<JsIdentifier *>(pattern_ptr)) {
      return VisitIdentifier(*identifier);
    } else if (JsMemberExpression *member_expression =
                   dynamic_cast<JsMemberExpression *>(pattern_ptr)) {
      return VisitMemberExpression(*member_expression);
    } else if (JsParenthesizedExpression *parenthesized_expression =
                   dynamic_cast<JsParenthesizedExpression *>(pattern_ptr)) {
      return VisitParenthesizedExpression(*parenthesized_expression);
    } else if (JsRestElement *rest_element =
                   dynamic_cast<JsRestElement *>(pattern_ptr)) {
      return VisitRestElement(*rest_element);
    } else if (JsObjectPattern *object_pattern =
                   dynamic_cast<JsObjectPattern *>(pattern_ptr)) {
      return VisitObjectPattern(*object_pattern);
    } else if (JsArrayPattern *array_pattern =
                   dynamic_cast<JsArrayPattern *>(pattern_ptr)) {
      return VisitArrayPattern(*array_pattern);
    } else if (JsAssignmentPattern *assignment_pattern =
                   dynamic_cast<JsAssignmentPattern *>(pattern_ptr)) {
      return VisitAssignmentPattern(*assignment_pattern);
    }
  }

  R VisitLVal(JsLVal &l_val) {
    JsLVal *l_val_ptr = &l_val;
    if (JsIdentifier *identifier = dynamic_cast<JsIdentifier *>(l_val_ptr)) {
      return VisitIdentifier(*identifier);
    } else if (JsMemberExpression *member_expression =
                   dynamic_cast<JsMemberExpression *>(l_val_ptr)) {
      return VisitMemberExpression(*member_expression);
    } else if (JsParenthesizedExpression *parenthesized_expression =
                   dynamic_cast<JsParenthesizedExpression *>(l_val_ptr)) {
      return VisitParenthesizedExpression(*parenthesized_expression);
    } else if (JsRestElement *rest_element =
                   dynamic_cast<JsRestElement *>(l_val_ptr)) {
      return VisitRestElement(*rest_element);
    } else if (JsObjectPattern *object_pattern =
                   dynamic_cast<JsObjectPattern *>(l_val_ptr)) {
      return VisitObjectPattern(*object_pattern);
    } else if (JsArrayPattern *array_pattern =
                   dynamic_cast<JsArrayPattern *>(l_val_ptr)) {
      return VisitArrayPattern(*array_pattern);
    } else if (JsAssignmentPattern *assignment_pattern =
                   dynamic_cast<JsAssignmentPattern *>(l_val_ptr)) {
      return VisitAssignmentPattern(*assignment_pattern);
    }
  }
};

class EmptyJsAstVisitor : public JsAstVisitor<void> {
 public:
  ~EmptyJsAstVisitor() override = default;

  void VisitInterpreterDirective(
      const JsInterpreterDirective &interpreter_directive) override {}

  void VisitDirectiveLiteral(
      const JsDirectiveLiteral &directive_literal) override {}

  void VisitDirective(const JsDirective &directive) override {}

  void VisitProgram(const JsProgram &program) override {}

  void VisitFile(const JsFile &file) override {}

  void VisitIdentifier(const JsIdentifier &identifier) override {}

  void VisitPrivateName(const JsPrivateName &private_name) override {}

  void VisitRegExpLiteral(const JsRegExpLiteral &reg_exp_literal) override {}

  void VisitNullLiteral(const JsNullLiteral &null_literal) override {}

  void VisitStringLiteral(const JsStringLiteral &string_literal) override {}

  void VisitBooleanLiteral(const JsBooleanLiteral &boolean_literal) override {}

  void VisitNumericLiteral(const JsNumericLiteral &numeric_literal) override {}

  void VisitBigIntLiteral(const JsBigIntLiteral &big_int_literal) override {}

  void VisitBlockStatement(const JsBlockStatement &block_statement) override {}

  void VisitExpressionStatement(
      const JsExpressionStatement &expression_statement) override {}

  void VisitEmptyStatement(const JsEmptyStatement &empty_statement) override {}

  void VisitDebuggerStatement(
      const JsDebuggerStatement &debugger_statement) override {}

  void VisitWithStatement(const JsWithStatement &with_statement) override {}

  void VisitReturnStatement(
      const JsReturnStatement &return_statement) override {}

  void VisitLabeledStatement(
      const JsLabeledStatement &labeled_statement) override {}

  void VisitBreakStatement(const JsBreakStatement &break_statement) override {}

  void VisitContinueStatement(
      const JsContinueStatement &continue_statement) override {}

  void VisitIfStatement(const JsIfStatement &if_statement) override {}

  void VisitSwitchCase(const JsSwitchCase &switch_case) override {}

  void VisitSwitchStatement(
      const JsSwitchStatement &switch_statement) override {}

  void VisitThrowStatement(const JsThrowStatement &throw_statement) override {}

  void VisitCatchClause(const JsCatchClause &catch_clause) override {}

  void VisitTryStatement(const JsTryStatement &try_statement) override {}

  void VisitWhileStatement(const JsWhileStatement &while_statement) override {}

  void VisitDoWhileStatement(
      const JsDoWhileStatement &do_while_statement) override {}

  void VisitVariableDeclarator(
      const JsVariableDeclarator &variable_declarator) override {}

  void VisitVariableDeclaration(
      const JsVariableDeclaration &variable_declaration) override {}

  void VisitForStatement(const JsForStatement &for_statement) override {}

  void VisitForInStatement(const JsForInStatement &for_in_statement) override {}

  void VisitForOfStatement(const JsForOfStatement &for_of_statement) override {}

  void VisitFunctionDeclaration(
      const JsFunctionDeclaration &function_declaration) override {}

  void VisitSuper(const JsSuper &super) override {}

  void VisitImport(const JsImport &import) override {}

  void VisitThisExpression(const JsThisExpression &this_expression) override {}

  void VisitArrowFunctionExpression(
      const JsArrowFunctionExpression &arrow_function_expression) override {}

  void VisitYieldExpression(
      const JsYieldExpression &yield_expression) override {}

  void VisitAwaitExpression(
      const JsAwaitExpression &await_expression) override {}

  void VisitSpreadElement(const JsSpreadElement &spread_element) override {}

  void VisitArrayExpression(
      const JsArrayExpression &array_expression) override {}

  void VisitObjectProperty(const JsObjectProperty &object_property) override {}

  void VisitObjectMethod(const JsObjectMethod &object_method) override {}

  void VisitObjectExpression(
      const JsObjectExpression &object_expression) override {}

  void VisitFunctionExpression(
      const JsFunctionExpression &function_expression) override {}

  void VisitUnaryExpression(
      const JsUnaryExpression &unary_expression) override {}

  void VisitUpdateExpression(
      const JsUpdateExpression &update_expression) override {}

  void VisitBinaryExpression(
      const JsBinaryExpression &binary_expression) override {}

  void VisitAssignmentExpression(
      const JsAssignmentExpression &assignment_expression) override {}

  void VisitLogicalExpression(
      const JsLogicalExpression &logical_expression) override {}

  void VisitMemberExpression(
      const JsMemberExpression &member_expression) override {}

  void VisitOptionalMemberExpression(
      const JsOptionalMemberExpression &optional_member_expression) override {}

  void VisitConditionalExpression(
      const JsConditionalExpression &conditional_expression) override {}

  void VisitCallExpression(const JsCallExpression &call_expression) override {}

  void VisitOptionalCallExpression(
      const JsOptionalCallExpression &optional_call_expression) override {}

  void VisitNewExpression(const JsNewExpression &new_expression) override {}

  void VisitSequenceExpression(
      const JsSequenceExpression &sequence_expression) override {}

  void VisitParenthesizedExpression(
      const JsParenthesizedExpression &parenthesized_expression) override {}

  void VisitTemplateElement(
      const JsTemplateElement &template_element) override {}

  void VisitTemplateLiteral(
      const JsTemplateLiteral &template_literal) override {}

  void VisitTaggedTemplateExpression(
      const JsTaggedTemplateExpression &tagged_template_expression) override {}

  void VisitRestElement(const JsRestElement &rest_element) override {}

  void VisitObjectPattern(const JsObjectPattern &object_pattern) override {}

  void VisitArrayPattern(const JsArrayPattern &array_pattern) override {}

  void VisitAssignmentPattern(
      const JsAssignmentPattern &assignment_pattern) override {}

  void VisitClassMethod(const JsClassMethod &class_method) override {}

  void VisitClassPrivateMethod(
      const JsClassPrivateMethod &class_private_method) override {}

  void VisitClassProperty(const JsClassProperty &class_property) override {}

  void VisitClassPrivateProperty(
      const JsClassPrivateProperty &class_private_property) override {}

  void VisitClassBody(const JsClassBody &class_body) override {}

  void VisitClassDeclaration(
      const JsClassDeclaration &class_declaration) override {}

  void VisitClassExpression(
      const JsClassExpression &class_expression) override {}

  void VisitMetaProperty(const JsMetaProperty &meta_property) override {}

  void VisitImportSpecifier(
      const JsImportSpecifier &import_specifier) override {}

  void VisitImportDefaultSpecifier(
      const JsImportDefaultSpecifier &import_default_specifier) override {}

  void VisitImportNamespaceSpecifier(
      const JsImportNamespaceSpecifier &import_namespace_specifier) override {}

  void VisitImportAttribute(
      const JsImportAttribute &import_attribute) override {}

  void VisitImportDeclaration(
      const JsImportDeclaration &import_declaration) override {}

  void VisitExportSpecifier(
      const JsExportSpecifier &export_specifier) override {}

  void VisitExportNamedDeclaration(
      const JsExportNamedDeclaration &export_named_declaration) override {}

  void VisitExportDefaultDeclaration(
      const JsExportDefaultDeclaration &export_default_declaration) override {}

  void VisitExportAllDeclaration(
      const JsExportAllDeclaration &export_all_declaration) override {}
};

// Redirects every Visit* method to a single VisitNodeDefault() method. This
// provides a simple way of defining a callback for AstWalker.
//
// Example:
//
// ```
// class MyJsVisitor : public DefaultJsAstVisitor {
//   void VisitNodeDefault(const JsNode& node) override {
//     ...
//     // Do something.
//   }
// };
//
// JsFile file = ...;
// MyJsVisitor visitor;
// JsAstWalker walker(&visitor, /*preorder_callback=*/&visitor);
// walker.VisitFile(file);
// ```
class DefaultJsAstVisitor : public EmptyJsAstVisitor {
 public:
  ~DefaultJsAstVisitor() override = default;

  void VisitInterpreterDirective(
      const JsInterpreterDirective &interpreter_directive) override {
    this->VisitNodeDefault(interpreter_directive);
  }

  void VisitDirectiveLiteral(
      const JsDirectiveLiteral &directive_literal) override {
    this->VisitNodeDefault(directive_literal);
  }

  void VisitDirective(const JsDirective &directive) override {
    this->VisitNodeDefault(directive);
  }

  void VisitProgram(const JsProgram &program) override {
    this->VisitNodeDefault(program);
  }

  void VisitFile(const JsFile &file) override { this->VisitNodeDefault(file); }

  void VisitIdentifier(const JsIdentifier &identifier) override {
    this->VisitNodeDefault(identifier);
  }

  void VisitPrivateName(const JsPrivateName &private_name) override {
    this->VisitNodeDefault(private_name);
  }

  void VisitRegExpLiteral(const JsRegExpLiteral &reg_exp_literal) override {
    this->VisitNodeDefault(reg_exp_literal);
  }

  void VisitNullLiteral(const JsNullLiteral &null_literal) override {
    this->VisitNodeDefault(null_literal);
  }

  void VisitStringLiteral(const JsStringLiteral &string_literal) override {
    this->VisitNodeDefault(string_literal);
  }

  void VisitBooleanLiteral(const JsBooleanLiteral &boolean_literal) override {
    this->VisitNodeDefault(boolean_literal);
  }

  void VisitNumericLiteral(const JsNumericLiteral &numeric_literal) override {
    this->VisitNodeDefault(numeric_literal);
  }

  void VisitBigIntLiteral(const JsBigIntLiteral &big_int_literal) override {
    this->VisitNodeDefault(big_int_literal);
  }

  void VisitBlockStatement(const JsBlockStatement &block_statement) override {
    this->VisitNodeDefault(block_statement);
  }

  void VisitExpressionStatement(
      const JsExpressionStatement &expression_statement) override {
    this->VisitNodeDefault(expression_statement);
  }

  void VisitEmptyStatement(const JsEmptyStatement &empty_statement) override {
    this->VisitNodeDefault(empty_statement);
  }

  void VisitDebuggerStatement(
      const JsDebuggerStatement &debugger_statement) override {
    this->VisitNodeDefault(debugger_statement);
  }

  void VisitWithStatement(const JsWithStatement &with_statement) override {
    this->VisitNodeDefault(with_statement);
  }

  void VisitReturnStatement(
      const JsReturnStatement &return_statement) override {
    this->VisitNodeDefault(return_statement);
  }

  void VisitLabeledStatement(
      const JsLabeledStatement &labeled_statement) override {
    this->VisitNodeDefault(labeled_statement);
  }

  void VisitBreakStatement(const JsBreakStatement &break_statement) override {
    this->VisitNodeDefault(break_statement);
  }

  void VisitContinueStatement(
      const JsContinueStatement &continue_statement) override {
    this->VisitNodeDefault(continue_statement);
  }

  void VisitIfStatement(const JsIfStatement &if_statement) override {
    this->VisitNodeDefault(if_statement);
  }

  void VisitSwitchCase(const JsSwitchCase &switch_case) override {
    this->VisitNodeDefault(switch_case);
  }

  void VisitSwitchStatement(
      const JsSwitchStatement &switch_statement) override {
    this->VisitNodeDefault(switch_statement);
  }

  void VisitThrowStatement(const JsThrowStatement &throw_statement) override {
    this->VisitNodeDefault(throw_statement);
  }

  void VisitCatchClause(const JsCatchClause &catch_clause) override {
    this->VisitNodeDefault(catch_clause);
  }

  void VisitTryStatement(const JsTryStatement &try_statement) override {
    this->VisitNodeDefault(try_statement);
  }

  void VisitWhileStatement(const JsWhileStatement &while_statement) override {
    this->VisitNodeDefault(while_statement);
  }

  void VisitDoWhileStatement(
      const JsDoWhileStatement &do_while_statement) override {
    this->VisitNodeDefault(do_while_statement);
  }

  void VisitVariableDeclarator(
      const JsVariableDeclarator &variable_declarator) override {
    this->VisitNodeDefault(variable_declarator);
  }

  void VisitVariableDeclaration(
      const JsVariableDeclaration &variable_declaration) override {
    this->VisitNodeDefault(variable_declaration);
  }

  void VisitForStatement(const JsForStatement &for_statement) override {
    this->VisitNodeDefault(for_statement);
  }

  void VisitForInStatement(const JsForInStatement &for_in_statement) override {
    this->VisitNodeDefault(for_in_statement);
  }

  void VisitForOfStatement(const JsForOfStatement &for_of_statement) override {
    this->VisitNodeDefault(for_of_statement);
  }

  void VisitFunctionDeclaration(
      const JsFunctionDeclaration &function_declaration) override {
    this->VisitNodeDefault(function_declaration);
  }

  void VisitSuper(const JsSuper &super) override {
    this->VisitNodeDefault(super);
  }

  void VisitImport(const JsImport &import) override {
    this->VisitNodeDefault(import);
  }

  void VisitThisExpression(const JsThisExpression &this_expression) override {
    this->VisitNodeDefault(this_expression);
  }

  void VisitArrowFunctionExpression(
      const JsArrowFunctionExpression &arrow_function_expression) override {
    this->VisitNodeDefault(arrow_function_expression);
  }

  void VisitYieldExpression(
      const JsYieldExpression &yield_expression) override {
    this->VisitNodeDefault(yield_expression);
  }

  void VisitAwaitExpression(
      const JsAwaitExpression &await_expression) override {
    this->VisitNodeDefault(await_expression);
  }

  void VisitSpreadElement(const JsSpreadElement &spread_element) override {
    this->VisitNodeDefault(spread_element);
  }

  void VisitArrayExpression(
      const JsArrayExpression &array_expression) override {
    this->VisitNodeDefault(array_expression);
  }

  void VisitObjectProperty(const JsObjectProperty &object_property) override {
    this->VisitNodeDefault(object_property);
  }

  void VisitObjectMethod(const JsObjectMethod &object_method) override {
    this->VisitNodeDefault(object_method);
  }

  void VisitObjectExpression(
      const JsObjectExpression &object_expression) override {
    this->VisitNodeDefault(object_expression);
  }

  void VisitFunctionExpression(
      const JsFunctionExpression &function_expression) override {
    this->VisitNodeDefault(function_expression);
  }

  void VisitUnaryExpression(
      const JsUnaryExpression &unary_expression) override {
    this->VisitNodeDefault(unary_expression);
  }

  void VisitUpdateExpression(
      const JsUpdateExpression &update_expression) override {
    this->VisitNodeDefault(update_expression);
  }

  void VisitBinaryExpression(
      const JsBinaryExpression &binary_expression) override {
    this->VisitNodeDefault(binary_expression);
  }

  void VisitAssignmentExpression(
      const JsAssignmentExpression &assignment_expression) override {
    this->VisitNodeDefault(assignment_expression);
  }

  void VisitLogicalExpression(
      const JsLogicalExpression &logical_expression) override {
    this->VisitNodeDefault(logical_expression);
  }

  void VisitMemberExpression(
      const JsMemberExpression &member_expression) override {
    this->VisitNodeDefault(member_expression);
  }

  void VisitOptionalMemberExpression(
      const JsOptionalMemberExpression &optional_member_expression) override {
    this->VisitNodeDefault(optional_member_expression);
  }

  void VisitConditionalExpression(
      const JsConditionalExpression &conditional_expression) override {
    this->VisitNodeDefault(conditional_expression);
  }

  void VisitCallExpression(const JsCallExpression &call_expression) override {
    this->VisitNodeDefault(call_expression);
  }

  void VisitOptionalCallExpression(
      const JsOptionalCallExpression &optional_call_expression) override {
    this->VisitNodeDefault(optional_call_expression);
  }

  void VisitNewExpression(const JsNewExpression &new_expression) override {
    this->VisitNodeDefault(new_expression);
  }

  void VisitSequenceExpression(
      const JsSequenceExpression &sequence_expression) override {
    this->VisitNodeDefault(sequence_expression);
  }

  void VisitParenthesizedExpression(
      const JsParenthesizedExpression &parenthesized_expression) override {
    this->VisitNodeDefault(parenthesized_expression);
  }

  void VisitTemplateElement(
      const JsTemplateElement &template_element) override {
    this->VisitNodeDefault(template_element);
  }

  void VisitTemplateLiteral(
      const JsTemplateLiteral &template_literal) override {
    this->VisitNodeDefault(template_literal);
  }

  void VisitTaggedTemplateExpression(
      const JsTaggedTemplateExpression &tagged_template_expression) override {
    this->VisitNodeDefault(tagged_template_expression);
  }

  void VisitRestElement(const JsRestElement &rest_element) override {
    this->VisitNodeDefault(rest_element);
  }

  void VisitObjectPattern(const JsObjectPattern &object_pattern) override {
    this->VisitNodeDefault(object_pattern);
  }

  void VisitArrayPattern(const JsArrayPattern &array_pattern) override {
    this->VisitNodeDefault(array_pattern);
  }

  void VisitAssignmentPattern(
      const JsAssignmentPattern &assignment_pattern) override {
    this->VisitNodeDefault(assignment_pattern);
  }

  void VisitClassMethod(const JsClassMethod &class_method) override {
    this->VisitNodeDefault(class_method);
  }

  void VisitClassPrivateMethod(
      const JsClassPrivateMethod &class_private_method) override {
    this->VisitNodeDefault(class_private_method);
  }

  void VisitClassProperty(const JsClassProperty &class_property) override {
    this->VisitNodeDefault(class_property);
  }

  void VisitClassPrivateProperty(
      const JsClassPrivateProperty &class_private_property) override {
    this->VisitNodeDefault(class_private_property);
  }

  void VisitClassBody(const JsClassBody &class_body) override {
    this->VisitNodeDefault(class_body);
  }

  void VisitClassDeclaration(
      const JsClassDeclaration &class_declaration) override {
    this->VisitNodeDefault(class_declaration);
  }

  void VisitClassExpression(
      const JsClassExpression &class_expression) override {
    this->VisitNodeDefault(class_expression);
  }

  void VisitMetaProperty(const JsMetaProperty &meta_property) override {
    this->VisitNodeDefault(meta_property);
  }

  void VisitImportSpecifier(
      const JsImportSpecifier &import_specifier) override {
    this->VisitNodeDefault(import_specifier);
  }

  void VisitImportDefaultSpecifier(
      const JsImportDefaultSpecifier &import_default_specifier) override {
    this->VisitNodeDefault(import_default_specifier);
  }

  void VisitImportNamespaceSpecifier(
      const JsImportNamespaceSpecifier &import_namespace_specifier) override {
    this->VisitNodeDefault(import_namespace_specifier);
  }

  void VisitImportAttribute(
      const JsImportAttribute &import_attribute) override {
    this->VisitNodeDefault(import_attribute);
  }

  void VisitImportDeclaration(
      const JsImportDeclaration &import_declaration) override {
    this->VisitNodeDefault(import_declaration);
  }

  void VisitExportSpecifier(
      const JsExportSpecifier &export_specifier) override {
    this->VisitNodeDefault(export_specifier);
  }

  void VisitExportNamedDeclaration(
      const JsExportNamedDeclaration &export_named_declaration) override {
    this->VisitNodeDefault(export_named_declaration);
  }

  void VisitExportDefaultDeclaration(
      const JsExportDefaultDeclaration &export_default_declaration) override {
    this->VisitNodeDefault(export_default_declaration);
  }

  void VisitExportAllDeclaration(
      const JsExportAllDeclaration &export_all_declaration) override {
    this->VisitNodeDefault(export_all_declaration);
  }

  // override this method to customize the default behavior on each JS AST node.
  virtual void VisitNodeDefault(const JsNode &node) = 0;
};

class EmptyMutableJsAstVisitor : public MutableJsAstVisitor<void> {
 public:
  ~EmptyMutableJsAstVisitor() override = default;

  void VisitInterpreterDirective(
      JsInterpreterDirective &interpreter_directive) override {}

  void VisitDirectiveLiteral(JsDirectiveLiteral &directive_literal) override {}

  void VisitDirective(JsDirective &directive) override {}

  void VisitProgram(JsProgram &program) override {}

  void VisitFile(JsFile &file) override {}

  void VisitIdentifier(JsIdentifier &identifier) override {}

  void VisitPrivateName(JsPrivateName &private_name) override {}

  void VisitRegExpLiteral(JsRegExpLiteral &reg_exp_literal) override {}

  void VisitNullLiteral(JsNullLiteral &null_literal) override {}

  void VisitStringLiteral(JsStringLiteral &string_literal) override {}

  void VisitBooleanLiteral(JsBooleanLiteral &boolean_literal) override {}

  void VisitNumericLiteral(JsNumericLiteral &numeric_literal) override {}

  void VisitBigIntLiteral(JsBigIntLiteral &big_int_literal) override {}

  void VisitBlockStatement(JsBlockStatement &block_statement) override {}

  void VisitExpressionStatement(
      JsExpressionStatement &expression_statement) override {}

  void VisitEmptyStatement(JsEmptyStatement &empty_statement) override {}

  void VisitDebuggerStatement(
      JsDebuggerStatement &debugger_statement) override {}

  void VisitWithStatement(JsWithStatement &with_statement) override {}

  void VisitReturnStatement(JsReturnStatement &return_statement) override {}

  void VisitLabeledStatement(JsLabeledStatement &labeled_statement) override {}

  void VisitBreakStatement(JsBreakStatement &break_statement) override {}

  void VisitContinueStatement(
      JsContinueStatement &continue_statement) override {}

  void VisitIfStatement(JsIfStatement &if_statement) override {}

  void VisitSwitchCase(JsSwitchCase &switch_case) override {}

  void VisitSwitchStatement(JsSwitchStatement &switch_statement) override {}

  void VisitThrowStatement(JsThrowStatement &throw_statement) override {}

  void VisitCatchClause(JsCatchClause &catch_clause) override {}

  void VisitTryStatement(JsTryStatement &try_statement) override {}

  void VisitWhileStatement(JsWhileStatement &while_statement) override {}

  void VisitDoWhileStatement(JsDoWhileStatement &do_while_statement) override {}

  void VisitVariableDeclarator(
      JsVariableDeclarator &variable_declarator) override {}

  void VisitVariableDeclaration(
      JsVariableDeclaration &variable_declaration) override {}

  void VisitForStatement(JsForStatement &for_statement) override {}

  void VisitForInStatement(JsForInStatement &for_in_statement) override {}

  void VisitForOfStatement(JsForOfStatement &for_of_statement) override {}

  void VisitFunctionDeclaration(
      JsFunctionDeclaration &function_declaration) override {}

  void VisitSuper(JsSuper &super) override {}

  void VisitImport(JsImport &import) override {}

  void VisitThisExpression(JsThisExpression &this_expression) override {}

  void VisitArrowFunctionExpression(
      JsArrowFunctionExpression &arrow_function_expression) override {}

  void VisitYieldExpression(JsYieldExpression &yield_expression) override {}

  void VisitAwaitExpression(JsAwaitExpression &await_expression) override {}

  void VisitSpreadElement(JsSpreadElement &spread_element) override {}

  void VisitArrayExpression(JsArrayExpression &array_expression) override {}

  void VisitObjectProperty(JsObjectProperty &object_property) override {}

  void VisitObjectMethod(JsObjectMethod &object_method) override {}

  void VisitObjectExpression(JsObjectExpression &object_expression) override {}

  void VisitFunctionExpression(
      JsFunctionExpression &function_expression) override {}

  void VisitUnaryExpression(JsUnaryExpression &unary_expression) override {}

  void VisitUpdateExpression(JsUpdateExpression &update_expression) override {}

  void VisitBinaryExpression(JsBinaryExpression &binary_expression) override {}

  void VisitAssignmentExpression(
      JsAssignmentExpression &assignment_expression) override {}

  void VisitLogicalExpression(
      JsLogicalExpression &logical_expression) override {}

  void VisitMemberExpression(JsMemberExpression &member_expression) override {}

  void VisitOptionalMemberExpression(
      JsOptionalMemberExpression &optional_member_expression) override {}

  void VisitConditionalExpression(
      JsConditionalExpression &conditional_expression) override {}

  void VisitCallExpression(JsCallExpression &call_expression) override {}

  void VisitOptionalCallExpression(
      JsOptionalCallExpression &optional_call_expression) override {}

  void VisitNewExpression(JsNewExpression &new_expression) override {}

  void VisitSequenceExpression(
      JsSequenceExpression &sequence_expression) override {}

  void VisitParenthesizedExpression(
      JsParenthesizedExpression &parenthesized_expression) override {}

  void VisitTemplateElement(JsTemplateElement &template_element) override {}

  void VisitTemplateLiteral(JsTemplateLiteral &template_literal) override {}

  void VisitTaggedTemplateExpression(
      JsTaggedTemplateExpression &tagged_template_expression) override {}

  void VisitRestElement(JsRestElement &rest_element) override {}

  void VisitObjectPattern(JsObjectPattern &object_pattern) override {}

  void VisitArrayPattern(JsArrayPattern &array_pattern) override {}

  void VisitAssignmentPattern(
      JsAssignmentPattern &assignment_pattern) override {}

  void VisitClassMethod(JsClassMethod &class_method) override {}

  void VisitClassPrivateMethod(
      JsClassPrivateMethod &class_private_method) override {}

  void VisitClassProperty(JsClassProperty &class_property) override {}

  void VisitClassPrivateProperty(
      JsClassPrivateProperty &class_private_property) override {}

  void VisitClassBody(JsClassBody &class_body) override {}

  void VisitClassDeclaration(JsClassDeclaration &class_declaration) override {}

  void VisitClassExpression(JsClassExpression &class_expression) override {}

  void VisitMetaProperty(JsMetaProperty &meta_property) override {}

  void VisitImportSpecifier(JsImportSpecifier &import_specifier) override {}

  void VisitImportDefaultSpecifier(
      JsImportDefaultSpecifier &import_default_specifier) override {}

  void VisitImportNamespaceSpecifier(
      JsImportNamespaceSpecifier &import_namespace_specifier) override {}

  void VisitImportAttribute(JsImportAttribute &import_attribute) override {}

  void VisitImportDeclaration(
      JsImportDeclaration &import_declaration) override {}

  void VisitExportSpecifier(JsExportSpecifier &export_specifier) override {}

  void VisitExportNamedDeclaration(
      JsExportNamedDeclaration &export_named_declaration) override {}

  void VisitExportDefaultDeclaration(
      JsExportDefaultDeclaration &export_default_declaration) override {}

  void VisitExportAllDeclaration(
      JsExportAllDeclaration &export_all_declaration) override {}
};

class DefaultMutableJsAstVisitor : public virtual MutableJsAstVisitor<void> {
 public:
  ~DefaultMutableJsAstVisitor() override = default;

  void VisitInterpreterDirective(
      JsInterpreterDirective &interpreter_directive) override {
    this->VisitNodeDefault(interpreter_directive);
  }

  void VisitDirectiveLiteral(JsDirectiveLiteral &directive_literal) override {
    this->VisitNodeDefault(directive_literal);
  }

  void VisitDirective(JsDirective &directive) override {
    this->VisitNodeDefault(directive);
  }

  void VisitProgram(JsProgram &program) override {
    this->VisitNodeDefault(program);
  }

  void VisitFile(JsFile &file) override { this->VisitNodeDefault(file); }

  void VisitIdentifier(JsIdentifier &identifier) override {
    this->VisitNodeDefault(identifier);
  }

  void VisitPrivateName(JsPrivateName &private_name) override {
    this->VisitNodeDefault(private_name);
  }

  void VisitRegExpLiteral(JsRegExpLiteral &reg_exp_literal) override {
    this->VisitNodeDefault(reg_exp_literal);
  }

  void VisitNullLiteral(JsNullLiteral &null_literal) override {
    this->VisitNodeDefault(null_literal);
  }

  void VisitStringLiteral(JsStringLiteral &string_literal) override {
    this->VisitNodeDefault(string_literal);
  }

  void VisitBooleanLiteral(JsBooleanLiteral &boolean_literal) override {
    this->VisitNodeDefault(boolean_literal);
  }

  void VisitNumericLiteral(JsNumericLiteral &numeric_literal) override {
    this->VisitNodeDefault(numeric_literal);
  }

  void VisitBigIntLiteral(JsBigIntLiteral &big_int_literal) override {
    this->VisitNodeDefault(big_int_literal);
  }

  void VisitBlockStatement(JsBlockStatement &block_statement) override {
    this->VisitNodeDefault(block_statement);
  }

  void VisitExpressionStatement(
      JsExpressionStatement &expression_statement) override {
    this->VisitNodeDefault(expression_statement);
  }

  void VisitEmptyStatement(JsEmptyStatement &empty_statement) override {
    this->VisitNodeDefault(empty_statement);
  }

  void VisitDebuggerStatement(
      JsDebuggerStatement &debugger_statement) override {
    this->VisitNodeDefault(debugger_statement);
  }

  void VisitWithStatement(JsWithStatement &with_statement) override {
    this->VisitNodeDefault(with_statement);
  }

  void VisitReturnStatement(JsReturnStatement &return_statement) override {
    this->VisitNodeDefault(return_statement);
  }

  void VisitLabeledStatement(JsLabeledStatement &labeled_statement) override {
    this->VisitNodeDefault(labeled_statement);
  }

  void VisitBreakStatement(JsBreakStatement &break_statement) override {
    this->VisitNodeDefault(break_statement);
  }

  void VisitContinueStatement(
      JsContinueStatement &continue_statement) override {
    this->VisitNodeDefault(continue_statement);
  }

  void VisitIfStatement(JsIfStatement &if_statement) override {
    this->VisitNodeDefault(if_statement);
  }

  void VisitSwitchCase(JsSwitchCase &switch_case) override {
    this->VisitNodeDefault(switch_case);
  }

  void VisitSwitchStatement(JsSwitchStatement &switch_statement) override {
    this->VisitNodeDefault(switch_statement);
  }

  void VisitThrowStatement(JsThrowStatement &throw_statement) override {
    this->VisitNodeDefault(throw_statement);
  }

  void VisitCatchClause(JsCatchClause &catch_clause) override {
    this->VisitNodeDefault(catch_clause);
  }

  void VisitTryStatement(JsTryStatement &try_statement) override {
    this->VisitNodeDefault(try_statement);
  }

  void VisitWhileStatement(JsWhileStatement &while_statement) override {
    this->VisitNodeDefault(while_statement);
  }

  void VisitDoWhileStatement(JsDoWhileStatement &do_while_statement) override {
    this->VisitNodeDefault(do_while_statement);
  }

  void VisitVariableDeclarator(
      JsVariableDeclarator &variable_declarator) override {
    this->VisitNodeDefault(variable_declarator);
  }

  void VisitVariableDeclaration(
      JsVariableDeclaration &variable_declaration) override {
    this->VisitNodeDefault(variable_declaration);
  }

  void VisitForStatement(JsForStatement &for_statement) override {
    this->VisitNodeDefault(for_statement);
  }

  void VisitForInStatement(JsForInStatement &for_in_statement) override {
    this->VisitNodeDefault(for_in_statement);
  }

  void VisitForOfStatement(JsForOfStatement &for_of_statement) override {
    this->VisitNodeDefault(for_of_statement);
  }

  void VisitFunctionDeclaration(
      JsFunctionDeclaration &function_declaration) override {
    this->VisitNodeDefault(function_declaration);
  }

  void VisitSuper(JsSuper &super) override { this->VisitNodeDefault(super); }

  void VisitImport(JsImport &import) override {
    this->VisitNodeDefault(import);
  }

  void VisitThisExpression(JsThisExpression &this_expression) override {
    this->VisitNodeDefault(this_expression);
  }

  void VisitArrowFunctionExpression(
      JsArrowFunctionExpression &arrow_function_expression) override {
    this->VisitNodeDefault(arrow_function_expression);
  }

  void VisitYieldExpression(JsYieldExpression &yield_expression) override {
    this->VisitNodeDefault(yield_expression);
  }

  void VisitAwaitExpression(JsAwaitExpression &await_expression) override {
    this->VisitNodeDefault(await_expression);
  }

  void VisitSpreadElement(JsSpreadElement &spread_element) override {
    this->VisitNodeDefault(spread_element);
  }

  void VisitArrayExpression(JsArrayExpression &array_expression) override {
    this->VisitNodeDefault(array_expression);
  }

  void VisitObjectProperty(JsObjectProperty &object_property) override {
    this->VisitNodeDefault(object_property);
  }

  void VisitObjectMethod(JsObjectMethod &object_method) override {
    this->VisitNodeDefault(object_method);
  }

  void VisitObjectExpression(JsObjectExpression &object_expression) override {
    this->VisitNodeDefault(object_expression);
  }

  void VisitFunctionExpression(
      JsFunctionExpression &function_expression) override {
    this->VisitNodeDefault(function_expression);
  }

  void VisitUnaryExpression(JsUnaryExpression &unary_expression) override {
    this->VisitNodeDefault(unary_expression);
  }

  void VisitUpdateExpression(JsUpdateExpression &update_expression) override {
    this->VisitNodeDefault(update_expression);
  }

  void VisitBinaryExpression(JsBinaryExpression &binary_expression) override {
    this->VisitNodeDefault(binary_expression);
  }

  void VisitAssignmentExpression(
      JsAssignmentExpression &assignment_expression) override {
    this->VisitNodeDefault(assignment_expression);
  }

  void VisitLogicalExpression(
      JsLogicalExpression &logical_expression) override {
    this->VisitNodeDefault(logical_expression);
  }

  void VisitMemberExpression(JsMemberExpression &member_expression) override {
    this->VisitNodeDefault(member_expression);
  }

  void VisitOptionalMemberExpression(
      JsOptionalMemberExpression &optional_member_expression) override {
    this->VisitNodeDefault(optional_member_expression);
  }

  void VisitConditionalExpression(
      JsConditionalExpression &conditional_expression) override {
    this->VisitNodeDefault(conditional_expression);
  }

  void VisitCallExpression(JsCallExpression &call_expression) override {
    this->VisitNodeDefault(call_expression);
  }

  void VisitOptionalCallExpression(
      JsOptionalCallExpression &optional_call_expression) override {
    this->VisitNodeDefault(optional_call_expression);
  }

  void VisitNewExpression(JsNewExpression &new_expression) override {
    this->VisitNodeDefault(new_expression);
  }

  void VisitSequenceExpression(
      JsSequenceExpression &sequence_expression) override {
    this->VisitNodeDefault(sequence_expression);
  }

  void VisitParenthesizedExpression(
      JsParenthesizedExpression &parenthesized_expression) override {
    this->VisitNodeDefault(parenthesized_expression);
  }

  void VisitTemplateElement(JsTemplateElement &template_element) override {
    this->VisitNodeDefault(template_element);
  }

  void VisitTemplateLiteral(JsTemplateLiteral &template_literal) override {
    this->VisitNodeDefault(template_literal);
  }

  void VisitTaggedTemplateExpression(
      JsTaggedTemplateExpression &tagged_template_expression) override {
    this->VisitNodeDefault(tagged_template_expression);
  }

  void VisitRestElement(JsRestElement &rest_element) override {
    this->VisitNodeDefault(rest_element);
  }

  void VisitObjectPattern(JsObjectPattern &object_pattern) override {
    this->VisitNodeDefault(object_pattern);
  }

  void VisitArrayPattern(JsArrayPattern &array_pattern) override {
    this->VisitNodeDefault(array_pattern);
  }

  void VisitAssignmentPattern(
      JsAssignmentPattern &assignment_pattern) override {
    this->VisitNodeDefault(assignment_pattern);
  }

  void VisitClassMethod(JsClassMethod &class_method) override {
    this->VisitNodeDefault(class_method);
  }

  void VisitClassPrivateMethod(
      JsClassPrivateMethod &class_private_method) override {
    this->VisitNodeDefault(class_private_method);
  }

  void VisitClassProperty(JsClassProperty &class_property) override {
    this->VisitNodeDefault(class_property);
  }

  void VisitClassPrivateProperty(
      JsClassPrivateProperty &class_private_property) override {
    this->VisitNodeDefault(class_private_property);
  }

  void VisitClassBody(JsClassBody &class_body) override {
    this->VisitNodeDefault(class_body);
  }

  void VisitClassDeclaration(JsClassDeclaration &class_declaration) override {
    this->VisitNodeDefault(class_declaration);
  }

  void VisitClassExpression(JsClassExpression &class_expression) override {
    this->VisitNodeDefault(class_expression);
  }

  void VisitMetaProperty(JsMetaProperty &meta_property) override {
    this->VisitNodeDefault(meta_property);
  }

  void VisitImportSpecifier(JsImportSpecifier &import_specifier) override {
    this->VisitNodeDefault(import_specifier);
  }

  void VisitImportDefaultSpecifier(
      JsImportDefaultSpecifier &import_default_specifier) override {
    this->VisitNodeDefault(import_default_specifier);
  }

  void VisitImportNamespaceSpecifier(
      JsImportNamespaceSpecifier &import_namespace_specifier) override {
    this->VisitNodeDefault(import_namespace_specifier);
  }

  void VisitImportAttribute(JsImportAttribute &import_attribute) override {
    this->VisitNodeDefault(import_attribute);
  }

  void VisitImportDeclaration(
      JsImportDeclaration &import_declaration) override {
    this->VisitNodeDefault(import_declaration);
  }

  void VisitExportSpecifier(JsExportSpecifier &export_specifier) override {
    this->VisitNodeDefault(export_specifier);
  }

  void VisitExportNamedDeclaration(
      JsExportNamedDeclaration &export_named_declaration) override {
    this->VisitNodeDefault(export_named_declaration);
  }

  void VisitExportDefaultDeclaration(
      JsExportDefaultDeclaration &export_default_declaration) override {
    this->VisitNodeDefault(export_default_declaration);
  }

  void VisitExportAllDeclaration(
      JsExportAllDeclaration &export_all_declaration) override {
    this->VisitNodeDefault(export_all_declaration);
  }

  // override this method to customize the default behavior on each JS AST node.
  virtual void VisitNodeDefault(JsNode &node) = 0;
};

// This is a wrapper class of `DefaultMutableJsAstVisitor`. Users can specify
// the default behavior without explicitly defining a new derived visitor class.
class DefaultMutableJsAstVisitorLambdaWrapper
    : public DefaultMutableJsAstVisitor {
 public:
  explicit DefaultMutableJsAstVisitorLambdaWrapper(
      std::function<void(JsNode &)> func)
      : func_(std::move(func)) {}

  void VisitNodeDefault(JsNode &node) override { func_(node); }

 private:
  std::function<void(JsNode &)> func_;
};

}  // namespace maldoca

#endif  // MALDOCA_JS_AST_AST_VISITOR_H_
