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

#include "maldoca/astgen/ir_to_ast_source_printer.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/container/btree_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "maldoca/astgen/ast_def.h"
#include "maldoca/astgen/ast_def.pb.h"
#include "maldoca/astgen/ast_gen_utils.h"
#include "maldoca/astgen/symbol.h"
#include "maldoca/astgen/type.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"

namespace maldoca {
namespace {

MaybeNull OptionalnessToMaybeNull(Optionalness optionalness) {
  switch (optionalness) {
    case OPTIONALNESS_UNSPECIFIED:
    case OPTIONALNESS_REQUIRED:
      return MaybeNull::kNo;
    case OPTIONALNESS_MAYBE_NULL:
    case OPTIONALNESS_MAYBE_UNDEFINED:
      return MaybeNull::kYes;
  }
}

static Symbol GetVisitor(const NodeDef& node, FieldKind kind) {
  auto visitor = Symbol("Visit") + node.name();
  if (kind == FIELD_KIND_ATTR) {
    visitor += "Attr";
  }
  if (kind == FIELD_KIND_LVAL) {
    visitor += "Ref";
  }
  return visitor;
}

}  // namespace

void IrToAstSourcePrinter::PrintAst(const AstDef& ast,
                                    absl::string_view cc_namespace,
                                    absl::string_view ast_path,
                                    absl::string_view ir_path) {
  auto ast_header_path = GetAstHeaderPath(ast_path);

  PrintLicense();
  Println();

  PrintCodeGenerationWarning();
  Println();

  Println("// IWYU pragma: begin_keep");
  Println("// NOLINTBEGIN(whitespace/line_length)");
  Println("// clang-format off");
  Println();

  if (!header_include_path_.empty()) {
    PrintIncludeHeader(header_include_path_);
  } else {
    absl::string_view ir_lang = ir_lang_name_.empty()
                                    ? ast.lang_name()
                                    : absl::string_view(ir_lang_name_);
    PrintIncludeHeader(
        absl::StrCat(ir_path, "/conversion/", ir_lang, "ir_to_ast.h"));
  }
  Println();

  Println("#include <memory>");
  Println("#include <optional>");
  Println("#include <string>");
  Println("#include <utility>");
  Println("#include <variant>");
  Println("#include <vector>");
  Println();

  PrintIncludeHeaders({
      "llvm/ADT/APFloat.h",
      "llvm/ADT/TypeSwitch.h",
      "llvm/Support/Casting.h",
      "mlir/IR/Attributes.h",
      "mlir/IR/Block.h",
      "mlir/IR/Builders.h",
      "mlir/IR/BuiltinAttributes.h",
      "mlir/IR/BuiltinTypes.h",
      "mlir/IR/Operation.h",
      "mlir/IR/Region.h",
      "mlir/IR/Value.h",
      "absl/cleanup/cleanup.h",
      "absl/log/check.h",
      "absl/log/log.h",
      "absl/status/status.h",
      "absl/status/status_macros.h",
      "absl/status/statusor.h",
      "absl/strings/str_cat.h",
      "absl/types/optional.h",
      "absl/types/variant.h",
      "maldoca/astgen/ir_to_ast_util.h",
      std::string(ast_header_path),
      absl::StrCat(ir_path, "/ir.h"),
  });
  Println();

  PrintEnterNamespace(cc_namespace);
  Println();

  for (const auto* node : ast.topological_sorted_nodes()) {
    if (!node->children().empty() && node->should_generate_dispatch()) {
      for (FieldKind kind : node->aggregated_kinds()) {
        PrintNonLeafNode(ast, *node, kind);
      }
    }

    if (!node->should_generate_ir_op()) {
      continue;
    }

    for (FieldKind kind : node->aggregated_kinds()) {
      PrintLeafNode(ast, *node, kind);
    }
  }

  Println("// clang-format on");
  Println("// NOLINTEND(whitespace/line_length)");
  Println("// IWYU pragma: end_keep");
  Println();

  PrintExitNamespace(cc_namespace);
}

void IrToAstSourcePrinter::PrintNonLeafNode(const AstDef& ast,
                                            const NodeDef& node,
                                            FieldKind kind) {
  absl::string_view ir_lang = ir_lang_name_.empty()
                                  ? ast.lang_name()
                                  : absl::string_view(ir_lang_name_);
  auto ir_op_name = node.ir_op_name(ir_lang, kind);
  std::string input_type;
  if (ir_op_name.has_value()) {
    input_type = ir_op_name->ToPascalCase();
  } else {
    switch (kind) {
      case FIELD_KIND_UNSPECIFIED:
        LOG(FATAL) << "Invalid FieldKind: FIELD_KIND_UNSPECIFIED.";
      case FIELD_KIND_ATTR:
        input_type = "mlir::Attribute";
        break;
      case FIELD_KIND_LVAL:
      case FIELD_KIND_RVAL:
      case FIELD_KIND_STMT:
        input_type = "mlir::Operation*";
        break;
    }
  }
  auto ir_name = Symbol(absl::StrCat(ir_lang, "ir"));
  auto visitor = GetVisitor(node, kind);
  std::string class_name = class_name_.empty()
                               ? absl::StrCat(ir_name.ToPascalCase(), "ToAst")
                               : class_name_;

  auto vars = WithVars({
      {"InputType", input_type},
      {"BaseName",
       kind == FIELD_KIND_ATTR ? "mlir::Attribute" : "mlir::Operation*"},
      {"Name", (Symbol(ast.lang_name()) + node.name()).ToPascalCase()},
      {"name", kind == FIELD_KIND_ATTR ? "attr" : "op"},
      {"ClassName", class_name},
      {"Visitor", visitor.ToPascalCase()},
  });

  Println("absl::StatusOr<std::unique_ptr<$Name$>>");
  Println("$ClassName$::$Visitor$($InputType$ $name$) {");
  {
    auto indent = WithIndent();
    Println("using Ret = absl::StatusOr<std::unique_ptr<$Name$>>;");
    Println("return llvm::TypeSwitch<$BaseName$, Ret>($name$)");
    {
      auto indent = WithIndent();
      for (const NodeDef* leaf : node.leaves()) {
        if (node.dispatch_skip().contains(leaf->name())) {
          continue;
        }
        std::string visitor_name;
        std::string op_name;
        auto it = node.dispatch_overrides().find(leaf->name());

        if (it != node.dispatch_overrides().end() &&
            !it->second.visitor.empty()) {
          visitor_name = it->second.visitor;
        } else {
          visitor_name = GetVisitor(*leaf, kind).ToPascalCase();
        }

        if (it != node.dispatch_overrides().end() &&
            it->second.ir_op_name.has_value()) {
          op_name = *it->second.ir_op_name;
        } else {
          auto ir_op = leaf->ir_op_name(ir_lang, kind);
          if (!ir_op.has_value()) {
            LOG(FATAL) << "Leaf node " << leaf->name()
                       << " has no IR op name for " << ir_lang;
          }
          op_name = ir_op->ToPascalCase();
        }
        auto vars = WithVars({
            {"LeafOpName", op_name},
            {"LeafVisitor", visitor_name},
        });
        Println(".Case([&]($LeafOpName$ $name$) {");
        Println("  return $LeafVisitor$($name$);");
        Println("})");
      }

      Println(".Default([&]($BaseName$ op) {");
      Println("  return absl::InvalidArgumentError(\"Unrecognized op\");");
      Println("});");
    }
  }
  Println("}");
  Println();
}

void IrToAstSourcePrinter::PrintLeafNode(const AstDef& ast, const NodeDef& node,
                                         FieldKind kind) {
  absl::string_view ir_lang = ir_lang_name_.empty()
                                  ? ast.lang_name()
                                  : absl::string_view(ir_lang_name_);
  auto ir_op_name = node.ir_op_name(ir_lang, kind).value();
  auto ir_name = Symbol(absl::StrCat(ir_lang, "ir"));

  auto visitor = Symbol("Visit") + node.name();
  if (kind == FIELD_KIND_LVAL) {
    visitor += "Ref";
  }

  std::string class_name = class_name_.empty()
                               ? absl::StrCat(ir_name.ToPascalCase(), "ToAst")
                               : class_name_;

  auto vars = WithVars({
      {"OpName", ir_op_name.ToPascalCase()},
      {"Name", (Symbol(ast.lang_name()) + node.name()).ToPascalCase()},
      {"name", kind == FIELD_KIND_ATTR ? "attr" : "op"},
      {"ClassName", class_name},
      {"Visitor", visitor.ToPascalCase()},
  });

  Println("absl::StatusOr<std::unique_ptr<$Name$>>");
  Println("$ClassName$::$Visitor$($OpName$ $name$) {");
  {
    auto indent = WithIndent();
    for (const auto* field : node.aggregated_fields()) {
      if (FieldIsSplit(field) && FieldIsInIrOp(field)) {
        PrintSplitField(ast, node, *field);
      } else if (FieldHasLiteralAttr(field) && FieldIsInIrOp(field)) {
        PrintLiteralAttrField(ast, node, *field, kind);
      } else if (FieldIsArgument(field)) {
        PrintField(ast, node, *field, kind);
      } else if (FieldIsRegion(field)) {
        PrintRegion(ast, node, *field, kind);
      }
    }

    // Call the constructor.
    Print("return Create<$Name$>(\n");
    {
      auto indent = WithIndent(4);
      Print("$name$");

      for (const FieldDef* field : node.aggregated_fields()) {
        if (FieldIsOwnAstOnly(node, field)) {
          // The IR doesn't hold the field. The fields inherited from the
          // ancestors (e.g. the trivia of a `Node`) are filled in by
          // `Create<>`; the node's own fields are passed as unset.
          CHECK(field->optionalness() != OPTIONALNESS_REQUIRED)
              << node.name() << "." << field->name().ToCcVarName()
              << ": A field that is only in the AST must be optional.";
          Print(",\nstd::nullopt");
          continue;
        }
        if (!FieldIsInIrOp(field)) {
          continue;
        }

        // A required split field is held in an std::optional until we've
        // checked that it is set.
        bool unwrap = FieldIsSplit(field) &&
                      field->optionalness() == OPTIONALNESS_REQUIRED;
        auto vars = WithVars({
            {"field_name", field->name().ToCcVarName()},
            {"deref", unwrap ? "*" : ""},
        });
        Print(",\n$deref$std::move($field_name$)");
      }
    }

    Println(");");
  }
  Println("}");
  Println();
}

void IrToAstSourcePrinter::PrintField(const AstDef& ast, const NodeDef& node,
                                      const FieldDef& field,
                                      FieldKind op_kind) {
  MaybeNull maybe_null = OptionalnessToMaybeNull(field.optionalness());
  FieldKind kind = FieldIrKind(field, op_kind);

  auto mlir_getter = FieldIrName(field).ToMlirGetter();

  std::string rhs;
  switch (kind) {
    case FIELD_KIND_UNSPECIFIED:
      LOG(FATAL) << "Unspecified FieldKind.";
    case FIELD_KIND_ATTR: {
      rhs = absl::StrCat("op.", mlir_getter, "Attr()");
      break;
    }
    case FIELD_KIND_LVAL:
    case FIELD_KIND_RVAL: {
      rhs = absl::StrCat("op.", mlir_getter, "()");
      break;
    }
    case FIELD_KIND_STMT: {
      LOG(FATAL) << "Unsupported FieldKind.";
    }
  }

  std::unique_ptr<Type> narrowed = NarrowedIrType(ast, field, op_kind);
  // A class-typed field narrowed in the LVAL op (see
  // FieldDef::LvalOpOverride::type) also
  // accepts an op of the field's own type, which the AST -> IR conversion
  // falls back to when the AST holds a node without the narrowed form.
  bool narrowed_with_fallback =
      narrowed != nullptr && field.type().IsA<ClassType>();
  auto lhs = narrowed != nullptr ? (field.name() + "ir").ToCcVarName()
                                 : field.name().ToCcVarName();

  auto vars = WithVars({
      {"lhs", lhs},
      {"rhs", rhs},
  });

  Println("ABSL_ASSIGN_OR_RETURN(");
  {
    auto indent = WithIndent(4);
    Println("auto $lhs$,");
    Print("Convert(\n");
    {
      auto indent = WithIndent(4);
      Println("$rhs$,");
      if (FieldHasPerMemberKinds(field)) {
        PrintPerMemberKindsConverter(ast, field, op_kind, maybe_null);
      } else if (narrowed_with_fallback) {
        CHECK(maybe_null == MaybeNull::kNo);
        Println("OpVariant(");
        {
          auto indent = WithIndent(4);
          PrintConverter(ast, *narrowed, ast.lang_name(), kind, MaybeNull::kNo);
          Print(",\n");
          PrintConverter(ast, field.type(), ast.lang_name(), field.kind(),
                         MaybeNull::kNo);
          Println();
        }
        Print(")");
      } else {
        PrintConverter(ast, narrowed != nullptr ? *narrowed : field.type(),
                       ast.lang_name(), kind, maybe_null);
      }
      Println();
    }
    Println(")");
  }
  Println(");");

  if (narrowed != nullptr) {
    // The op holds a narrower type than the AST field. Downcast the converted
    // value to the field's type.
    auto vars = WithVars({
        {"field_name", field.name().ToCcVarName()},
        {"CcType", field.type().CcType()},
    });
    // The extra parentheses protect the commas in $CcType$ from the macro.
    Println(
        "ABSL_ASSIGN_OR_RETURN(auto $field_name$, "
        "(DynCastAstNode<$CcType$>(std::move($lhs$))));");
  }
}

void IrToAstSourcePrinter::PrintLiteralAttrField(const AstDef& ast,
                                                 const NodeDef& node,
                                                 const FieldDef& field,
                                                 FieldKind op_kind) {
  absl::string_view ir_lang = ir_lang_name_.empty()
                                  ? ast.lang_name()
                                  : absl::string_view(ir_lang_name_);
  const FieldDef::LiteralAttr& literal_attr_def = *field.literal_attr();

  Symbol literal_attr{literal_attr_def.name};
  Symbol operand = field.ir_name();
  const std::vector<std::string>& type_names = literal_attr_def.types;

  auto vars = WithVars({
      {"CcType", field.type().CcType()},
      {"field_name", field.name().ToCcVarName()},
      {"literal_unless", Symbol(literal_attr_def.discriminator).ToCcVarName()},
      {"literal_attr", literal_attr.ToCcVarName()},
      {"literalAttrGetter", literal_attr.ToMlirGetter()},
      {"operand", operand.ToCcVarName()},
      {"operandGetter", operand.ToMlirGetter()},
      {"literal_types", absl::StrJoin(type_names, " or ")},
  });

  Println("$CcType$ $field_name$;");
  Println("bool $literal_unless$;");
  Println(
      "if (mlir::Attribute $literal_attr$ = op.$literalAttrGetter$Attr()) {");
  {
    auto indent = WithIndent();
    Println("$literal_unless$ = false;");
    IfStmtPrinter if_printer(this);
    for (const std::string& type_name : type_names) {
      bool exact = false;
      std::optional<size_t> member =
          FindMemberHolding(ast, field.type(), type_name, &exact);
      CHECK(member.has_value())
          << field.name().ToCcVarName() << ": literal_attr_types \""
          << type_name << "\" isn't held by any member of the field.";

      const NodeDef& type_node = *ast.nodes().at(type_name);
      auto typed_attr = (literal_attr + Symbol(type_name)).ToCcVarName();
      auto vars = WithVars({
          {"AttrType",
           type_node.ir_op_name(ir_lang, FIELD_KIND_ATTR)->ToPascalCase()},
          {"typed_attr", typed_attr},
          {"typed_ast", (Symbol(typed_attr) + "ast").ToCcVarName()},
          {"Visitor", GetVisitor(type_node, FIELD_KIND_ATTR).ToPascalCase()},
          {"index", std::to_string(*member)},
      });
      if_printer.PrintCase({
          [&] {
            Print(
                "auto $typed_attr$ = "
                "llvm::dyn_cast<$AttrType$>($literal_attr$)");
          },
          [&] {
            Println(
                "ABSL_ASSIGN_OR_RETURN(auto $typed_ast$, "
                "$Visitor$($typed_attr$));");
            if (field.type().IsA<VariantType>()) {
              Println("$field_name$.emplace<$index$>(std::move($typed_ast$));");
            } else {
              Println("$field_name$ = std::move($typed_ast$);");
            }
          },
      });
    }
    Println(" else {");
    Println(
        "  return absl::InvalidArgumentError(\"$literal_attr$ must be "
        "$literal_types$.\");");
    Println("}");
  }
  Println("} else if (op.$operandGetter$() != nullptr) {");
  {
    auto indent = WithIndent();
    Println("$literal_unless$ = true;");
    Println("ABSL_ASSIGN_OR_RETURN(");
    {
      auto indent = WithIndent(4);
      Println("$field_name$,");
      Print("Convert(\n");
      {
        auto indent = WithIndent(4);
        Println("op.$operandGetter$(),");
        PrintConverter(ast, field.type(), ast.lang_name(),
                       FieldIrKind(field, op_kind), MaybeNull::kNo);
        Println();
      }
      Println(")");
    }
    Println(");");
  }
  Println("} else {");
  Println(
      "  return absl::InvalidArgumentError(\"Exactly one of $literal_attr$ and "
      "$operand$ must be set.\");");
  Println("}");
}

void IrToAstSourcePrinter::PrintSplitField(const AstDef& ast,
                                           const NodeDef& node,
                                           const FieldDef& field) {
  absl::string_view ir_lang = ir_lang_name_.empty()
                                  ? ast.lang_name()
                                  : absl::string_view(ir_lang_name_);
  std::vector<std::string> member_names = MemberNames(field.type());

  auto vars = WithVars({
      {"CcType", field.type().CcType()},
      {"field_name", field.name().ToCcVarName()},
  });
  Println("std::optional<$CcType$> $field_name$;");

  std::vector<std::string> slot_names;
  for (const FieldDef::IrSplit& split : field.ir_splits()) {
    Symbol slot{split.name};
    slot_names.push_back(slot.ToCcVarName());

    std::vector<std::string> types{split.types.begin(), split.types.end()};
    std::unique_ptr<Type> group_type =
        MakeClassOrVariantType(types, ast.lang_name());

    auto vars = WithVars({
        {"mlirGetter", slot.ToMlirGetter()},
        {"slot_ast", (slot + "ast").ToCcVarName()},
    });
    if (!split.is_region) {
      Println("if (op.$mlirGetter$() != nullptr) {");
    } else {
      Println("if (!op.$mlirGetter$().empty()) {");
    }
    {
      auto indent = WithIndent();
      Println("ABSL_ASSIGN_OR_RETURN(");
      {
        auto indent = WithIndent(4);
        Println("auto $slot_ast$,");
        Print("Convert(\n");
        {
          auto indent = WithIndent(4);
          Println("op.$mlirGetter$(),");
          if (!split.is_region) {
            PrintConverter(ast, *group_type, ast.lang_name(), split.kind,
                           MaybeNull::kNo);
            Println();
          } else {
            std::string converter_type;
            if (split.kind == FIELD_KIND_STMT) {
              converter_type = "StmtRegion";
            } else {
              converter_type = absl::StrCat(
                  "ExprRegion<",
                  Symbol(absl::StrCat(ir_lang, "ir")).ToPascalCase(),
                  "ExprRegionEndOp>");
            }
            auto vars = WithVars({{"ConverterType", converter_type}});
            Print("$ConverterType$(\n");
            {
              auto indent = WithIndent(4);
              PrintConverter(ast, *group_type, ast.lang_name(), split.kind,
                             MaybeNull::kNo);
              Println();
            }
            Println(")");
          }
        }
        Println(")");
      }
      Println(");");

      if (types.size() == 1) {
        auto index = absl::c_find(member_names, types.front());
        CHECK(index != member_names.end());
        auto vars = WithVars({
            {"index", std::to_string(index - member_names.begin())},
        });
        Println(
            "$field_name$.emplace(std::in_place_index<$index$>, "
            "std::move($slot_ast$));");
      } else {
        Println(
            "$field_name$ = WidenVariant<$CcType$>(std::move($slot_ast$));");
      }
    }
    Println("}");
  }

  if (field.optionalness() == OPTIONALNESS_REQUIRED) {
    auto vars = WithVars({
        {"slot_names", absl::StrJoin(slot_names, ", ")},
    });
    Println("if (!$field_name$.has_value()) {");
    Println(
        "  return absl::InvalidArgumentError(\"Exactly one of $slot_names$ "
        "must be set.\");");
    Println("}");
  }
}

void IrToAstSourcePrinter::PrintRegion(const AstDef& ast, const NodeDef& node,
                                       const FieldDef& field,
                                       FieldKind op_kind) {
  absl::string_view ir_lang = ir_lang_name_.empty()
                                  ? ast.lang_name()
                                  : absl::string_view(ir_lang_name_);
  MaybeNull maybe_null = OptionalnessToMaybeNull(field.optionalness());
  FieldKind kind = FieldIrKind(field, op_kind);

  std::string converter_type = [&]() -> std::string {
    switch (kind) {
      case FIELD_KIND_UNSPECIFIED:
        LOG(FATAL) << "Unspecified FieldKind.";
      case FIELD_KIND_ATTR:
        LOG(FATAL) << "Unsupported FieldKind: " << kind;
      case FIELD_KIND_RVAL:
      case FIELD_KIND_LVAL: {
        if (field.type().IsA<ListType>()) {
          std::string end_op =
              absl::StrCat(Symbol(absl::StrCat(ir_lang, "ir")).ToPascalCase(),
                           "ExprsRegionEndOp");
          return absl::StrCat("ExprsRegion<", end_op, ">");
        } else {
          std::string end_op =
              absl::StrCat(Symbol(absl::StrCat(ir_lang, "ir")).ToPascalCase(),
                           "ExprRegionEndOp");
          return absl::StrCat("ExprRegion<", end_op, ">");
        }
      }
      case FIELD_KIND_STMT: {
        if (field.type().IsA<ListType>()) {
          return "StmtsRegion";
        } else {
          return "StmtRegion";
        }
      }
    }
  }();

  auto vars = WithVars({
      {"ConverterType", converter_type},
      {"lhs", field.name().ToCcVarName()},
      {"mlirGetter", FieldIrName(field).ToMlirGetter()},
  });

  Println("ABSL_ASSIGN_OR_RETURN(");
  {
    auto indent = WithIndent(4);
    Println("auto $lhs$,");
    Print("Convert(\n");
    {
      auto indent = WithIndent(4);
      Println("op.$mlirGetter$(),");

      switch (maybe_null) {
        case MaybeNull::kYes: {
          Println("Nullable(");
          {
            auto indent = WithIndent(4);
            Print("$ConverterType$(\n");
            {
              auto indent = WithIndent(4);
              PrintConverter(ast, field.type(), ast.lang_name(), kind,
                             MaybeNull::kNo);
              Println();
            }
            Println(")");
          }
          Println(")");
          break;
        }
        case MaybeNull::kNo: {
          Print("$ConverterType$(\n");
          {
            auto indent = WithIndent(4);
            PrintConverter(ast, field.type(), ast.lang_name(), kind,
                           MaybeNull::kNo);
            Println();
          }
          Println(")");
        }
      }
    }
    Println(")");
  }
  Println(");");
}

void IrToAstSourcePrinter::PrintConverter(const AstDef& ast, const Type& type,
                                          absl::string_view lang_name,
                                          FieldKind kind,
                                          MaybeNull maybe_null) {
  absl::string_view ir_lang = ir_lang_name_.empty()
                                  ? ast.lang_name()
                                  : absl::string_view(ir_lang_name_);
  if (maybe_null == MaybeNull::kYes) {
    if (kind == FIELD_KIND_LVAL || kind == FIELD_KIND_RVAL) {
      std::string none_op = absl::StrCat(
          Symbol(absl::StrCat(ir_lang, "ir")).ToPascalCase(), "NoneOp");
      Print(absl::StrCat("Nullable<", none_op, ">(\n"));
    } else {
      Print("Nullable(\n");
    }
    {
      auto indent = WithIndent(4);
      PrintConverter(ast, type, lang_name, kind, MaybeNull::kNo);
      Println();
    }
    Print(")");
    return;
  }

  switch (type.kind()) {
    case TypeKind::kList: {
      const auto& list_type = static_cast<const ListType&>(type);
      PrintListConverter(ast, list_type, lang_name, kind);
      break;
    }

    case TypeKind::kVariant: {
      const auto& variant_type = static_cast<const VariantType&>(type);
      PrintVariantConverter(ast, variant_type, lang_name, kind);
      break;
    }

    case TypeKind::kClass: {
      const auto& class_type = static_cast<const ClassType&>(type);
      PrintClassConverter(class_type, lang_name, kind);
      break;
    }

    case TypeKind::kEnum: {
      const auto& enum_type = static_cast<const EnumType&>(type);
      PrintEnumConverter(enum_type, lang_name);
      break;
    }

    case TypeKind::kBuiltin: {
      const auto& builtin_type = static_cast<const BuiltinType&>(type);
      PrintBuiltinConverter(builtin_type, kind);
      break;
    }
  }
}

void IrToAstSourcePrinter::PrintBuiltinConverter(
    const BuiltinType& builtin_type, FieldKind kind) {
  switch (builtin_type.builtin_kind()) {
    case BuiltinTypeKind::kString:
      Print("ToString()");
      break;
    case BuiltinTypeKind::kBool:
      Print("ToBool()");
      break;
    case BuiltinTypeKind::kInt64:
      Print("ToInt64()");
      break;
    case BuiltinTypeKind::kDouble:
      Print("ToDouble()");
      break;
    default:
      LOG(FATAL) << "Unreachable code.";
  }
}

void IrToAstSourcePrinter::PrintEnumConverter(const EnumType& enum_type,
                                              absl::string_view lang_name) {
  auto enum_name = Symbol(lang_name) + enum_type.name();

  auto vars = WithVars({
      {"EnumName", enum_name.ToPascalCase()},
      {"cc_type", enum_type.CcType()},
  });
  Print("Enum<$cc_type$>(StringTo$EnumName$)");
}

void IrToAstSourcePrinter::PrintClassConverter(const ClassType& class_type,
                                               absl::string_view lang_name,
                                               FieldKind kind) {
  auto visitor = Symbol("Visit") + class_type.name();
  if (kind == FIELD_KIND_ATTR) {
    visitor += "Attr";
  }
  if (kind == FIELD_KIND_LVAL) {
    visitor += "Ref";
  }

  auto vars = WithVars({{"Visitor", visitor.ToPascalCase()}});
  if (kind == FIELD_KIND_ATTR) {
    Print("ToAttrConverter($Visitor$)");
  } else {
    Print("ToOpConverter($Visitor$)");
  }
}

void IrToAstSourcePrinter::PrintVariantConverter(
    const AstDef& ast, const VariantType& variant_type,
    absl::string_view lang_name, FieldKind kind) {
  if (kind == FIELD_KIND_ATTR) {
    Println("AttrVariant(");
  } else {
    Println("OpVariant(");
  }
  {
    auto indent = WithIndent(4);
    TabPrinter tab_printer{{
        .print_separator = [this] { Print(",\n"); },
    }};

    for (const auto& type : variant_type.types()) {
      tab_printer.Print();
      PrintConverter(ast, *type, lang_name, kind, MaybeNull::kNo);
    }
    Println();
  }
  Print(")");
}

void IrToAstSourcePrinter::PrintPerMemberKindsConverter(const AstDef& ast,
                                                        const FieldDef& field,
                                                        FieldKind op_kind,
                                                        MaybeNull maybe_null) {
  absl::string_view ir_lang = ir_lang_name_.empty()
                                  ? ast.lang_name()
                                  : absl::string_view(ir_lang_name_);
  CHECK(field.type().IsA<VariantType>())
      << field.name().ToCcVarName()
      << ": Alternatives of different kinds require a variant field.";
  const auto& variant_type = static_cast<const VariantType&>(field.type());

  if (maybe_null == MaybeNull::kYes) {
    std::string none_op = absl::StrCat(
        Symbol(absl::StrCat(ir_lang, "ir")).ToPascalCase(), "NoneOp");
    Print(absl::StrCat("Nullable<", none_op, ">(\n"));
  }
  {
    auto indent = WithIndent(maybe_null == MaybeNull::kYes ? 4 : 0);
    Println("OpVariant(");
    {
      auto indent = WithIndent(4);
      TabPrinter tab_printer{{
          .print_separator = [this] { Print(",\n"); },
      }};
      for (const auto& member : variant_type.types()) {
        CHECK(member->IsA<ClassType>())
            << field.name().ToCcVarName()
            << ": Alternatives of different kinds require class members.";
        const auto& class_type = static_cast<const ClassType&>(*member);
        tab_printer.Print();
        PrintClassConverter(
            class_type, ast.lang_name(),
            FieldMemberIrKind(field, class_type.name().ToPascalCase(),
                              op_kind));
      }
      Println();
    }
    Print(")");
  }
  if (maybe_null == MaybeNull::kYes) {
    Println();
    Print(")");
  }
}

void IrToAstSourcePrinter::PrintListConverter(const AstDef& ast,

                                              const ListType& list_type,
                                              absl::string_view lang_name,
                                              FieldKind kind) {
  Println("List(");
  {
    auto indent = WithIndent(4);
    PrintConverter(ast, list_type.element_type(), lang_name, kind,
                   list_type.element_maybe_null());
    Println();
  }
  Print(")");
}

std::string PrintIrToAstSource(const AstDef& ast,
                               absl::string_view cc_namespace,
                               absl::string_view ast_path,
                               absl::string_view ir_path,
                               absl::string_view ir_lang_name,
                               absl::string_view header_include_path,
                               absl::string_view class_name) {
  std::string str;
  {
    google::protobuf::io::StringOutputStream os(&str);
    IrToAstSourcePrinter printer(&os, ir_lang_name, header_include_path,
                                 class_name);
    printer.PrintAst(ast, cc_namespace, ast_path, ir_path);
  }

  return str;
}

}  // namespace maldoca
