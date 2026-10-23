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

#include "maldoca/astgen/ast_to_ir_source_printer.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
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

// Gets the name of the *RegionEndOp.
// - For an lval or rval (expression): <Ir>ExprRegionEndOp.
// - For a list of lvals or rvals (expressions): <Ir>ExprsRegionEndOp.
Symbol GetRegionEndOp(FieldKind kind, bool is_list, absl::string_view ir_lang) {
  auto ir_name = Symbol(absl::StrCat(ir_lang, "ir"));

  Symbol region_end_op;
  switch (kind) {
    case FIELD_KIND_UNSPECIFIED:
      LOG(FATAL) << "Unspecified FieldKind.";
    case FIELD_KIND_ATTR:
      LOG(FATAL) << "Unsupported FieldKind: " << kind;
    case FIELD_KIND_RVAL:
    case FIELD_KIND_LVAL: {
      if (is_list) {
        return ir_name + "ExprsRegionEndOp";
      } else {
        return ir_name + "ExprRegionEndOp";
      }
    }
    case FIELD_KIND_STMT: {
      return Symbol{};
    }
  }
}

// The names of the C++ variables holding the arguments of a field, in the
// order of the op builder's parameters.
std::vector<Symbol> GetArgumentVariables(const FieldDef& field) {
  if (FieldIsSplit(&field)) {
    std::vector<Symbol> variables;
    for (const FieldIrSplitPb& split : field.ir_splits()) {
      if (split.has_operand()) {
        variables.push_back(Symbol("mlir") + Symbol(split.operand()));
      }
    }
    return variables;
  }
  if (FieldHasLiteralAttr(&field)) {
    return {
        Symbol("mlir") + Symbol(field.ir_mapping()->literal_attr()),
        Symbol("mlir") + Symbol(field.ir_mapping()->operand()),
    };
  }
  return {Symbol("mlir") + field.name()};
}

}  // namespace

void AstToIrSourcePrinter::PrintAst(const AstDef& ast,
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
        absl::StrCat(ir_path, "/conversion/ast_to_", ir_lang, "ir.h"));
  }
  Println();

  Println("#include <memory>");
  Println("#include <utility>");
  Println("#include <vector>");
  Println();

  PrintIncludeHeaders({
      "llvm/ADT/APFloat.h",
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
      "absl/types/optional.h",
      "absl/types/variant.h",
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

void AstToIrSourcePrinter::PrintNonLeafNode(const AstDef& ast,
                                            const NodeDef& node,
                                            FieldKind kind) {
  absl::string_view ir_lang = ir_lang_name_.empty()
                                  ? ast.lang_name()
                                  : absl::string_view(ir_lang_name_);
  auto ir_op_name = node.ir_op_name(ir_lang, kind);
  std::string return_type;
  if (ir_op_name.has_value()) {
    return_type = ir_op_name.value().ToPascalCase();
  } else {
    switch (kind) {
      case FIELD_KIND_UNSPECIFIED:
        LOG(FATAL) << "Invalid FieldKind: FIELD_KIND_UNSPECIFIED.";
      case FIELD_KIND_ATTR: {
        return_type = "mlir::Attribute";
        break;
      }
      case FIELD_KIND_LVAL:
      case FIELD_KIND_RVAL: {
        return_type = "mlir::Value";
        break;
      }
      case FIELD_KIND_STMT: {
        return_type = "mlir::Operation*";
        break;
      }
    }
  }
  auto ir_name = Symbol(absl::StrCat(ir_lang, "ir"));
  auto visitor = GetVisitor(node, kind);
  std::string class_name = class_name_.empty()
                               ? absl::StrCat("AstTo", ir_name.ToPascalCase())
                               : class_name_;

  auto vars = WithVars({
      {"Ret", return_type},
      {"Name", (Symbol(ast.lang_name()) + node.name()).ToPascalCase()},
      {"ClassName", class_name},
      {"Visitor", visitor.ToPascalCase()},
  });

  Println(
      "$Ret$ $ClassName$::$Visitor$(mlir::OpBuilder &builder, const $Name$ "
      "*node) {");
  {
    auto indent = WithIndent();
    for (const NodeDef* leaf : node.leaves()) {
      if (node.dispatch_skip().contains(leaf->name())) {
        continue;
      }
      auto it = node.dispatch_overrides().find(leaf->name());
      std::string visitor_name;
      if (it != node.dispatch_overrides().end() &&
          !it->second.visitor.empty()) {
        visitor_name = it->second.visitor;
      } else {
        visitor_name = GetVisitor(*leaf, kind).ToPascalCase();
      }
      auto vars = WithVars({
          {"LeafName", (Symbol(ast.lang_name()) + leaf->name()).ToPascalCase()},
          {"leaf_name", Symbol(leaf->name()).ToCcVarName()},
          {"LeafVisitor", visitor_name},
      });
      Println(
          "if (auto *$leaf_name$ = dynamic_cast<const $LeafName$ *>(node)) {");
      Println("  return $LeafVisitor$(builder, $leaf_name$);");
      Println("}");
    }

    Println("LOG(FATAL) << \"Unreachable code.\";");
  }
  Println("}");
  Println();
}

void AstToIrSourcePrinter::PrintLeafNode(const AstDef& ast, const NodeDef& node,
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

  auto creator = Symbol("Create");
  switch (kind) {
    case FIELD_KIND_UNSPECIFIED:
    case FIELD_KIND_ATTR:
      LOG(FATAL) << "Unsupported kind: " << kind;
    case FIELD_KIND_LVAL:
    case FIELD_KIND_RVAL:
      creator += "Expr";
      break;
    case FIELD_KIND_STMT:
      creator += "Stmt";
      break;
  }

  std::string class_name = class_name_.empty()
                               ? absl::StrCat("AstTo", ir_name.ToPascalCase())
                               : class_name_;

  auto vars = WithVars({
      {"OpName", ir_op_name.ToPascalCase()},
      {"Name", (Symbol(ast.lang_name()) + node.name()).ToPascalCase()},
      {"ClassName", class_name},
      {"Visitor", visitor.ToPascalCase()},
      {"Creator", creator.ToPascalCase()},
  });

  Println(
      "$OpName$ $ClassName$::$Visitor$(mlir::OpBuilder &builder, const "
      "$Name$ *node) {");
  {
    auto indent = WithIndent();

    for (const auto* field : node.aggregated_fields()) {
      if (FieldIsArgument(field)) {
        PrintField(ast, node, *field, kind);
      }
    }

    bool has_regions = absl::c_any_of(
        node.aggregated_fields(),
        [](const FieldDef* field) { return FieldIsRegion(field); });
    if (has_regions) {
      Print("auto op = ");
    } else {
      Print("return ");
    }

    Print("$Creator$<$OpName$>(builder, node");
    {
      auto indent = WithIndent(4);
      for (const auto* field : node.aggregated_fields()) {
        if (!FieldIsArgument(field)) {
          continue;
        }

        for (const Symbol& mlir_field_name : GetArgumentVariables(*field)) {
          auto vars = WithVars({
              {"mlir_field_name", mlir_field_name.ToCcVarName()},
          });

          Print(", $mlir_field_name$");
        }
      }
    }
    Println(");");

    if (has_regions) {
      for (const auto* field : node.aggregated_fields()) {
        if (FieldIsRegion(field)) {
          PrintRegion(ast, node, *field, kind);
        }
      }

      Println("return op;");
    }
  }

  Println("}");
  Println();
}

void AstToIrSourcePrinter::PrintField(const AstDef& ast, const NodeDef& node,
                                      const FieldDef& field,
                                      FieldKind op_kind) {
  if (FieldIsSplit(&field)) {
    return PrintSplitField(ast, field, /*regions=*/false);
  }
  if (FieldHasLiteralAttr(&field)) {
    return PrintLiteralAttrField(ast, node, field, op_kind);
  }

  MaybeNull maybe_null = OptionalnessToMaybeNull(field.optionalness());
  FieldKind kind = FieldIrKind(field, op_kind);

  auto lhs = Symbol("mlir") + field.name();
  auto rhs = absl::StrCat("node->", field.name().ToCcVarName(), "()");

  if (std::unique_ptr<Type> narrowed = NarrowedIrType(ast, field, op_kind)) {
    // The op holds a narrower type than the AST field, e.g. only the
    // `Pattern` members of `Expression | Pattern` in an LVAL op. Downcast the
    // field value to it.
    CHECK(maybe_null == MaybeNull::kNo)
        << field.name().ToCcVarName()
        << ": Narrowed IR types are only supported for required fields.";
    const auto& narrowed_class = static_cast<const ClassType&>(*narrowed);
    auto narrowed_rhs = (field.name() + "ir").ToCcVarName();
    auto vars = WithVars({
        {"NarrowedType", narrowed_class.CcClassName()},
        {"narrowed_rhs", narrowed_rhs},
        {"rhs", rhs},
        {"field", field.name().ToCcVarName()},
    });
    if (field.type().IsA<VariantType>()) {
      Println(
          "const $NarrowedType$ *$narrowed_rhs$ = std::visit([](const auto "
          "*value) { return dynamic_cast<const $NarrowedType$ *>(value); }, "
          "$rhs$);");
    } else {
      Println(
          "const $NarrowedType$ *$narrowed_rhs$ = dynamic_cast<const "
          "$NarrowedType$ *>($rhs$);");
    }
    Println(
        "CHECK($narrowed_rhs$ != nullptr) << \"$field$ must be a "
        "$NarrowedType$.\";");
    PrintToIr(ast, Action::kDef, *narrowed, RefOrVal::kRef, kind, lhs,
              narrowed_rhs);
    return;
  }

  PrintNullableToIr(ast, Action::kDef, field.type(), maybe_null, RefOrVal::kRef,
                    kind, lhs, rhs);
}

void AstToIrSourcePrinter::PrintLiteralAttrField(const AstDef& ast,
                                                 const NodeDef& node,
                                                 const FieldDef& field,
                                                 FieldKind op_kind) {
  const FieldIrMappingPb& mapping = *field.ir_mapping();

  auto literal_lhs = Symbol("mlir") + Symbol(mapping.literal_attr());
  auto operand_lhs = Symbol("mlir") + Symbol(mapping.operand());
  auto rhs = absl::StrCat("node->", field.name().ToCcVarName(), "()");

  auto vars = WithVars({
      {"literal_lhs", literal_lhs.ToCcVarName()},
      {"operand_lhs", operand_lhs.ToCcVarName()},
      {"rhs", rhs},
      {"field", field.name().ToCcVarName()},
  });

  Println("mlir::Attribute $literal_lhs$;");
  Println("mlir::Value $operand_lhs$;");

  if (mapping.has_literal_unless()) {
    auto vars = WithVars({
        {"literal_unless", Symbol(mapping.literal_unless()).ToCcVarName()},
    });
    Println("if (!node->$literal_unless$()) {");
  } else {
    LOG(FATAL) << field.name().ToCcVarName()
               << ": literal_attr without literal_unless is not supported.";
  }
  {
    auto indent = WithIndent();
    IfStmtPrinter if_printer(this);
    for (const std::string& type_name : mapping.literal_attr_types()) {
      bool exact = false;
      std::optional<size_t> member =
          FindMemberHolding(ast, field.type(), type_name, &exact);
      CHECK(member.has_value())
          << field.name().ToCcVarName() << ": literal_attr_types \""
          << type_name << "\" isn't held by any member of the field.";

      std::string member_rhs = rhs;
      if (field.type().IsA<VariantType>()) {
        member_rhs = absl::StrFormat("std::get<%zu>(%s)", *member, rhs);
      }
      std::string literal_rhs;
      std::string condition;
      std::string type_class =
          (Symbol(ast.lang_name()) + type_name).ToPascalCase();
      auto literal_var = (field.name() + Symbol(type_name)).ToCcVarName();
      if (field.type().IsA<VariantType>()) {
        condition = absl::StrFormat(
            "const auto *%s = %s.index() == %zu ? %s : nullptr", literal_var,
            rhs, *member,
            exact ? member_rhs
                  : absl::StrFormat("dynamic_cast<const %s *>(%s)", type_class,
                                    member_rhs));
      } else {
        condition = absl::StrFormat(
            "const auto *%s = %s", literal_var,
            exact ? member_rhs
                  : absl::StrFormat("dynamic_cast<const %s *>(%s)", type_class,
                                    member_rhs));
      }

      ClassType literal_type{Symbol(type_name), ast.lang_name()};
      if_printer.PrintCase({
          [&] { Print(condition); },
          [&] {
            PrintClassToIr(ast, Action::kAssign, literal_type, FIELD_KIND_ATTR,
                           literal_lhs, literal_var);
          },
      });
    }
    Println(" else {");
    {
      auto indent = WithIndent();
      std::vector<std::string> type_names{mapping.literal_attr_types().begin(),
                                          mapping.literal_attr_types().end()};
      auto vars = WithVars({
          {"literal_unless", Symbol(mapping.literal_unless()).ToCcVarName()},
          {"literal_types", absl::StrJoin(type_names, " or ")},
      });
      Println(
          "LOG(FATAL) << \"If $literal_unless$ is false, $field$ must be "
          "$literal_types$.\";");
    }
    Println("}");
  }
  Println("} else {");
  {
    auto indent = WithIndent();
    PrintToIr(ast, Action::kAssign, field.type(), RefOrVal::kRef,
              FieldIrKind(field, op_kind), operand_lhs, rhs);
  }
  Println("}");
}

void AstToIrSourcePrinter::PrintSplitField(const AstDef& ast,
                                           const FieldDef& field,
                                           bool regions) {
  absl::string_view ir_lang = ir_lang_name_.empty()
                                  ? ast.lang_name()
                                  : absl::string_view(ir_lang_name_);
  absl::Span<const FieldIrSplitPb> splits = field.ir_splits();
  const auto& variant = static_cast<const VariantType&>(field.type());
  MaybeNull maybe_null = OptionalnessToMaybeNull(field.optionalness());

  // Finds the split holding a variant member.
  auto find_split = [&](const ScalarType& member) -> const FieldIrSplitPb& {
    std::string member_name =
        static_cast<const ClassType&>(member).name().ToPascalCase();
    for (const FieldIrSplitPb& split : splits) {
      if (absl::c_linear_search(split.types(), member_name)) {
        return split;
      }
    }
    LOG(FATAL) << field.name().ToCcVarName() << ": variant member "
               << member_name << " isn't covered by any split.";
  };

  if (!regions) {
    // The operands are defined before the op is created.
    for (const FieldIrSplitPb& split : splits) {
      if (split.has_operand()) {
        auto vars = WithVars({
            {"lhs", (Symbol("mlir") + Symbol(split.operand())).ToCcVarName()},
        });
        Println("mlir::Value $lhs$;");
      }
    }
  }

  bool has_any = absl::c_any_of(splits, [&](const auto& split) {
    return regions ? split.has_region() : split.has_operand();
  });
  if (!has_any) {
    return;
  }

  auto rhs = absl::StrCat("node->", field.name().ToCcVarName(), "()");
  auto print_switch = [&](const std::string& rhs) {
    auto vars = WithVars({{"rhs", rhs}});
    Println("switch ($rhs$.index()) {");
    {
      auto indent = WithIndent();
      for (size_t i = 0; i != variant.types().size(); ++i) {
        const ScalarType& member = *variant.types()[i];
        const FieldIrSplitPb& split = find_split(member);
        if (regions ? !split.has_region() : !split.has_operand()) {
          continue;
        }

        auto member_rhs = absl::StrFormat("std::get<%zu>(%s)", i, rhs);
        auto vars = WithVars({{"i", std::to_string(i)}});
        Println("case $i$: {");
        {
          auto indent = WithIndent();
          if (!regions) {
            PrintToIr(ast, Action::kAssign, member, RefOrVal::kRef,
                      split.kind(), Symbol("mlir") + Symbol(split.operand()),
                      member_rhs);
          } else {
            auto lhs = Symbol("mlir") + Symbol(split.region());
            auto vars = WithVars({
                {"lhs", lhs.ToCcVarName()},
                {"lhs_region", (lhs + "region").ToCcVarName()},
                {"mlirGetter", Symbol(split.region()).ToMlirGetter()},
                {"RegionEndOp",
                 GetRegionEndOp(split.kind(), /*is_list=*/false, ir_lang)
                     .ToPascalCase()},
            });
            Println("mlir::Region &$lhs_region$ = op.$mlirGetter$();");
            Println("AppendNewBlockAndPopulate(builder, $lhs_region$, [&] {");
            {
              auto indent = WithIndent();
              if (split.kind() == FIELD_KIND_STMT) {
                PrintToIr(ast, Action::kCreate, member, RefOrVal::kRef,
                          split.kind(), lhs, member_rhs);
              } else {
                PrintToIr(ast, Action::kDef, member, RefOrVal::kRef,
                          split.kind(), lhs, member_rhs);
                Println("CreateStmt<$RegionEndOp$>(builder, nullptr, $lhs$);");
              }
            }
            Println("});");
          }
          Println("break;");
        }
        Println("}");
      }
      Println("default:");
      Println("  break;");
    }
    Println("}");
  };

  switch (maybe_null) {
    case MaybeNull::kYes: {
      auto vars = WithVars({{"rhs", rhs}});
      Println("if ($rhs$.has_value()) {");
      {
        auto indent = WithIndent();
        print_switch(absl::StrCat(rhs, ".value()"));
      }
      Println("}");
      break;
    }
    case MaybeNull::kNo:
      print_switch(rhs);
      break;
  }
}

void AstToIrSourcePrinter::PrintRegion(const AstDef& ast, const NodeDef& node,
                                       const FieldDef& field,
                                       FieldKind op_kind) {
  if (FieldIsSplit(&field)) {
    return PrintSplitField(ast, field, /*regions=*/true);
  }

  MaybeNull maybe_null = OptionalnessToMaybeNull(field.optionalness());
  FieldKind kind = FieldIrKind(field, op_kind);

  auto lhs = Symbol("mlir") + field.name();
  auto lhs_region = lhs + "region";
  auto rhs = absl::StrCat("node->", field.name().ToCcVarName(), "()");
  absl::string_view ir_lang = ir_lang_name_.empty()
                                  ? ast.lang_name()
                                  : absl::string_view(ir_lang_name_);
  auto ir_name = Symbol(absl::StrCat(ir_lang, "ir"));

  auto vars = WithVars({
      {"lhs", lhs.ToCcVarName()},
      {"lhs_region", lhs_region.ToCcVarName()},
      {"mlirGetter", FieldIrName(field).ToMlirGetter()},
      {"rhs", rhs},
  });

  auto populate_region = [&] {
    Println("mlir::Region &$lhs_region$ = op.$mlirGetter$();");
    Println("AppendNewBlockAndPopulate(builder, $lhs_region$, [&] {");
    {
      auto indent = WithIndent();

      Action action = [&] {
        switch (kind) {
          case FIELD_KIND_UNSPECIFIED:
            LOG(FATAL) << "Unspecified FieldKind.";
          case FIELD_KIND_ATTR:
            LOG(FATAL) << "Unsupported FieldKind: " << kind;
          case FIELD_KIND_RVAL:
          case FIELD_KIND_LVAL: {
            return Action::kDef;
          }
          case FIELD_KIND_STMT: {
            return Action::kCreate;
          }
        }
      }();

      Symbol region_end_op =
          GetRegionEndOp(kind, field.type().IsA<ListType>(), ir_lang);
      PrintToIr(ast, action, field.type(), RefOrVal::kRef, kind, lhs, rhs);

      auto vars = WithVars({
          {"RegionEndOp", region_end_op.ToPascalCase()},
      });

      switch (action) {
        case Action::kAssign:
          LOG(FATAL) << "Unsupported Action: Assign.";
        case Action::kCreate:
          break;
        case Action::kDef: {
          Println("CreateStmt<$RegionEndOp$>(builder, nullptr, $lhs$);");
          break;
        }
      }
    }
    Println("});");
  };

  switch (maybe_null) {
    case MaybeNull::kYes: {
      Println("if ($rhs$.has_value()) {");
      {
        auto indent = WithIndent();
        absl::StrAppend(&rhs, ".value()");
        auto vars = WithVars({
            {"rhs", rhs},
        });
        populate_region();
      }
      Println("}");
      break;
    }
    case MaybeNull::kNo:
      populate_region();
      break;
  }
}

void AstToIrSourcePrinter::PrintBuiltinToIr(const AstDef& ast, Action action,
                                            const BuiltinType& type,
                                            const Symbol& lhs,
                                            const std::string& rhs) {
  absl::string_view ir_lang = ir_lang_name_.empty()
                                  ? ast.lang_name()
                                  : absl::string_view(ir_lang_name_);
  auto vars = WithVars({
      {"mlir_type", type.CcMlirBuilderType(ir_lang, FIELD_KIND_ATTR)},
      {"lhs", lhs.ToCcVarName()},
      {"rhs", rhs},
  });

  switch (action) {
    case Action::kDef:
      Print("$mlir_type$ ");
      ABSL_FALLTHROUGH_INTENDED;
    case Action::kAssign:
      Print("$lhs$ = ");
      break;
    case Action::kCreate:
      break;
  }

  switch (type.builtin_kind()) {
    case BuiltinTypeKind::kBool: {
      Print("builder.getBoolAttr($rhs$)");
      break;
    }
    case BuiltinTypeKind::kInt64: {
      Print("builder.getI64IntegerAttr($rhs$)");
      break;
    }
    case BuiltinTypeKind::kString: {
      Print("builder.getStringAttr($rhs$)");
      break;
    }
    case BuiltinTypeKind::kDouble: {
      Print("builder.getF64FloatAttr($rhs$)");
      break;
    }
  }

  Println(";");
}

void AstToIrSourcePrinter::PrintClassToIr(const AstDef& ast, Action action,
                                          const ClassType& type, FieldKind kind,
                                          const Symbol& lhs,
                                          const std::string& rhs) {
  absl::string_view ir_lang = ir_lang_name_.empty()
                                  ? ast.lang_name()
                                  : absl::string_view(ir_lang_name_);
  auto vars = WithVars({
      {"ClassName", type.name().ToPascalCase()},
      {"lhs", lhs.ToCcVarName()},
      {"rhs", rhs},
  });

  switch (action) {
    case Action::kDef: {
      auto vars = WithVars({
          {"cc_mlir_type", type.CcMlirBuilderType(ir_lang, kind)},
      });
      Print("$cc_mlir_type$ ");
      ABSL_FALLTHROUGH_INTENDED;
    }
    case Action::kAssign:
      Print("$lhs$ = ");
      break;
    case Action::kCreate:
      break;
  }

  switch (kind) {
    case FIELD_KIND_UNSPECIFIED:
    case FIELD_KIND_ATTR:
      Println("Visit$ClassName$Attr(builder, $rhs$);");
      break;
    case FIELD_KIND_RVAL:
    case FIELD_KIND_STMT: {
      Println("Visit$ClassName$(builder, $rhs$);");
      break;
    }
    case FIELD_KIND_LVAL: {
      Println("Visit$ClassName$Ref(builder, $rhs$);");
      break;
    }
  }
}

void AstToIrSourcePrinter::PrintClassToIr(const AstDef& ast, Action action,
                                          const ClassType& type,
                                          RefOrVal ref_or_val, FieldKind kind,
                                          const Symbol& lhs,
                                          const std::string& rhs) {
  switch (ref_or_val) {
    case RefOrVal::kRef:
      return PrintClassToIr(ast, action, type, kind, lhs, rhs);
    case RefOrVal::kVal:
      return PrintClassToIr(ast, action, type, kind, lhs,
                            absl::StrCat(rhs, ".get()"));
  }
}

void AstToIrSourcePrinter::PrintEnumToIr(const AstDef& ast, Action action,
                                         const EnumType& type,
                                         const Symbol& lhs,
                                         const std::string& rhs) {
  auto enum_name = (Symbol(ast.lang_name()) + type.name()).ToPascalCase();
  auto rhs_str = absl::StrCat(enum_name, "ToString(", rhs, ")");

  BuiltinType string_type{BuiltinTypeKind::kString, ast.lang_name()};
  return PrintBuiltinToIr(ast, action, string_type, lhs, rhs_str);
}

void AstToIrSourcePrinter::PrintVariantToIr(const AstDef& ast, Action action,
                                            const VariantType& type,
                                            RefOrVal ref_or_val, FieldKind kind,
                                            const Symbol& lhs,
                                            const std::string& rhs) {
  absl::string_view ir_lang = ir_lang_name_.empty()
                                  ? ast.lang_name()
                                  : absl::string_view(ir_lang_name_);
  auto vars = WithVars({
      {"lhs", lhs.ToCcVarName()},
      {"rhs", rhs},
  });

  Action case_action;
  switch (action) {
    case Action::kDef: {
      auto vars = WithVars({
          {"cc_mlir_type", type.CcMlirBuilderType(ir_lang, kind)},
      });
      Println("$cc_mlir_type$ $lhs$;");
      case_action = Action::kAssign;
      break;
    }
    case Action::kAssign:
      case_action = Action::kAssign;
      break;
    case Action::kCreate:
      case_action = Action::kCreate;
      break;
  }

  Println("switch ($rhs$.index()) {");
  {
    auto indent = WithIndent();

    for (size_t i = 0; i != type.types().size(); ++i) {
      auto vars = WithVars({
          {"i", std::to_string(i)},
      });

      Println("case $i$: {");
      {
        auto indent = WithIndent();
        const ScalarType& scalar_type = *type.types()[i];
        PrintToIr(ast, case_action, scalar_type, ref_or_val, kind, lhs,
                  absl::StrFormat("std::get<%zu>(%s)", i, rhs));
        Println("break;");
      }

      Println("}");
    }

    Println("default:");
    Println("  LOG(FATAL) << \"Unreachable code.\";");
  }
  Println("}");
}

void AstToIrSourcePrinter::PrintListToIr(const AstDef& ast, Action action,
                                         const ListType& type, FieldKind kind,
                                         const Symbol& lhs,
                                         const std::string& rhs) {
  absl::string_view ir_lang = ir_lang_name_.empty()
                                  ? ast.lang_name()
                                  : absl::string_view(ir_lang_name_);
  const auto lhs_element = Symbol("mlir_element");
  const auto rhs_element = "element";

  auto vars = WithVars({
      {"lhs", lhs.ToCcVarName()},
      {"lhs_data", (lhs + "data").ToCcVarName()},
      {"rhs", rhs},
      {"lhs_element", lhs_element.ToCcVarName()},
      {"rhs_element", rhs_element},
  });

  switch (kind) {
    case FIELD_KIND_UNSPECIFIED:
      LOG(FATAL) << "FieldKind unspecified.";
    case FIELD_KIND_STMT: {
      // Case: List of Statements.
      CHECK(action == Action::kCreate)
          << "We never collect statement ops in a vector.";

      Println("for (const auto &$rhs_element$ : *$rhs$) {");
      {
        auto indent = WithIndent();
        PrintNullableToIr(ast, Action::kCreate, type.element_type(),
                          type.element_maybe_null(), RefOrVal::kVal, kind,
                          lhs_element, rhs_element);
      }
      Println("}");
      break;
    }
    case FIELD_KIND_ATTR: {
      // Case: List of Attributes.
      //
      // We first create and fill a std::vector<mlir::Attribute> and then
      // convert it into a mlir::ArrayAttr (what the builder takes).

      Println("std::vector<mlir::Attribute> $lhs_data$;");
      Println("for (const auto &$rhs_element$ : *$rhs$) {");
      {
        auto indent = WithIndent();
        PrintNullableToIr(ast, Action::kDef, type.element_type(),
                          type.element_maybe_null(), RefOrVal::kVal, kind,
                          lhs_element, rhs_element);
        Println("$lhs_data$.push_back(std::move($lhs_element$));");
      }
      Println("}");

      switch (action) {
        case Action::kDef: {
          Println("auto $lhs$ = builder.getArrayAttr($lhs_data$);");
          break;
        }
        case Action::kAssign: {
          Println("$lhs$ = builder.getArrayAttr($lhs_data$);");
          break;
        }
        case Action::kCreate:
          LOG(FATAL) << "We never put attributes in a region.";
      }
      break;
    }

    case FIELD_KIND_LVAL:
    case FIELD_KIND_RVAL: {
      // Case: List of Values.
      //
      // We create and fill a std::vector<mlir::Value> which can be implicitly
      // converted to a mlir::ValueRange (what the builder takes).

      switch (action) {
        case Action::kDef:
          Println("std::vector<mlir::Value> $lhs$;");
          break;
        case Action::kAssign:
          // Do nothing.
          break;
        case Action::kCreate:
          LOG(FATAL) << "We must put expressions in a vector.";
      }

      Println("for (const auto &$rhs_element$ : *$rhs$) {");
      {
        auto indent = WithIndent();
        switch (type.element_maybe_null()) {
          case MaybeNull::kNo: {
            PrintToIr(ast, Action::kDef, type.element_type(), RefOrVal::kVal,
                      kind, lhs_element, rhs_element);
            break;
          }

          case MaybeNull::kYes: {
            // Unfortunately, in the std::vector<mlir::Value> we can't have any
            // nullptr. In order to represent optional, we need the special
            // <Lang>irNoneOp.

            Println("mlir::Value $lhs_element$;");
            Println("if ($rhs_element$.has_value()) {");
            {
              auto indent = WithIndent();
              PrintToIr(ast, Action::kAssign, type.element_type(),
                        RefOrVal::kVal, kind, lhs_element,
                        absl::StrCat(rhs_element, ".value()"));
            }
            Println("} else {");
            {
              auto indent = WithIndent();
              auto none_op = Symbol(absl::StrCat(ir_lang, "ir")) + "NoneOp";
              auto vars = WithVars({
                  {"NoneOp", none_op.ToPascalCase()},
              });

              Println("$lhs_element$ = CreateExpr<$NoneOp$>(builder, node);");
            }
            Println("}");

            break;
          }
        }

        Println("$lhs$.push_back(std::move($lhs_element$));");
      }

      Println("}");
    }
  }
}

void AstToIrSourcePrinter::PrintToIr(const AstDef& ast, Action action,
                                     const Type& type, RefOrVal ref_or_val,
                                     FieldKind kind, const Symbol& lhs,
                                     const std::string& rhs) {
  switch (type.kind()) {
    case TypeKind::kBuiltin: {
      const auto& builtin_type = static_cast<const BuiltinType&>(type);
      return PrintBuiltinToIr(ast, action, builtin_type, lhs, rhs);
    }

    case TypeKind::kClass: {
      const auto& class_type = static_cast<const ClassType&>(type);
      return PrintClassToIr(ast, action, class_type, ref_or_val, kind, lhs,
                            rhs);
    }

    case TypeKind::kEnum: {
      const auto& enum_type = static_cast<const EnumType&>(type);
      return PrintEnumToIr(ast, action, enum_type, lhs, rhs);
    }

    case TypeKind::kVariant: {
      const auto& variant_type = static_cast<const VariantType&>(type);
      return PrintVariantToIr(ast, action, variant_type, ref_or_val, kind, lhs,
                              rhs);
    }

    case TypeKind::kList: {
      const auto& list_type = static_cast<const ListType&>(type);
      CHECK(ref_or_val == RefOrVal::kRef);
      return PrintListToIr(ast, action, list_type, kind, lhs, rhs);
    }
  }
}

void AstToIrSourcePrinter::PrintNullableToIr(const AstDef& ast, Action action,
                                             const Type& type,
                                             MaybeNull maybe_null,
                                             RefOrVal ref_or_val,
                                             FieldKind kind, const Symbol& lhs,
                                             const std::string& rhs) {
  absl::string_view ir_lang = ir_lang_name_.empty()
                                  ? ast.lang_name()
                                  : absl::string_view(ir_lang_name_);
  auto vars = WithVars({
      {"lhs", lhs.ToCcVarName()},
      {"rhs", rhs},
  });

  switch (maybe_null) {
    case MaybeNull::kYes: {
      Action non_null_action;
      switch (action) {
        case Action::kAssign:
          non_null_action = Action::kAssign;
          break;
        case Action::kCreate:
          non_null_action = Action::kCreate;
          break;
        case Action::kDef: {
          auto vars = WithVars({
              {"mlir_type", type.CcMlirBuilderType(ir_lang, kind)},
          });
          Println("$mlir_type$ $lhs$;");
          non_null_action = Action::kAssign;
          break;
        }
      }
      Println("if ($rhs$.has_value()) {");
      {
        auto indent = WithIndent();
        auto new_rhs = absl::StrCat(rhs, ".value()");
        PrintToIr(ast, non_null_action, type, ref_or_val, kind, lhs, new_rhs);
      }
      Println("}");
      break;
    }

    case MaybeNull::kNo: {
      PrintToIr(ast, action, type, ref_or_val, kind, lhs, rhs);
      break;
    }
  }
}

std::string PrintAstToIrSource(const AstDef& ast,
                               absl::string_view cc_namespace,
                               absl::string_view ast_path,
                               absl::string_view ir_path,
                               absl::string_view ir_lang_name,
                               absl::string_view header_include_path,
                               absl::string_view class_name) {
  std::string str;
  {
    google::protobuf::io::StringOutputStream os(&str);
    AstToIrSourcePrinter printer(&os, ir_lang_name, header_include_path,
                                 class_name);
    printer.PrintAst(ast, cc_namespace, ast_path, ir_path);
  }

  return str;
}

}  // namespace maldoca
