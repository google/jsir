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

#include <string>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/container/btree_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
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

  PrintIncludeHeader(
      absl::StrCat(ir_path, "/conversion/", ast.lang_name(), "ir_to_ast.h"));
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
      "absl/status/statusor.h",
      "absl/strings/str_cat.h",
      "absl/types/optional.h",
      "absl/types/variant.h",
      "maldoca/astgen/ir_to_ast_util.h",
      "maldoca/base/status_macros.h",
      std::string(ast_header_path),
      absl::StrCat(ir_path, "/ir.h"),
  });
  Println();

  PrintEnterNamespace(cc_namespace);
  Println();

  for (const auto* node : ast.topological_sorted_nodes()) {
    if (!node->children().empty()) {
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
  auto ir_op_name = node.ir_op_name(ast.lang_name(), kind);
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
  auto ir_name = Symbol(absl::StrCat(ast.lang_name(), "ir"));
  auto visitor = GetVisitor(node, kind);

  auto vars = WithVars({
      {"InputType", input_type},
      {"BaseName",
       kind == FIELD_KIND_ATTR ? "mlir::Attribute" : "mlir::Operation*"},
      {"Name", (Symbol(ast.lang_name()) + node.name()).ToPascalCase()},
      {"name", kind == FIELD_KIND_ATTR ? "attr" : "op"},
      {"IrName", ir_name.ToPascalCase()},
      {"Visitor", visitor.ToPascalCase()},
  });

  Println("absl::StatusOr<std::unique_ptr<$Name$>>");
  Println("$IrName$ToAst::$Visitor$($InputType$ $name$) {");
  {
    auto indent = WithIndent();
    Println("using Ret = absl::StatusOr<std::unique_ptr<$Name$>>;");
    Println("return llvm::TypeSwitch<$BaseName$, Ret>($name$)");
    {
      auto indent = WithIndent();
      for (const NodeDef* leaf : node.leafs()) {
        auto vars = WithVars({
            {"LeafOpName",
             leaf->ir_op_name(ast.lang_name(), kind)->ToPascalCase()},
            {"LeafVisitor", GetVisitor(*leaf, kind).ToPascalCase()},
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
  auto ir_op_name = node.ir_op_name(ast.lang_name(), kind).value();
  auto ir_name = Symbol(absl::StrCat(ast.lang_name(), "ir"));

  auto visitor = Symbol("Visit") + node.name();
  if (kind == FIELD_KIND_LVAL) {
    visitor += "Ref";
  }

  auto vars = WithVars({
      {"OpName", ir_op_name.ToPascalCase()},
      {"Name", (Symbol(ast.lang_name()) + node.name()).ToPascalCase()},
      {"name", kind == FIELD_KIND_ATTR ? "attr" : "op"},
      {"IrName", ir_name.ToPascalCase()},
      {"Visitor", visitor.ToPascalCase()},
  });

  Println("absl::StatusOr<std::unique_ptr<$Name$>>");
  Println("$IrName$ToAst::$Visitor$($OpName$ $name$) {");
  {
    auto indent = WithIndent();
    for (const auto* field : node.aggregated_fields()) {
      if (FieldIsArgument(field)) {
        PrintField(ast, node, *field);
      } else if (FieldIsRegion(field)) {
        PrintRegion(ast, node, *field);
      }
    }

    // Call the constructor.
    Print("return Create<$Name$>(\n");
    {
      auto indent = WithIndent(4);
      Print("$name$");

      for (const FieldDef* field : node.aggregated_fields()) {
        if (!FieldIsArgument(field) && !FieldIsRegion(field)) {
          continue;
        }

        auto vars = WithVars({
            {"field_name", field->name().ToCcVarName()},
        });
        Print(",\nstd::move($field_name$)");
      }
    }

    Println(");");
  }
  Println("}");
  Println();
}

void IrToAstSourcePrinter::PrintField(const AstDef& ast, const NodeDef& node,
                                      const FieldDef& field) {
  MaybeNull maybe_null = OptionalnessToMaybeNull(field.optionalness());

  auto mlir_getter = field.name().ToMlirGetter();

  std::string rhs;
  switch (field.kind()) {
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

  auto vars = WithVars({
      {"lhs", field.name().ToCcVarName()},
      {"rhs", rhs},
  });

  Println("MALDOCA_ASSIGN_OR_RETURN(");
  {
    auto indent = WithIndent(4);
    Println("auto $lhs$,");
    Print("Convert(\n");
    {
      auto indent = WithIndent(4);
      Println("$rhs$,");
      PrintConverter(ast, field.type(), ast.lang_name(), field.kind(),
                     maybe_null);
      Println();
    }
    Println(")");
  }
  Println(");");
}

void IrToAstSourcePrinter::PrintRegion(const AstDef& ast, const NodeDef& node,
                                       const FieldDef& field) {
  MaybeNull maybe_null = OptionalnessToMaybeNull(field.optionalness());

  std::string converter_type = [&]() -> std::string {
    switch (field.kind()) {
      case FIELD_KIND_UNSPECIFIED:
        LOG(FATAL) << "Unspecified FieldKind.";
      case FIELD_KIND_ATTR:
        LOG(FATAL) << "Unsupported FieldKind: " << field.kind();
      case FIELD_KIND_RVAL:
      case FIELD_KIND_LVAL: {
        if (field.type().IsA<ListType>()) {
          auto end_op =
              Symbol(absl::StrCat(ast.lang_name(), "ir")) + "ExprsRegionEndOp";
          return absl::StrCat("ExprsRegion<", end_op.ToPascalCase(), ">");
        } else {
          auto end_op =
              Symbol(absl::StrCat(ast.lang_name(), "ir")) + "ExprRegionEndOp";
          return absl::StrCat("ExprRegion<", end_op.ToPascalCase(), ">");
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
      {"mlirGetter", field.name().ToMlirGetter()},
  });

  Println("MALDOCA_ASSIGN_OR_RETURN(");
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
              PrintConverter(ast, field.type(), ast.lang_name(), field.kind(),
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
            PrintConverter(ast, field.type(), ast.lang_name(), field.kind(),
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
  if (maybe_null == MaybeNull::kYes) {
    if (kind == FIELD_KIND_LVAL || kind == FIELD_KIND_RVAL) {
      auto none_op = Symbol(absl::StrCat(ast.lang_name(), "ir")) + "NoneOp";
      Print(absl::StrCat("Nullable<", none_op.ToPascalCase(), ">(\n"));
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
                               absl::string_view ir_path) {
  std::string str;
  {
    google::protobuf::io::StringOutputStream os(&str);
    IrToAstSourcePrinter printer(&os);
    printer.PrintAst(ast, cc_namespace, ast_path, ir_path);
  }

  return str;
}

}  // namespace maldoca
