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

#include "maldoca/astgen/ir_table_gen_printer.h"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
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

void IrTableGenPrinter::PrintAst(const AstDef& ast, absl::string_view ir_path) {
  PrintLicense();
  Println();

  PrintCodeGenerationWarning();
  Println();

  // E.g. lang_name == "js", then ir_name == "jsir".
  const auto ir_name = absl::StrCat(ast.lang_name(), "ir");

  // E.g. "<ir_path>/jsir_ops.generated.td".
  const auto td_path = absl::StrCat(ir_path, "/", ir_name, "_ops.generated.td");

  PrintEnterHeaderGuard(td_path);
  Println();

  std::vector<std::string> imports = {
      "mlir/Interfaces/ControlFlowInterfaces.td",
      "mlir/Interfaces/InferTypeOpInterface.td",
      "mlir/Interfaces/LoopLikeInterface.td",
      "mlir/Interfaces/SideEffectInterfaces.td",
      "mlir/IR/OpBase.td",
      "mlir/IR/SymbolInterfaces.td",
      absl::StrCat(ir_path, "/interfaces.td"),
      absl::StrCat(ir_path, "/", ast.lang_name(), "ir_dialect.td"),
      absl::StrCat(ir_path, "/", ast.lang_name(), "ir_types.td"),
  };
  for (const auto& import : imports) {
    Println(absl::StrCat("include \"", import, "\""));
  }
  Println();

  bool has_expr_region = false;
  bool has_exprs_region = false;
  for (const auto* node : ast.topological_sorted_nodes()) {
    for (const auto* field : node->aggregated_fields()) {
      if (FieldIsSplit(field)) {
        for (const FieldIrSplitPb& split : field->ir_splits()) {
          if (split.has_region() && (split.kind() == FIELD_KIND_LVAL ||
                                     split.kind() == FIELD_KIND_RVAL)) {
            has_expr_region = true;
          }
        }
        continue;
      }
      if (!field->enclose_in_region()) {
        continue;
      }
      if (field->kind() != FIELD_KIND_LVAL &&
          field->kind() != FIELD_KIND_RVAL) {
        continue;
      }
      if (field->type().IsA<ListType>()) {
        has_exprs_region = true;
      } else {
        has_expr_region = true;
      }
    }
  }

  const auto region_end_comment = UnIndentedSource(R"(
// $ir$.*_region_end: An artificial op at the end of a region to collect
// expression-related values.
//
// Take $ir$.exprs_region_end as example:
// ======================================
//
// Consider the following function declaration:
// ```
// function foo(arg1, arg2 = defaultValue) {
//   ...
// }
// ```
//
// We lower it to the following IR (simplified):
// ```
// %0 = $ir$.identifier_ref {"foo"}
// $ir$.function_declaration(%0) (
//   // params
//   {
//     %1 = $ir$.identifier_ref {"a"}
//     %2 = $ir$.identifier_ref {"b"}
//     %3 = $ir$.identifier {"defaultValue"}
//     %4 = $ir$.assignment_pattern_ref(%2, %3)
//     $ir$.exprs_region_end(%1, %4)
//   },
//   // body
//   {
//     ...
//   }
// )
// ```
//
// We can see that:
//
// 1. We put the parameter-related ops in a region, instead of taking them as
//    normal arguments. In other words, we don't do this:
//
//    ```
//    %0 = $ir$.identifier_ref {"foo"}
//    %1 = $ir$.identifier_ref {"a"}
//    %2 = $ir$.identifier_ref {"b"}
//    %3 = $ir$.identifier {"defaultValue"}
//    %4 = $ir$.assignment_pattern_ref(%2, %3)
//    $ir$.function_declaration(%0, [%1, %4]) (
//      // body
//      {
//        ...
//      }
//    )
//    ```
//
//    The reason is that sometimes an argument might have a default value, and
//    the evaluation of that default value happens once for each function call
//    (i.e. it happens "within" the function). If we take the parameter as
//    normal argument, then %3 is only evaluated once - at function definition
//    time.
//
// 2. Even though the function has two parameters, we use 4 ops to represent
//    them. This is because some parameters are more complex and require more
//    than one op.
//
// 3. We use "$ir$.exprs_region_end" to list the "top-level" ops for the
//    parameters. In the example above, ops [%2, %3, %4] all represent the
//    parameter "b = defaultValue", but %4 is the top-level one. In other words,
//    %4 is the root of the tree [%2, %3, %4].
//
// 4. Strictly speaking, we don't really need "$ir$.exprs_region_end". The ops
//    within the "params" region form several trees, and we can figure out what
//    the roots are (a root is an op whose return value is not used by any other
//    op). So the use of "$ir$.exprs_region_end" is mostly for convenience.
  )");

  if (has_expr_region || has_exprs_region) {
    Symbol ir{absl::StrCat(ast.lang_name(), "ir")};

    auto vars = WithVars({
        {"ir", ir.ToSnakeCase()},
        {"Ir", ir.ToPascalCase()},
    });
    Println(region_end_comment);

    if (has_expr_region) {
      const auto expr_region_end = UnIndentedSource(R"(
        def $Ir$ExprRegionEndOp : $Ir$_Op<"expr_region_end", [Terminator]> {
          let arguments = (ins
            AnyType: $$argument
          );
        }
      )");
      Println(expr_region_end);
      Println();
    }

    if (has_exprs_region) {
      const auto exprs_region_end = UnIndentedSource(R"(
        def $Ir$ExprsRegionEndOp : $Ir$_Op<"exprs_region_end", [Terminator]> {
          let arguments = (ins
            Variadic<AnyType>: $$arguments
          );
        }
      )");
      Println(exprs_region_end);
      Println();
    }
  }

  for (const auto* node : ast.topological_sorted_nodes()) {
    if (!node->should_generate_ir_op()) {
      continue;
    }

    for (auto kind : node->aggregated_kinds()) {
      PrintNode(ast, *node, kind);
    }
  }

  PrintExitHeaderGuard(td_path);
}

void IrTableGenPrinter::PrintNode(const AstDef& ast, const NodeDef& node,
                                  FieldKind kind) {
  auto ir_name = absl::StrCat(ast.lang_name(), "ir");
  auto hir_name =
      absl::StrCat(ast.lang_name(), node.has_control_flow() ? "hir" : "ir");

  auto vars = WithVars({
      {"OpName", node.ir_op_name(ast.lang_name(), kind).value().ToPascalCase()},
      {"op_mnemonic", node.ir_op_mnemonic(kind).value().ToCcVarName()},
      {"Name", node.name()},
      {"name", Symbol(node.name()).ToCcVarName()},
      {"IrName", Symbol(ir_name).ToPascalCase()},
      {"HirName", Symbol(hir_name).ToPascalCase()},
  });

  std::vector<std::string> traits;
  for (const NodeDef* parent : node.parents()) {
    if (!absl::c_linear_search(parent->aggregated_kinds(), kind)) {
      continue;
    }
    auto parent_ir_op_name = parent->ir_op_name(ast.lang_name(), kind);
    if (!parent_ir_op_name.has_value()) {
      continue;
    }
    traits.push_back((*parent_ir_op_name + "Traits").ToPascalCase());
  }

  // When there is more than one variadic operand, we must append the
  // AttrSizedOperandSegments trait. This is because MLIR internally stores
  // operands as a single array and without additional information, it cannot
  // attributes ranges of that array into the corresponding variadic operands.
  //
  // MLIR doesn't allow universally adding AttrSizedOperandSegments - only ops
  // with more than one variadic operand are allowed.
  //
  // See: https://mlir.llvm.org/docs/OpDefinitions/#variadic-operands
  size_t num_variadic_operands = 0;
  for (const FieldDef& field : node.fields()) {
    if (!FieldIsArgument(&field)) {
      continue;
    }

    // Operands of a split or a literal attribute are always optional.
    if (FieldIsSplit(&field)) {
      for (const FieldIrSplitPb& split : field.ir_splits()) {
        if (split.has_operand()) {
          num_variadic_operands++;
        }
      }
      continue;
    }
    if (FieldHasLiteralAttr(&field)) {
      num_variadic_operands++;
      continue;
    }

    switch (FieldIrKind(field, kind)) {
      case FIELD_KIND_UNSPECIFIED: {
        LOG(QFATAL) << node.name() << "::" << field.name().ToCcVarName()
                    << ": FieldKind unspecified.";
      }
      case FIELD_KIND_ATTR:
      case FIELD_KIND_STMT: {
        break;
      }
      case FIELD_KIND_LVAL:
      case FIELD_KIND_RVAL: {
        if (field.type().IsA<ListType>() ||
            field.optionalness() == OPTIONALNESS_MAYBE_NULL ||
            field.optionalness() == OPTIONALNESS_MAYBE_UNDEFINED) {
          num_variadic_operands++;
        }
      }
    }
  }
  if (num_variadic_operands > 1) {
    traits.push_back("AttrSizedOperandSegments");
  }

  if (absl::c_any_of(node.aggregated_fields(), FieldIsRegion)) {
    traits.push_back("NoTerminator");
  }

  for (auto mlir_trait : node.aggregated_additional_mlir_traits()) {
    switch (mlir_trait) {
      case MLIR_TRAIT_INVALID:
        LOG(FATAL) << "Invalid MlirTrait.";
      case MLIR_TRAIT_PURE:
        traits.push_back("Pure");
        break;
      case MLIR_TRAIT_ISOLATED_FROM_ABOVE:
        traits.push_back("IsolatedFromAbove");
        break;
      case MLIR_TRAIT_JSIR_FUNCTION_OP_INTERFACE:
        traits.push_back("JsirFunctionOpInterfaceTraits");
        break;
    }
  }

  // Fields held in several arguments/regions require that exactly one (or at
  // most one, if the field is optional) of them is set.
  for (const FieldDef* field : node.aggregated_fields()) {
    std::optional<std::string> trait =
        GetExclusiveSlotsTrait(ast, node, *field, kind);
    if (trait.has_value()) {
      traits.push_back(*std::move(trait));
    }
  }

  if (traits.empty()) {
    Println("def $OpName$ : $IrName$_Op<\"$op_mnemonic$\", []> {");
  } else {
    // Example:
    // ```
    // def JsirBinaryExpressionOp : Jsir_Op<
    //     "binary_expression", [
    //         DeclareOpInterfaceMethods<JsirNodeOpInterface>,
    //         DeclareOpInterfaceMethods<JsirExpressionOpInterface>
    //     ]> {
    // ```
    Print(
        "def $OpName$ : $HirName$_Op<\n"
        "    \"$op_mnemonic$\", [\n");

    {
      auto indent = WithIndent(8);
      TabPrinter tab_printer{{
          .print_separator = [&] { Print(",\n"); },
      }};

      for (const std::string& trait : traits) {
        auto vars = WithVars({
            {"Trait", trait},
        });

        tab_printer.Print();
        Print("$Trait$");
      }
    }

    Println("\n    ]> {");
  }
  {
    auto indent = WithIndent();
    TabPrinter line_separator_printer{{
        .print_separator = [&] { Print("\n"); },
    }};
    if (node.has_fold(kind)) {
      line_separator_printer.Print();
      Println("let hasFolder = 1;");
    }

    if (absl::c_any_of(node.aggregated_fields(), FieldIsArgument)) {
      line_separator_printer.Print();

      Println("let arguments = (ins");
      {
        auto indent = WithIndent();
        TabPrinter separator_printer{{
            .print_separator = [&] { Print(",\n"); },
        }};
        for (const auto* field : node.aggregated_fields()) {
          if (!FieldIsArgument(field)) {
            continue;
          }

          for (const std::string& argument :
               GetArguments(ast, node, *field, kind)) {
            separator_printer.Print();
            auto vars = WithVars({{"argument", argument}});
            Print("$argument$");
          }
        }
      }
      Println();
      Println(");");
    }

    if (absl::c_any_of(node.aggregated_fields(), FieldIsRegion)) {
      line_separator_printer.Print();

      Println("let regions = (region");
      {
        auto indent = WithIndent();
        TabPrinter separator_printer{{
            .print_separator = [&] { Print(",\n"); },
        }};
        for (const auto* field : node.aggregated_fields()) {
          if (!FieldIsRegion(field)) {
            continue;
          }

          for (const std::string& region :
               GetRegions(ast, node, *field, kind)) {
            separator_printer.Print();
            auto vars = WithVars({{"region", region}});
            Print("$region$");
          }
        }
      }
      Println();
      Println(");");
    }

    // Only expressions have results.
    if (kind == FIELD_KIND_LVAL || kind == FIELD_KIND_RVAL) {
      line_separator_printer.Print();

      Println("let results = (outs");
      Println("  $IrName$AnyType");
      Println(");");
    }
  }

  Println("}");
  Println();
}

namespace {

std::string RegionTdType(FieldKind kind, bool is_list) {
  switch (kind) {
    case FIELD_KIND_UNSPECIFIED:
      LOG(FATAL) << "FieldKind is unspecified.";
    case FIELD_KIND_ATTR:
      LOG(FATAL) << "Region of attributes not supported.";
    case FIELD_KIND_LVAL:
    case FIELD_KIND_RVAL:
      return is_list ? "ExprsRegion" : "ExprRegion";
    case FIELD_KIND_STMT:
      return is_list ? "StmtsRegion" : "StmtRegion";
  }
}

}  // namespace

std::vector<std::string> IrTableGenPrinter::GetArguments(const AstDef& ast,
                                                         const NodeDef& node,
                                                         const FieldDef& field,
                                                         FieldKind op_kind) {
  std::vector<std::string> arguments;
  if (FieldIsSplit(&field)) {
    for (const FieldIrSplitPb& split : field.ir_splits()) {
      if (!split.has_operand()) {
        continue;
      }
      arguments.push_back(absl::StrCat("Optional<AnyType>: $",
                                       Symbol(split.operand()).ToCcVarName()));
    }
    return arguments;
  }

  if (FieldHasLiteralAttr(&field)) {
    const FieldIrMappingPb& mapping = *field.ir_mapping();
    std::vector<std::string> attr_types;
    for (const std::string& type_name : mapping.literal_attr_types()) {
      attr_types.push_back(ast.nodes()
                               .at(type_name)
                               ->ir_op_name(ast.lang_name(), FIELD_KIND_ATTR)
                               .value()
                               .ToPascalCase());
    }
    // Always an AnyAttrOf<>, even for a single type, so that the attribute is
    // built and read as an mlir::Attribute regardless of the number of types.
    std::string attr_type =
        absl::StrCat("AnyAttrOf<[", absl::StrJoin(attr_types, ", "), "]>");
    arguments.push_back(
        absl::StrCat("OptionalAttr<", attr_type, ">: $",
                     Symbol(mapping.literal_attr()).ToCcVarName()));
    arguments.push_back(absl::StrCat("Optional<AnyType>: $",
                                     Symbol(mapping.operand()).ToCcVarName()));
    return arguments;
  }

  arguments.push_back(absl::StrCat(
      field.type().TdType(field.optionalness(), FieldIrKind(field, op_kind)),
      ": $", FieldIrName(field).ToCcVarName()));
  return arguments;
}

std::vector<std::string> IrTableGenPrinter::GetRegions(const AstDef& ast,
                                                       const NodeDef& node,
                                                       const FieldDef& field,
                                                       FieldKind op_kind) {
  std::vector<std::string> regions;
  if (FieldIsSplit(&field)) {
    for (const FieldIrSplitPb& split : field.ir_splits()) {
      if (!split.has_region()) {
        continue;
      }
      regions.push_back(absl::StrCat(
          "OptionalRegion<", RegionTdType(split.kind(), /*is_list=*/false),
          ">: $", Symbol(split.region()).ToCcVarName()));
    }
    return regions;
  }

  std::string region_type =
      RegionTdType(FieldIrKind(field, op_kind), field.type().IsA<ListType>());

  switch (field.optionalness()) {
    case OPTIONALNESS_UNSPECIFIED:
      LOG(FATAL) << "Optionalness unspecified.";
    case OPTIONALNESS_REQUIRED:
      break;
    case OPTIONALNESS_MAYBE_NULL:
    case OPTIONALNESS_MAYBE_UNDEFINED:
      region_type = absl::StrCat("OptionalRegion<", region_type, ">");
  }

  regions.push_back(
      absl::StrCat(region_type, ": $", FieldIrName(field).ToCcVarName()));
  return regions;
}

std::optional<std::string> IrTableGenPrinter::GetExclusiveSlotsTrait(
    const AstDef& ast, const NodeDef& node, const FieldDef& field,
    FieldKind op_kind) {
  if (!FieldIsInIrOp(&field) || field.derived_in_ir()) {
    return std::nullopt;
  }

  std::string op_name =
      node.ir_op_name(ast.lang_name(), op_kind).value().ToPascalCase();
  auto getter = [&](absl::string_view name) {
    return absl::StrCat("::llvm::cast<", op_name, ">(&$_op).",
                        Symbol(name).ToMlirGetter());
  };

  // (slot name, C++ condition that the slot is set).
  std::vector<std::pair<std::string, std::string>> slots;
  bool exactly_one;
  if (FieldIsSplit(&field)) {
    for (const FieldIrSplitPb& split : field.ir_splits()) {
      if (split.has_operand()) {
        slots.push_back(
            {Symbol(split.operand()).ToCcVarName(),
             absl::StrCat(getter(split.operand()), "() != nullptr")});
      } else {
        slots.push_back(
            {Symbol(split.region()).ToCcVarName(),
             absl::StrCat("!", getter(split.region()), "().empty()")});
      }
    }
    exactly_one = field.optionalness() == OPTIONALNESS_REQUIRED;
  } else if (FieldHasLiteralAttr(&field)) {
    const FieldIrMappingPb& mapping = *field.ir_mapping();
    slots.push_back(
        {Symbol(mapping.literal_attr()).ToCcVarName(),
         absl::StrCat(getter(mapping.literal_attr()), "Attr() != nullptr")});
    slots.push_back({Symbol(mapping.operand()).ToCcVarName(),
                     absl::StrCat(getter(mapping.operand()), "() != nullptr")});
    exactly_one = true;
  } else {
    return std::nullopt;
  }

  std::vector<std::string> names;
  std::vector<std::string> conditions;
  for (const auto& [name, condition] : slots) {
    names.push_back(name);
    conditions.push_back(absl::StrCat("int(", condition, ")"));
  }

  return absl::StrCat("PredOpTrait<\"", exactly_one ? "exactly" : "at most",
                      " one of ", absl::StrJoin(names, ", "),
                      " is set\", CPred<\"(", absl::StrJoin(conditions, " + "),
                      ") ", exactly_one ? "==" : "<=", " 1\">>");
}

std::string PrintIrTableGen(const AstDef& ast, absl::string_view ir_path) {
  std::string str;
  {
    google::protobuf::io::StringOutputStream os(&str);
    IrTableGenPrinter printer(&os);
    printer.PrintAst(ast, ir_path);
  }

  return str;
}

}  // namespace maldoca
