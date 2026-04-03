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
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "maldoca/astgen/ast_def.h"
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

  std::vector<Symbol> traits;
  for (const NodeDef* parent : node.parents()) {
    if (!absl::c_linear_search(parent->aggregated_kinds(), kind)) {
      continue;
    }
    auto parent_ir_op_name = parent->ir_op_name(ast.lang_name(), kind);
    if (!parent_ir_op_name.has_value()) {
      continue;
    }
    traits.push_back(*parent_ir_op_name + "Traits");
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
    if (field.enclose_in_region()) {
      continue;
    }

    switch (field.kind()) {
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
    traits.push_back(Symbol("AttrSizedOperandSegments"));
  }

  if (absl::c_any_of(node.aggregated_fields(), FieldIsRegion)) {
    traits.push_back(Symbol("NoTerminator"));
  }

  for (auto mlir_trait : node.aggregated_additional_mlir_traits()) {
    switch (mlir_trait) {
      case MLIR_TRAIT_INVALID:
        LOG(FATAL) << "Invalid MlirTrait.";
      case MLIR_TRAIT_PURE:
        traits.push_back(Symbol("Pure"));
        break;
      case MLIR_TRAIT_ISOLATED_FROM_ABOVE:
        traits.push_back(Symbol("IsolatedFromAbove"));
        break;
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

      for (const Symbol& trait : traits) {
        auto vars = WithVars({
            {"Trait", trait.ToPascalCase()},
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
    if (node.has_fold()) {
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

          separator_printer.Print();
          PrintArgument(ast, node, *field);
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

          separator_printer.Print();
          PrintRegion(ast, node, *field);
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

void IrTableGenPrinter::PrintArgument(const AstDef& ast, const NodeDef& node,
                                      const FieldDef& field) {
  auto vars = WithVars({
      {"type", field.type().TdType(field.optionalness(), field.kind())},
      {"name", field.name().ToCcVarName()},
  });
  Print("$type$: $$$name$");
}

void IrTableGenPrinter::PrintRegion(const AstDef& ast, const NodeDef& node,
                                    const FieldDef& field) {
  std::string region_type = [&] {
    switch (field.kind()) {
      case FIELD_KIND_UNSPECIFIED:
        LOG(FATAL) << "FieldKind is unspecified.";
      case FIELD_KIND_ATTR:
        LOG(FATAL) << "Region of attributes not supported.";
      case FIELD_KIND_LVAL:
      case FIELD_KIND_RVAL:
        if (field.type().IsA<ListType>()) {
          return "ExprsRegion";
        } else {
          return "ExprRegion";
        }
      case FIELD_KIND_STMT:
        if (field.type().IsA<ListType>()) {
          return "StmtsRegion";
        } else {
          return "StmtRegion";
        }
    }
  }();

  switch (field.optionalness()) {
    case OPTIONALNESS_UNSPECIFIED:
      LOG(FATAL) << "Optionalness unspecified.";
    case OPTIONALNESS_REQUIRED:
      break;
    case OPTIONALNESS_MAYBE_NULL:
    case OPTIONALNESS_MAYBE_UNDEFINED:
      region_type = absl::StrCat("OptionalRegion<", region_type, ">");
  }

  auto vars = WithVars({
      {"name", field.name().ToCcVarName()},
      {"RegionType", region_type},
  });

  Print("$RegionType$: $$$name$");
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
