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

#include "maldoca/astgen/ast_gen.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "maldoca/base/path.h"
#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "maldoca/astgen/ast_def.h"
#include "maldoca/astgen/ast_def.pb.h"
#include "maldoca/astgen/symbol.h"
#include "maldoca/astgen/type.h"
#include "google/protobuf/io/printer.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"

namespace maldoca {
namespace {

constexpr absl::string_view kOsValueVariableName = "os";
constexpr absl::string_view kJsonValueVariableName = "json";

std::string GetAstHeaderPath(absl::string_view ast_path) {
  return JoinPath(ast_path, "ast.generated.h");
}

// FieldIs{Argument,Region}:
//
// If a field has ignore_in_ir(), then we don't define anything in the op.
//
// Example: Node::start does not lead to any argument/region in JSIR because we
// want to store the information in mlir::Location.
//
// If a field has enclose_in_region(), then it's an MLIR "region"; otherwise
// it's an MLIR "argument".
//
// An argument is either an mlir::Attribute or an mlir::Value;
// A region is an mlir::Region.
//
// See FieldDefPb::enclose_in_region for why we need to enclose certain fields
// in a region.
bool FieldIsArgument(const FieldDef *field) {
  return !field->ignore_in_ir() && !field->enclose_in_region();
}

bool FieldIsRegion(const FieldDef *field) {
  return !field->ignore_in_ir() && field->enclose_in_region();
}

// Gets the name of the *RegionEndOp.
// - For an lval or rval (expression): <Ir>ExprRegionEndOp.
// - For a list of lvals or rvals (expressions): <Ir>ExprsRegionEndOp.
Symbol GetRegionEndOp(const AstDef &ast, const FieldDef &field) {
  auto ir_name = Symbol(absl::StrCat(ast.lang_name(), "ir"));

  Symbol region_end_op;
  switch (field.kind()) {
    case FIELD_KIND_UNSPECIFIED:
      LOG(FATAL) << "Unspecified FieldKind.";
    case FIELD_KIND_ATTR:
      LOG(FATAL) << "Unsupported FieldKind: " << field.kind();
    case FIELD_KIND_RVAL:
    case FIELD_KIND_LVAL: {
      if (field.type().IsA<ListType>()) {
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

struct TabPrinterOptions {
  std::function<void()> print_prefix = nullptr;
  std::function<void()> print_separator = nullptr;
  std::function<void()> print_postfix = nullptr;
};

class TabPrinter : private TabPrinterOptions {
 public:
  explicit TabPrinter(TabPrinterOptions options)
      : TabPrinterOptions(std::move(options)) {}

  ~TabPrinter() {
    if (!is_first_) {
      if (print_postfix) {
        print_postfix();
      }
    }
  }

  void Print() {
    if (is_first_) {
      if (print_prefix) {
        print_prefix();
      }
      is_first_ = false;
    } else {
      if (print_separator) {
        print_separator();
      }
    }
  }

 private:
  bool is_first_ = true;
};

// Consistently unindent lines of code so that the outmost line has no
// indentation.
//
// Example:
//
// Input:
// ```
//   abc
//     abc
//    abc
// ```
//
// Output:
// ```
// abc
//   abc
//  abc
// ```
std::string UnIndentedSource(absl::string_view source) {
  source = absl::StripTrailingAsciiWhitespace(source);

  std::vector<std::string> lines = absl::StrSplit(source, '\n');

  // Remove leading empty lines.
  lines.erase(lines.begin(), absl::c_find_if(lines, [](const auto &line) {
                return !line.empty();
              }));

  size_t min_indent = absl::c_accumulate(
      lines, std::numeric_limits<size_t>::max(),
      [](size_t current_min, const std::string &line) {
        size_t first_non_whitespace = line.find_first_not_of(' ');
        if (first_non_whitespace == std::string::npos) {
          return current_min;
        }
        return std::min(current_min, first_non_whitespace);
      });

  for (auto &line : lines) {
    if (line.size() >= min_indent) {
      line.erase(0, min_indent);
    }
  }

  return absl::StrJoin(lines, "\n");
}

}  // namespace

// =============================================================================
// TsInterfacePrinter
// =============================================================================

void TsInterfacePrinter::PrintAst(const AstDef &ast) {
  for (const EnumDef &enum_def : ast.enum_defs()) {
    PrintEnum(enum_def, ast.lang_name());
    Println();
  }

  for (const auto &name : ast.node_names()) {
    const NodeDef &node = *ast.nodes().at(name);
    PrintNode(node);
    Println();
  }
}

void TsInterfacePrinter::PrintEnum(const EnumDef &enum_def,
                                   absl::string_view lang_name) {
  auto vars = WithVars({
      {"EnumName", enum_def.name().ToPascalCase()},
  });

  Println("type $EnumName$ =");
  {
    auto indent = WithIndent(4);
    for (const EnumMemberDef &member : enum_def.members()) {
      auto vars = WithVars({
          {"string_value", absl::CEscape(member.string_value())},
      });

      Println("| \"$string_value$\"");
    }
  }
}

void TsInterfacePrinter::PrintNode(const NodeDef &node) {
  auto vars = WithVars({
      {"NodeType", node.name()},
  });
  Print("interface $NodeType$");

  if (!node.parents().empty()) {
    Print(" <: ");

    TabPrinter separator_printer{{
        .print_separator = [&] { Print(", "); },
    }};
    for (const NodeDef *parent : node.parents()) {
      separator_printer.Print();
      Print(parent->name());
    }
  }

  Println(" {");
  {
    auto indent = WithIndent();
    for (const FieldDef &field : node.fields()) {
      PrintFieldDef(field);
    }
  }
  Println("}");
}

void TsInterfacePrinter::PrintFieldDef(const FieldDef &field) {
  Print(field.name().ToCamelCase());

  if (field.optionalness() == OPTIONALNESS_MAYBE_UNDEFINED) {
    Print("?");
  }

  Print(": ");

  MaybeNull maybe_null = field.optionalness() == OPTIONALNESS_MAYBE_NULL
                             ? MaybeNull::kYes
                             : MaybeNull::kNo;
  Print(field.type().JsType(maybe_null));

  Println();
}

std::string PrintTsInterface(const AstDef &ast) {
  std::string ts_interface;
  {
    google::protobuf::io::StringOutputStream os(&ts_interface);
    TsInterfacePrinter printer(&os);
    printer.PrintAst(ast);
  }
  return ts_interface;
}

// =============================================================================
// CcPrinterBase
// =============================================================================

void CcPrinterBase::PrintLicense() {
  static const auto *kCcLicenceString = new std::string{UnIndentedSource(R"cc(
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
  )cc")};

  Println(kCcLicenceString->c_str());
}

void CcPrinterBase::PrintEnterNamespace(absl::string_view cc_namespace) {
  for (absl::string_view cc_namespace_piece :
       absl::StrSplit(cc_namespace, "::")) {
    auto vars = WithVars({
        {"cc_namespace_piece", std::string(cc_namespace_piece)},
    });
    Println("namespace $cc_namespace_piece$ {");
  }
}

void CcPrinterBase::PrintExitNamespace(absl::string_view cc_namespace) {
  std::vector<absl::string_view> pieces = absl::StrSplit(cc_namespace, "::");
  for (auto it = pieces.rbegin(); it != pieces.rend(); ++it) {
    auto vars = WithVars({
        {"cc_namespace_piece", std::string(*it)},
    });
    Println("}  // namespace $cc_namespace_piece$");
  }
}

static std::string ToHeaderGuard(absl::string_view header_path) {
  std::string header_guard = absl::AsciiStrToUpper(header_path);
  absl::StrReplaceAll({{"/", "_"}, {".", "_"}}, &header_guard);
  absl::StrAppend(&header_guard, "_");
  return header_guard;
}

void CcPrinterBase::PrintEnterHeaderGuard(absl::string_view header_path) {
  auto vars = WithVars({
      {"HEADER_GUARD", ToHeaderGuard(header_path)},
  });

  Println("#ifndef $HEADER_GUARD$");
  Println("#define $HEADER_GUARD$");
}

void CcPrinterBase::PrintExitHeaderGuard(absl::string_view header_path) {
  auto vars = WithVars({
      {"HEADER_GUARD", ToHeaderGuard(header_path)},
  });

  Println("#endif  // $HEADER_GUARD$");
}

void CcPrinterBase::PrintIncludeHeader(absl::string_view header_path) {
  auto vars = WithVars({
      {"header_path", std::string(header_path)},
  });

  Println("#include \"$header_path$\"");
}

void CcPrinterBase::PrintIncludeHeaders(std::vector<std::string> header_paths) {
  for (absl::string_view header_path : header_paths) {
    PrintIncludeHeader(header_path);
  }
}

void CcPrinterBase::PrintTitle(absl::string_view title) {
  std::vector<std::string> commented_lines;
  for (absl::string_view line : absl::StrSplit(title, '\n')) {
    if (line.empty()) {
      commented_lines.push_back("//");
    } else {
      commented_lines.push_back(absl::StrCat("// ", line));
    }
  }
  std::string commented_title = absl::StrJoin(commented_lines, "\n");

  auto vars = WithVars({
      {"CommentedTitle", commented_title},
  });

  static const auto *kCode = new std::string(absl::StripAsciiWhitespace(R"(
// =============================================================================
$CommentedTitle$
// =============================================================================
  )"));

  Println(kCode->c_str());
}

void CcPrinterBase::PrintCodeGenerationWarning() {
  PrintTitle("STOP!! DO NOT MODIFY!! THIS FILE IS AUTOMATICALLY GENERATED.");
}

// =============================================================================
// AstHeaderPrinter
// =============================================================================

void AstHeaderPrinter::PrintAst(const AstDef &ast,
                                absl::string_view cc_namespace,
                                absl::string_view ast_path) {
  auto header_path = GetAstHeaderPath(ast_path);

  PrintLicense();
  Println();

  PrintCodeGenerationWarning();
  Println();

  PrintEnterHeaderGuard(header_path);
  Println();

  Println("// IWYU pragma: begin_keep");
  Println("// NOLINTBEGIN(whitespace/line_length)");
  Println("// clang-format off");
  Println();

  Println("#include <optional>");
  Println("#include <string>");
  Println("#include <variant>");
  Println("#include <vector>");
  Println();

  PrintIncludeHeader("absl/status/statusor.h");
  PrintIncludeHeader("absl/strings/string_view.h");
  PrintIncludeHeader("nlohmann/json.hpp");
  Println();

  PrintEnterNamespace(cc_namespace);
  Println();

  for (const EnumDef &enum_def : ast.enum_defs()) {
    PrintEnum(enum_def, ast.lang_name());
    Println();
  }

  for (const NodeDef *node : ast.topological_sorted_nodes()) {
    PrintNode(*node, ast.lang_name());
    Println();
  }

  Println("// clang-format on");
  Println("// NOLINTEND(whitespace/line_length)");
  Println("// IWYU pragma: end_keep");
  Println();

  PrintExitNamespace(cc_namespace);
  Println();

  PrintExitHeaderGuard(header_path);
}

void AstHeaderPrinter::PrintEnum(const EnumDef &enum_def,
                                 absl::string_view lang_name) {
  auto vars = WithVars({
      {"EnumName", (Symbol(lang_name) + enum_def.name()).ToPascalCase()},
      {"enum_name", enum_def.name().ToSnakeCase()},
  });

  Println("enum class $EnumName$ {");
  {
    auto indent = WithIndent();
    for (const EnumMemberDef &member : enum_def.members()) {
      auto vars = WithVars({
          {"kMemberName", (Symbol("k") + member.name()).ToCamelCase()},
      });

      Println("$kMemberName$,");
    }
  }
  Println("};");
  Println();

  Println("absl::string_view $EnumName$ToString($EnumName$ $enum_name$);");
  Println(
      "absl::StatusOr<$EnumName$> StringTo$EnumName$(absl::string_view s);");
}

void AstHeaderPrinter::PrintNode(const NodeDef &node,
                                 absl::string_view lang_name) {
  auto vars = WithVars({
      {"NodeType", (Symbol(lang_name) + node.name()).ToPascalCase()},
      {"json_variable", kJsonValueVariableName},
      {"os_variable", kOsValueVariableName},
  });

  if (node.node_type_enum().has_value()) {
    PrintEnum(*node.node_type_enum().value(), lang_name);
    Println();
  }

  Print("class $NodeType$");
  if (!node.parents().empty()) {
    Print(" : ");
    TabPrinter separator_printer{{
        .print_separator = [&] { Print(", "); },
    }};
    for (const NodeDef *parent : node.parents()) {
      auto vars = WithVars({
          {"BaseType", (Symbol(lang_name) + parent->name()).ToPascalCase()},
      });

      separator_printer.Print();
      Print("public virtual $BaseType$");
    }
  }
  Println(" {");

  // Always print "public:" because the declaration of FromJson() always
  // exists.
  Println(" public:");
  {
    auto indent = WithIndent();

    // Constructor
    if (!node.aggregated_fields().empty()) {
      PrintConstructor(node, lang_name);
      Println();
    }

    // Destructor
    if (node.parents().empty() && !node.children().empty()) {
      Println("virtual ~$NodeType$() = default;");
      Println();
    }

    // Get type enum.
    if (node.node_type_enum().has_value()) {
      auto node_type_enum_name = node.node_type_enum().value()->name();
      auto vars = WithVars({
          {"NodeTypeEnum",
           (Symbol(lang_name) + node_type_enum_name).ToPascalCase()},
          {"node_type_enum", node_type_enum_name.ToCcVarName()},
      });

      Println("virtual $NodeTypeEnum$ $node_type_enum$() const = 0;");
      Println();

    } else if (node.children().empty()) {
      for (const NodeDef *ancestor : node.ancestors()) {
        if (!ancestor->node_type_enum().has_value()) {
          continue;
        }

        auto root_type_enum_name = ancestor->node_type_enum().value()->name();
        auto vars = WithVars({
            {"RootTypeEnum",
             (Symbol(lang_name) + root_type_enum_name).ToPascalCase()},
            {"root_type_enum", root_type_enum_name.ToCcVarName()},
            {"NodeTypeNoLang", Symbol(node.name()).ToPascalCase()},
        });

        Println("$RootTypeEnum$ $root_type_enum$() const override {");
        Println("  return $RootTypeEnum$::k$NodeTypeNoLang$;");
        Println("}");
        Println();
      }
    }

    // Serialize
    if (node.parents().empty()) {
      if (node.children().empty()) {
        // Non-virtual.
        Println("void Serialize(std::ostream& $os_variable$) const;");
        Println();
      } else {
        // Virtual base.
        // We define a pure virtual function here, and override it in leaf
        // types.
        Println(
            "virtual void Serialize(std::ostream& $os_variable$) "
            "const = 0;");
        Println();
      }
    } else {
      if (node.children().empty()) {
        // Leaf type.
        // We override the virtual function.
        Println(
            "void Serialize(std::ostream& $os_variable$) "
            "const override;");
        Println();
      } else {
        // Non-leaf type - skipped.
        // We only override in leaf types. Here it's still pure virtual.
      }
    }

    // FromJson
    Println(
        "static absl::StatusOr<std::unique_ptr<$NodeType$>> FromJson("
        "const nlohmann::json& $json_variable$);");
    Println();

    // Getters and setters.
    for (const FieldDef &field : node.fields()) {
      PrintGetterSetterDeclarations(field, lang_name);
      Println();
    }
  }

  Println(" protected:");
  {
    auto indent = WithIndent();

    // SerializeFields
    Println("// Internal function used by Serialize().");
    Println("// Sets the fields defined in this class.");
    Println("// Does not set fields defined in ancestors.");
    Println(
        "void SerializeFields(std::ostream& $os_variable$, "
        "bool &needs_comma) const;");

    // Get<FieldName>FromJson() functions.
    if (!node.fields().empty()) {
      Println();
      Println("// Internal functions used by FromJson().");
      Println("// Extracts a field from a JSON object.");
      for (const FieldDef &field : node.fields()) {
        PrintGetFromJson(field, lang_name);
      }
    }
  }

  // Print member variables.
  if (!node.fields().empty()) {
    Println();
    Println(" private:");
    {
      auto indent = WithIndent();
      for (const FieldDef &field : node.fields()) {
        PrintMemberVariable(field, lang_name);
      }
    }
  }

  Println("};");
}

void AstHeaderPrinter::PrintConstructor(const NodeDef &node,
                                        absl::string_view lang_name) {
  auto vars = WithVars({
      {"NodeType", (Symbol(lang_name) + node.name()).ToPascalCase()},
  });
  Print("explicit $NodeType$(");
  if (!node.aggregated_fields().empty()) {
    Println();
    {
      auto indent = WithIndent(4);
      TabPrinter separator_printer{{
          .print_separator = [this] { Print(",\n"); },
      }};
      for (const FieldDef *field : node.aggregated_fields()) {
        auto vars = WithVars({
            {"cc_type", CcType(*field)},
            {"field_name", field->name().ToCcVarName()},
        });

        separator_printer.Print();
        Print("$cc_type$ $field_name$");
      }
    }
  }
  Println(");");
}

void AstHeaderPrinter::PrintGetterSetterDeclarations(
    const FieldDef &field, absl::string_view lang_name) {
  std::string cc_getter_type = CcMutableGetterType(field);
  std::string cc_const_getter_type = CcConstGetterType(field);

  auto vars = WithVars({
      {"cc_getter_type", cc_getter_type},
      {"cc_const_getter_type", cc_const_getter_type},
      {"cc_type", CcType(field)},
      {"field_name", field.name().ToCcVarName()},
  });

  // If the mutable getter would return the same type as the const getter, skip
  // the mutable getter.
  if (cc_getter_type != cc_const_getter_type) {
    Println("$cc_getter_type$ $field_name$();");
  }
  Println("$cc_const_getter_type$ $field_name$() const;");
  Println("void set_$field_name$($cc_type$ $field_name$);");
}

void AstHeaderPrinter::PrintMemberVariable(const FieldDef &field,
                                           absl::string_view lang_name) {
  auto vars = WithVars({
      {"cc_type", CcType(field)},
      {"field_name", field.name().ToCcVarName()},
  });

  Println("$cc_type$ $field_name$_;");
}

void AstHeaderPrinter::PrintGetFromJson(const FieldDef &field,
                                        absl::string_view lang_name) {
  auto vars = WithVars({
      {"cc_type", CcType(field)},
      {"FieldName", field.name().ToPascalCase()},
      {"os_variable", kOsValueVariableName},
  });

  Println(
      "static absl::StatusOr<$cc_type$> "
      "Get$FieldName$(const nlohmann::json& $json_variable$);");
}

std::string PrintAstHeader(const AstDef &ast, absl::string_view cc_namespace,
                           absl::string_view ast_path) {
  std::string str;
  {
    google::protobuf::io::StringOutputStream os(&str);
    AstHeaderPrinter printer(&os);
    printer.PrintAst(ast, cc_namespace, ast_path);
  }

  return str;
}

// =============================================================================
// AstSourcePrinter
// =============================================================================

void AstSourcePrinter::PrintAst(const AstDef &ast,
                                absl::string_view cc_namespace,
                                absl::string_view ast_path) {
  auto header_path = GetAstHeaderPath(ast_path);

  PrintLicense();
  Println();

  PrintCodeGenerationWarning();
  Println();

  PrintIncludeHeader(header_path);
  Println();

  Println("// IWYU pragma: begin_keep");
  Println("// NOLINTBEGIN(whitespace/line_length)");
  Println("// clang-format off");
  Println();

  Println("#include <cstdint>");
  Println("#include <memory>");
  Println("#include <optional>");
  Println("#include <string>");
  Println("#include <utility>");
  Println("#include <variant>");
  Println("#include <vector>");
  Println();

  PrintIncludeHeader("absl/container/flat_hash_map.h");
  PrintIncludeHeader("absl/memory/memory.h");
  PrintIncludeHeader("absl/log/log.h");
  PrintIncludeHeader("absl/status/status.h");
  PrintIncludeHeader("absl/status/statusor.h");
  PrintIncludeHeader("absl/strings/str_cat.h");
  PrintIncludeHeader("absl/strings/string_view.h");
  PrintIncludeHeader("nlohmann/json.hpp");
  PrintIncludeHeader("maldoca/base/status_macros.h");
  Println();

  PrintEnterNamespace(cc_namespace);
  Println();

  for (const EnumDef &enum_def : ast.enum_defs()) {
    PrintEnum(enum_def, ast.lang_name());
    Println();
  }

  for (const NodeDef *node : ast.topological_sorted_nodes()) {
    PrintNode(*node, ast.lang_name());
  }

  Println("// clang-format on");
  Println("// NOLINTEND(whitespace/line_length)");
  Println("// IWYU pragma: end_keep");
  Println();

  PrintExitNamespace(cc_namespace);
}

void AstSourcePrinter::PrintEnum(const EnumDef &enum_def,
                                 absl::string_view lang_name) {
  auto vars = WithVars({
      {"EnumName", (Symbol(lang_name) + enum_def.name()).ToPascalCase()},
      {"enum_name", enum_def.name().ToSnakeCase()},
  });

  Println("absl::string_view $EnumName$ToString($EnumName$ $enum_name$) {");
  {
    auto indent = WithIndent();
    Println("switch ($enum_name$) {");
    {
      auto indent = WithIndent();
      for (const EnumMemberDef &member : enum_def.members()) {
        auto vars = WithVars({
            {"kMemberName", (Symbol("k") + member.name()).ToCamelCase()},
            {"string_value", absl::CEscape(member.string_value())},
        });

        Println("case $EnumName$::$kMemberName$:");
        Println("  return \"$string_value$\";");
      }
    }
    Println("}");
  }
  Println("}");
  Println();

  Println(
      "absl::StatusOr<$EnumName$> StringTo$EnumName$(absl::string_view s) {");
  {
    auto indent = WithIndent();

    Println(
        "static const auto *kMap = "
        "new absl::flat_hash_map<absl::string_view, $EnumName$> {");
    {
      auto indent = WithIndent(4);
      for (const EnumMemberDef &member : enum_def.members()) {
        auto vars = WithVars({
            {"kMemberName", (Symbol("k") + member.name()).ToCamelCase()},
            {"string_value", absl::CEscape(member.string_value())},
        });

        Println("{\"$string_value$\", $EnumName$::$kMemberName$},");
      }
    }
    Println("};");
    Println();

    const auto code = UnIndentedSource(R"(
auto it = kMap->find(s);
if (it == kMap->end()) {
  return absl::InvalidArgumentError(absl::StrCat("Invalid string for $EnumName$: ", s));
}
return it->second;
    )");
    Println(code);
  }
  Println("}");
}

void AstSourcePrinter::PrintNode(const NodeDef &node,
                                 absl::string_view lang_name) {
  PrintTitle((Symbol(lang_name) + node.name()).ToPascalCase());
  Println();

  auto vars = WithVars({
      {"NodeType", ClassType(Symbol(node.name()), lang_name).CcType()},
  });

  if (node.node_type_enum().has_value()) {
    PrintEnum(*node.node_type_enum().value(), lang_name);
    Println();
  }

  if (!node.aggregated_fields().empty()) {
    PrintConstructor(node, lang_name);
    Println();
  }

  for (const FieldDef &field : node.fields()) {
    const Type &type = field.type();
    bool is_optional = field.optionalness() != OPTIONALNESS_REQUIRED;

    std::string cc_getter_type = CcMutableGetterType(field);
    std::string cc_const_getter_type = CcConstGetterType(field);

    auto vars = WithVars({
        {"NodeType", (Symbol(lang_name) + node.name()).ToPascalCase()},
        {"cc_getter_type", cc_getter_type},
        {"cc_const_getter_type", cc_const_getter_type},
        {"cc_type", CcType(field)},
        {"field_name", field.name().ToCcVarName()},
    });

    // If both the mutable getter and const getter would have the same return
    // type, then we just skip the mutable getter and only keep the const
    // getter.
    if (cc_getter_type != cc_const_getter_type) {
      Println("$cc_getter_type$ $NodeType$::$field_name$() {");
      {
        auto indent = WithIndent();
        PrintGetterBody(field.name(), type, is_optional);
      }
      Println("}");
      Println();
    }

    Println("$cc_const_getter_type$ $NodeType$::$field_name$() const {");
    {
      auto indent = WithIndent();
      PrintGetterBody(field.name(), type, is_optional);
    }
    Println("}");
    Println();

    Println("void $NodeType$::set_$field_name$($cc_type$ $field_name$) {");
    {
      auto indent = WithIndent();
      PrintSetterBody(field.name(), type, is_optional);
    }
    Println("}");
    Println();
  }
}

void AstSourcePrinter::PrintConstructor(const NodeDef &node,
                                        absl::string_view lang_name) {
  auto vars = WithVars({
      {"NodeType", (Symbol(lang_name) + node.name()).ToPascalCase()},
  });
  Print("$NodeType$::$NodeType$(");
  if (!node.aggregated_fields().empty()) {
    Println();
    auto indent = WithIndent(4);

    TabPrinter separator_printer{{
        .print_separator = [this] { Print(",\n"); },
    }};
    for (const FieldDef *field : node.aggregated_fields()) {
      auto vars = WithVars({
          {"cc_type", CcType(*field)},
          {"field_name", field->name().ToCcVarName()},
      });

      separator_printer.Print();
      Print("$cc_type$ $field_name$");
    }
  }
  Println(")");

  {
    auto indent = WithIndent(4);

    TabPrinter tab_printer{{
        .print_prefix =
            [&] {
              Print(": ");
              Indent();
            },
        .print_separator = [&] { Print(",\n"); },
        .print_postfix = [&] { Outdent(); },
    }};
    for (const NodeDef *ancestor : node.ancestors()) {
      tab_printer.Print();

      auto vars = WithVars({
          {"AncestorType",
           (Symbol(lang_name) + ancestor->name()).ToPascalCase()},
      });
      Print("$AncestorType$(");

      TabPrinter ancestor_tab_printer{{
          .print_separator = [&] { Print(", "); },
      }};
      for (const FieldDef *field : ancestor->aggregated_fields()) {
        ancestor_tab_printer.Print();

        auto vars = WithVars({
            {"field_name", field->name().ToCcVarName()},
        });
        Print("std::move($field_name$)");
      }

      Print(")");
    }

    for (const FieldDef &field : node.fields()) {
      auto vars = WithVars({
          {"field_name", field.name().ToCcVarName()},
      });

      tab_printer.Print();
      Print("$field_name$_(std::move($field_name$))");
    }
  }

  Println(" {}");
}

void AstSourcePrinter::PrintGetterBody(const std::string &cc_expr,
                                       const Type &type) {
  auto vars = WithVars({
      {"cc_expr", cc_expr},
  });

  switch (type.kind()) {
    case TypeKind::kBuiltin: {
      Println("return $cc_expr$;");
      break;
    }

    case TypeKind::kEnum: {
      Println("return $cc_expr$;");
      break;
    }

    case TypeKind::kClass: {
      Println("return $cc_expr$.get();");
      break;
    }

    case TypeKind::kVariant: {
      const auto &variant_type = static_cast<const VariantType &>(type);

      Println("switch ($cc_expr$.index()) {");
      {
        auto indent = WithIndent();

        for (size_t i = 0; i != variant_type.types().size(); ++i) {
          auto vars = WithVars({
              {"i", std::to_string(i)},
          });
          const ScalarType &type = *variant_type.types().at(i);

          Println("case $i$: {");
          {
            auto indent = WithIndent();
            PrintGetterBody(absl::StrFormat("std::get<%zu>(%s)", i, cc_expr),
                            type);
          }
          Println("}");
        }

        Println("default:");
        Println("  LOG(FATAL) << \"Unreachable code.\";");
      }
      Println("}");

      break;
    }

    case TypeKind::kList: {
      Println("return &$cc_expr$;");
      break;
    }
  }
}

void AstSourcePrinter::PrintGetterBody(const Symbol &field_name,
                                       const Type &type, bool is_optional) {
  if (is_optional) {
    auto vars = WithVars({
        {"field_name", field_name.ToCcVarName()},
    });

    Println("if (!$field_name$_.has_value()) {");
    Println("  return std::nullopt;");
    Println("} else {");
    {
      auto indent = WithIndent();
      auto value_cc_expr = absl::StrCat(field_name.ToCcVarName(), "_.value()");
      PrintGetterBody(value_cc_expr, type);
    }
    Println("}");

  } else {
    PrintGetterBody(absl::StrCat(field_name.ToCcVarName(), "_"), type);
  }
}

void AstSourcePrinter::PrintSetterBody(const Symbol &field_name,
                                       const Type &type, bool is_optional) {
  auto vars = WithVars({
      {"field_name", field_name.ToCcVarName()},
  });

  if (type.IsA<BuiltinType>()) {
    const auto &builtin_type = static_cast<const BuiltinType &>(type);
    switch (builtin_type.builtin_kind()) {
      case BuiltinTypeKind::kBool:
      case BuiltinTypeKind::kDouble:
        Println("$field_name$_ = $field_name$;");
        return;
      default:
        break;
    }
  }

  if (type.IsA<EnumType>()) {
    Println("$field_name$_ = $field_name$;");
    return;
  }

  Println("$field_name$_ = std::move($field_name$);");
}

std::string PrintAstSource(const AstDef &ast, absl::string_view cc_namespace,
                           absl::string_view ast_path) {
  std::string str;
  {
    google::protobuf::io::StringOutputStream os(&str);
    AstSourcePrinter printer(&os);
    printer.PrintAst(ast, cc_namespace, ast_path);
  }

  return str;
}

// =============================================================================
// AstSerializePrinter
// =============================================================================

void AstSerializePrinter::PrintAst(const AstDef &ast,
                                   absl::string_view cc_namespace,
                                   absl::string_view ast_path) {
  auto vars = WithVars({
      {"os_variable", kOsValueVariableName},
  });

  auto header_path = GetAstHeaderPath(ast_path);

  PrintLicense();
  Println();

  PrintCodeGenerationWarning();
  Println();

  Println("// IWYU pragma: begin_keep");
  Println("// NOLINTBEGIN(whitespace/line_length)");
  Println("// clang-format off");
  Println();

  Println("#include <cmath>");
  Println("#include <limits>");
  Println("#include <ostream>");
  Println("#include <string>");
  Println("#include <utility>");
  Println();

  PrintIncludeHeaders({
      std::string(header_path),
      "absl/log/log.h",
      "absl/memory/memory.h",
      "absl/status/status.h",
      "absl/strings/string_view.h",
      "nlohmann/json.hpp",
      "maldoca/base/status_macros.h",
  });
  Println();

  PrintEnterNamespace(cc_namespace);
  Println();

  Println(
      R"(void MaybeAddComma(std::ostream &$os_variable$, bool &needs_comma) {
  if (needs_comma) {
    $os_variable$ << ",";
  }
  needs_comma = true;
}
)");

  for (const auto &node : ast.topological_sorted_nodes()) {
    PrintTitle((Symbol(ast.lang_name()) + node->name()).ToPascalCase());
    Println();

    PrintSerializeFieldsFunction(*node, ast.lang_name());
    Println();

    if (node->children().empty()) {
      PrintSerializeFunction(*node, ast.lang_name());
      Println();
    }
  }

  Println("// clang-format on");
  Println("// NOLINTEND(whitespace/line_length)");
  Println("// IWYU pragma: end_keep");
  Println();

  PrintExitNamespace(cc_namespace);
}

void AstSerializePrinter::PrintBuiltinSerialize(const BuiltinType &type,
                                                const std::string &lhs,
                                                const std::string &rhs) {
  auto vars = WithVars({
      {"os_variable", kOsValueVariableName},
      {"lhs", lhs},
      {"rhs", rhs},
  });

  if (!lhs.empty()) {
    Println("$os_variable$ << $lhs$ << (nlohmann::json($rhs$)).dump();");
  } else {
    Println("$os_variable$ << (nlohmann::json($rhs$)).dump();");
  }
}

void AstSerializePrinter::PrintEnumSerialize(const EnumType &type,
                                             const std::string &lhs,
                                             const std::string &rhs,
                                             absl::string_view lang_name) {
  auto vars = WithVars({
      {"os_variable", kOsValueVariableName},
      {"lhs", lhs},
      {"rhs", rhs},
      {"EnumName", (Symbol(lang_name) + type.name()).ToPascalCase()},
  });

  if (!lhs.empty()) {
    Println(
        R"($os_variable$ << $lhs$ << "\"" << $EnumName$ToString($rhs$) << "\"";)");
  } else {
    Println(R"($os_variable$ << "\"" << $EnumName$ToString($rhs$) << "\"";)");
  }
}

void AstSerializePrinter::PrintClassSerialize(const ClassType &type,
                                              const std::string &lhs,
                                              const std::string &rhs) {
  auto vars = WithVars({
      {"os_variable", kOsValueVariableName},
      {"lhs", lhs},
      {"rhs", rhs},
  });

  if (!lhs.empty()) {
    Println("$os_variable$ << $lhs$;");
  }
  Println("$rhs$->Serialize($os_variable$);");
}

void AstSerializePrinter::PrintVariantSerialize(const VariantType &variant_type,
                                                const std::string &lhs,
                                                const std::string &rhs,
                                                absl::string_view lang_name) {
  auto vars = WithVars({
      {"lhs", lhs},
      {"rhs", rhs},
  });

  Println("switch ($rhs$.index()) {");
  {
    auto indent = WithIndent();
    for (size_t i = 0; i != variant_type.types().size(); ++i) {
      auto vars = WithVars({
          {"i", std::to_string(i)},
      });

      Println("case $i$: {");
      {
        auto indent = WithIndent();
        const ScalarType &type = *variant_type.types()[i];
        PrintSerialize(type, lhs, absl::StrFormat("std::get<%zu>(%s)", i, rhs),
                       lang_name);
        Println("break;");
      }

      Println("}");
    }

    Println("default:");
    Println("  LOG(FATAL) << \"Unreachable code.\";");
  }
  Println("}");
}

void AstSerializePrinter::PrintListSerialize(const ListType &list_type,
                                             const std::string &lhs,
                                             const std::string &rhs,
                                             absl::string_view lang_name) {
  constexpr char kRhsElement[] = "element";
  CHECK_NE(lhs, kRhsElement);
  CHECK_NE(rhs, kRhsElement);

  constexpr char kLhsElement[] = "element_json";
  CHECK_NE(lhs, kLhsElement);
  CHECK_NE(rhs, kLhsElement);

  auto vars = WithVars({
      {"os_variable", kOsValueVariableName},
      {"lhs", lhs},
      {"rhs", rhs},
      {"lhs_element", kLhsElement},
      {"rhs_element", kRhsElement},
  });

  if (!lhs.empty()) {
    Println(R"($os_variable$ << $lhs$ << "[";)");
  } else {
    Println(R"($os_variable$ << "[";)");
  }
  Println("{");
  {
    auto indent = WithIndent();

    Println("bool needs_comma = false;");
    Println("for (const auto& $rhs_element$ : $rhs$) {");
    {
      auto indent = WithIndent();
      Println("MaybeAddComma($os_variable$, needs_comma);");
      PrintNullableToJson(list_type.element_type(),
                          list_type.element_maybe_null(), "", kRhsElement,
                          lang_name);
    }
    Println("}");
  }
  Println("}");
  Println(R"($os_variable$ << "]";)");
}

void AstSerializePrinter::PrintSerialize(const Type &type,
                                         const std::string &lhs,
                                         const std::string &rhs,
                                         absl::string_view lang_name) {
  switch (type.kind()) {
    case TypeKind::kBuiltin: {
      const auto &builtin_type = static_cast<const BuiltinType &>(type);
      PrintBuiltinSerialize(builtin_type, lhs, rhs);
      break;
    }

    case TypeKind::kEnum: {
      const auto &enum_type = static_cast<const EnumType &>(type);
      PrintEnumSerialize(enum_type, lhs, rhs, lang_name);
      break;
    }

    case TypeKind::kClass: {
      const auto &class_type = static_cast<const ClassType &>(type);
      PrintClassSerialize(class_type, lhs, rhs);
      break;
    }

    case TypeKind::kVariant: {
      const auto &variant_type = static_cast<const VariantType &>(type);
      PrintVariantSerialize(variant_type, lhs, rhs, lang_name);
      break;
    }

    case TypeKind::kList: {
      const auto &list_type = static_cast<const ListType &>(type);
      PrintListSerialize(list_type, lhs, rhs, lang_name);
      break;
    }
  }
}

void AstSerializePrinter::PrintNullableToJson(const Type &type,
                                              MaybeNull maybe_null,
                                              const std::string &lhs,
                                              const std::string &rhs,
                                              absl::string_view lang_name) {
  switch (maybe_null) {
    case MaybeNull::kNo: {
      PrintSerialize(type, lhs, rhs, lang_name);
      break;
    }

    case MaybeNull::kYes: {
      auto vars = WithVars({
          {"os_variable", kOsValueVariableName},
          {"lhs", lhs},
          {"rhs", rhs},
      });

      Println("if ($rhs$.has_value()) {");
      {
        auto indent = WithIndent();
        auto rhs_value = absl::StrCat(rhs, ".value()");
        PrintSerialize(type, lhs, rhs_value, lang_name);
      }
      Println("} else {");
      {
        auto indent = WithIndent();

        if (!lhs.empty()) {
          Println(R"($os_variable$ << $lhs$ << "null";)");
        } else {
          Println(R"($os_variable$ << "null";)");
        }
      }
      Println("}");
      break;
    }
  }
}

void AstSerializePrinter::PrintSerializeFieldsFunction(
    const NodeDef &node, absl::string_view lang_name) {
  auto vars = WithVars({
      {"NodeType", (Symbol(lang_name) + node.name()).ToPascalCase()},
      {"os_variable", kOsValueVariableName},
  });

  Println(
      "void $NodeType$::SerializeFields(std::ostream& $os_variable$, "
      "bool &needs_comma) const {");
  {
    auto indent = WithIndent();

    for (const FieldDef &field : node.fields()) {
      // E.g. "\"fieldName\":"
      auto lhs = absl::StrFormat(R"("\"%s\":")", field.name().ToCamelCase());

      // E.g. field_name_
      auto rhs = absl::StrCat(field.name().ToCcVarName(), "_");

      switch (field.optionalness()) {
        case OPTIONALNESS_UNSPECIFIED: {
          LOG(FATAL) << "Invalid Optionalness. Should be a bug.";
          break;
        }

        case OPTIONALNESS_REQUIRED: {
          Println("MaybeAddComma($os_variable$, needs_comma);");
          PrintSerialize(field.type(), lhs, rhs, lang_name);
          break;
        }

        case OPTIONALNESS_MAYBE_UNDEFINED: {
          auto vars = WithVars({
              {"rhs", rhs},
          });

          // If <rhs> == std::nullopt, the assignment does not happen.
          Println("if ($rhs$.has_value()) {");
          {
            auto indent = WithIndent();
            auto rhs_value = absl::StrCat(rhs, ".value()");
            Println("MaybeAddComma($os_variable$, needs_comma);");
            PrintSerialize(field.type(), lhs, rhs_value, lang_name);
          }
          Println("}");

          break;
        }
        case OPTIONALNESS_MAYBE_NULL: {
          Println("MaybeAddComma($os_variable$, needs_comma);");
          PrintNullableToJson(field.type(), MaybeNull::kYes, lhs, rhs,
                              lang_name);
          break;
        }
      }
    }
  }
  Println("}");
}

void AstSerializePrinter::PrintSerializeFunction(const NodeDef &node,
                                                 absl::string_view lang_name) {
  auto vars = WithVars({
      {"NodeType", (Symbol(lang_name) + node.name()).ToPascalCase()},
      {"NodeTypeNoLangName", node.name()},
      {"os_variable", kOsValueVariableName},
  });

  Println("void $NodeType$::Serialize(std::ostream& $os_variable$) const {");
  {
    auto indent = WithIndent();

    Println(R"($os_variable$ << "{";)");
    Println("{");
    {
      auto indent = WithIndent();
      Println("bool needs_comma = false;");

      // The "type" field.
      if (!node.parents().empty() || !node.children().empty()) {
        Println("MaybeAddComma($os_variable$, needs_comma);");
        Println(R"($os_variable$ << "\"type\":\"$NodeTypeNoLangName$\"";)");
      }

      // Assign fields of ancestors of this node.
      for (const NodeDef *ancestor : node.ancestors()) {
        auto vars = WithVars({
            {"AncestorType",
             (Symbol(lang_name) + ancestor->name()).ToPascalCase()},
        });
        Println(
            "$AncestorType$::SerializeFields($os_variable$, "
            "needs_comma);");
      }

      // Assign fields of the node itself.
      Println("$NodeType$::SerializeFields($os_variable$, needs_comma);");
    }
    Println("}");

    Println(R"($os_variable$ << "}";)");
  }
  Println("}");
}

std::string PrintAstToJson(const AstDef &ast, absl::string_view cc_namespace,
                           absl::string_view ast_path) {
  std::string str;
  {
    google::protobuf::io::StringOutputStream os(&str);
    AstSerializePrinter printer(&os);
    printer.PrintAst(ast, cc_namespace, ast_path);
  }

  return str;
}

// =============================================================================
// AstFromJsonPrinter
// =============================================================================

// Helper for printing an if-statement.
//
// Usage:
//  IfStmtPrinter printer(...);
//  printer.PrintCase({
//      [&] {
//        PrintConditionHere();
//      },
//      [&] {
//        PrintBodyHere();
//      },
//  });
//  printer.PrintCase({
//      [&] {
//        PrintAnotherConditionHere();
//      },
//      [&] {
//        PrintAnotherBodyHere();
//      },
//  });
//
// This helper adds the "else" keyword to all subsequent cases.
class IfStmtPrinter {
 public:
  explicit IfStmtPrinter(google::protobuf::io::Printer *printer)
      : is_first_(true), printer_(printer) {}

  struct IfStmtCase {
    std::function<void()> condition;
    std::function<void()> body;
  };

  void PrintCase(const IfStmtCase &kase) {
    if (is_first_) {
      printer_->Print("if (");
      is_first_ = false;
    } else {
      printer_->Print(" else if (");
    }
    kase.condition();
    printer_->Print(") {\n");
    {
      auto indent = printer_->WithIndent();
      kase.body();
    }
    printer_->Print("}");
  }

 private:
  bool is_first_;
  google::protobuf::io::Printer *printer_;
};

static void GetCheckedClasses(const Type &type, bool is_part_of_variant,
                              absl::flat_hash_set<std::string> *node_names) {
  switch (type.kind()) {
    case TypeKind::kBuiltin:
    case TypeKind::kEnum:
      break;
    case TypeKind::kClass: {
      if (is_part_of_variant) {
        const auto &class_type = static_cast<const ClassType &>(type);
        node_names->insert(class_type.name().ToPascalCase());
      }
      break;
    }
    case TypeKind::kVariant: {
      const auto &variant_type = static_cast<const VariantType &>(type);
      for (const auto &element_type : variant_type.types()) {
        GetCheckedClasses(*element_type, /*is_part_of_variant=*/true,
                          node_names);
      }
      break;
    }
    case TypeKind::kList: {
      const auto &list_type = static_cast<const ListType &>(type);
      GetCheckedClasses(list_type.element_type(), is_part_of_variant,
                        node_names);
      break;
    }
  }
}

static absl::flat_hash_set<std::string> GetCheckedClasses(const AstDef &ast) {
  absl::flat_hash_set<std::string> checked_classes;
  for (const NodeDef *node : ast.topological_sorted_nodes()) {
    for (const FieldDef &field : node->fields()) {
      GetCheckedClasses(field.type(), /*is_part_of_variant=*/false,
                        &checked_classes);
    }
  }
  return checked_classes;
}

void AstFromJsonPrinter::PrintAst(const AstDef &ast,
                                  absl::string_view cc_namespace,
                                  absl::string_view ast_path) {
  auto vars = WithVars({
      {"json_variable", kJsonValueVariableName},
  });

  auto header_path = GetAstHeaderPath(ast_path);

  PrintLicense();
  Println();

  PrintCodeGenerationWarning();
  Println();

  Println("// NOLINTBEGIN(whitespace/line_length)");
  Println("// clang-format off");
  Println("// IWYU pragma: begin_keep");
  Println();

  Println("#include <cstdint>");
  Println("#include <memory>");
  Println("#include <optional>");
  Println("#include <string>");
  Println("#include <utility>");
  Println("#include <variant>");
  Println("#include <vector>");
  Println();

  PrintIncludeHeaders({
      std::string(header_path),
      "absl/container/flat_hash_set.h",
      "absl/memory/memory.h",
      "absl/status/status.h",
      "absl/status/statusor.h",
      "absl/strings/str_cat.h",
      "absl/strings/string_view.h",
      "maldoca/base/status_macros.h",
      "nlohmann/json.hpp",
  });
  Println();

  PrintEnterNamespace(cc_namespace);
  Println();

  bool needs_get_type_function = absl::c_any_of(
      ast.topological_sorted_nodes(),
      [](const NodeDef *node) { return !node->children().empty(); });
  if (needs_get_type_function) {
    static const auto *kGetType = new std::string(UnIndentedSource(R"(
static absl::StatusOr<std::string> GetType(const nlohmann::json& $json_variable$) {
  auto type_it = $json_variable$.find("type");
  if (type_it == $json_variable$.end()) {
    return absl::InvalidArgumentError("`type` is undefined.");
  }
  const nlohmann::json& json_type = type_it.value();
  if (json_type.is_null()) {
    return absl::InvalidArgumentError("json_type is null.");
  }
  if (!json_type.is_string()) {
    return absl::InvalidArgumentError("`json_type` expected to be string.");
  }
  return json_type.get<std::string>();
}
    )"));

    Println(kGetType->c_str());
    Println();
  }

  absl::flat_hash_set<std::string> checked_classes = GetCheckedClasses(ast);

  for (const NodeDef *node : ast.topological_sorted_nodes()) {
    PrintTitle((Symbol(ast.lang_name()) + node->name()).ToPascalCase());
    Println();

    if (checked_classes.contains(node->name())) {
      PrintTypeChecker(*node);
      Println();
    }

    for (const FieldDef &field : node->fields()) {
      PrintGetFieldFunction(node->name(), field, ast.lang_name());
      Println();
    }

    PrintFromJsonFunction(*node, ast.lang_name());
    Println();
  }

  Println("// clang-format on");
  Println("// NOLINTEND(whitespace/line_length)");
  Println("// IWYU pragma: end_keep");
  Println();

  PrintExitNamespace(cc_namespace);
}

void AstFromJsonPrinter::PrintTypeChecker(const NodeDef &node) {
  auto vars = WithVars({
      {"NodeType", std::string(node.name())},
      {"json_variable", kJsonValueVariableName},
  });

  Println("static bool Is$NodeType$(const nlohmann::json& $json_variable$) {");
  absl::Cleanup end_body = [&] { Println("}"); };
  {
    auto indent = WithIndent();

    Println("if (!$json_variable$.is_object()) {");
    Println("  return false;");
    Println("}");

    if (node.children().empty() && node.parents().empty()) {
      // This is not a virtual class.
      Println("return true;");
      return;
    }

    const std::string code = UnIndentedSource(R"cc(
      auto type_it = $json_variable$.find("type");
      if (type_it == $json_variable$.end()) {
        return false;
      }
      const nlohmann::json &type_json = type_it.value();
      if (!type_json.is_string()) {
        return false;
      }
      const std::string &type = type_json.get<std::string>();
    )cc");
    Println(code);

    if (!node.leafs().empty()) {
      Println(
          "static const auto *kTypes = new absl::flat_hash_set<std::string>{");
      {
        auto indent = WithIndent(4);
        for (const NodeDef *leaf : node.leafs()) {
          auto vars = WithVars({
              {"LeafType", leaf->name()},
          });
          Println("\"$LeafType$\",");
        }
      }
      Println("};");
      Println();

      Println("return kTypes->contains(type);");

    } else {
      CHECK_EQ(node.name(), node.type().value());
      Println("return type == \"$NodeType$\";");
    }
  }
}

void AstFromJsonPrinter::PrintBuiltinJsonTypeCheck(const BuiltinType &type,
                                                   const Symbol &rhs) {
  auto vars = WithVars({
      {"rhs", rhs.ToCcVarName()},
  });

  switch (type.builtin_kind()) {
    case BuiltinTypeKind::kBool:
      Print("$rhs$.is_boolean()");
      break;
    case BuiltinTypeKind::kInt64:
      Print("$rhs$.is_number_integer()");
      break;
    case BuiltinTypeKind::kDouble:
      Print("$rhs$.is_number()");
      break;
    case BuiltinTypeKind::kString:
      Print("$rhs$.is_string()");
      break;
  }
}

void AstFromJsonPrinter::PrintClassJsonTypeCheck(const ClassType &class_type,
                                                 const Symbol &rhs) {
  auto vars = WithVars({
      {"ClassType", class_type.name().ToPascalCase()},
      {"rhs", rhs.ToCcVarName()},
  });

  Print("Is$ClassType$($rhs$)");
}

void AstFromJsonPrinter::PrintBuiltinFromJson(Action action,
                                              CheckJsonType check_json_type,
                                              const BuiltinType &builtin_type,
                                              const Symbol &lhs,
                                              const Symbol &rhs) {
  auto vars = WithVars({
      {"lhs", lhs.ToCcVarName()},
      {"rhs", rhs.ToCcVarName()},
      {"cc_type", builtin_type.CcType()},
      {"js_type", builtin_type.JsType()},
  });

  switch (check_json_type) {
    case CheckJsonType::kYes: {
      IfStmtPrinter if_stmt(this);
      if_stmt.PrintCase({
          [&] {
            // Print if-condition.
            Print("!");
            PrintBuiltinJsonTypeCheck(builtin_type, rhs);
          },

          [&] {
            // Print if-body.
            Print("return absl::InvalidArgumentError(\"Expecting ");
            PrintBuiltinJsonTypeCheck(builtin_type, rhs);
            Println(".\");");
          },
      });

      Println();
      break;
    }
    case CheckJsonType::kNo:
      break;
  }

  switch (action) {
    case Action::kAssign:
      Println("$lhs$ = $rhs$.get<$cc_type$>();");
      break;
    case Action::kDef:
      Println("auto $lhs$ = $rhs$.get<$cc_type$>();");
      break;
    case Action::kReturn:
      Println("return $rhs$.get<$cc_type$>();");
      break;
  }
}

void AstFromJsonPrinter::PrintEnumFromJson(Action action,
                                           const EnumType &enum_type,
                                           const Symbol &lhs, const Symbol &rhs,
                                           absl::string_view lang_name) {
  auto vars = WithVars({
      {"lhs", lhs.ToCcVarName()},
      {"rhs", rhs.ToCcVarName()},
      {"rhs_str", (rhs + "str").ToCcVarName()},
      {"EnumName", (Symbol(lang_name) + enum_type.name()).ToPascalCase()},
  });

  const auto check = UnIndentedSource(R"(
    if (!$rhs$.is_string()) {
      return absl::InvalidArgumentError("`$rhs$` expected to be a string.");
    }
    std::string $rhs_str$ = $rhs$.get<std::string>();
  )");
  Println(check);

  switch (action) {
    case Action::kAssign:
      Println(
          "MALDOCA_ASSIGN_OR_RETURN($lhs$, StringTo$EnumName$($rhs_str$));");
      break;
    case Action::kDef:
      Println(
          "MALDOCA_ASSIGN_OR_RETURN"
          "(auto $lhs$, StringTo$EnumName$($rhs_str$));");
      break;
    case Action::kReturn:
      Println("return StringTo$EnumName$($rhs_str$);");
      break;
  }
}

void AstFromJsonPrinter::PrintClassFromJson(Action action,
                                            const ClassType &class_type,
                                            const Symbol &lhs,
                                            const Symbol &rhs,
                                            absl::string_view lang_name) {
  auto vars = WithVars({
      {"lhs", lhs.ToCcVarName()},
      {"rhs", rhs.ToCcVarName()},
      {"Class", (Symbol(lang_name) + class_type.name()).ToPascalCase()},
  });

  switch (action) {
    case Action::kAssign:
      Println("MALDOCA_ASSIGN_OR_RETURN($lhs$, $Class$::FromJson($rhs$));");
      break;
    case Action::kDef:
      Println(
          "MALDOCA_ASSIGN_OR_RETURN(auto $lhs$, $Class$::FromJson($rhs$));");
      break;
    case Action::kReturn:
      Println("return $Class$::FromJson($rhs$);");
      break;
  }
}

void AstFromJsonPrinter::PrintVariantFromJson(Action action,
                                              const VariantType &variant_type,
                                              const Symbol &lhs,
                                              const Symbol &rhs,
                                              absl::string_view lang_name) {
  auto vars = WithVars({
      {"cc_type", variant_type.CcType()},
      {"lhs", lhs.ToCcVarName()},
      {"rhs", rhs.ToCcVarName()},
      {"json_variable", kJsonValueVariableName},
  });

  switch (action) {
    case Action::kDef:
      Println("$cc_type$ $lhs$;");
      break;

    case Action::kAssign:
    case Action::kReturn:
      break;
  }

  IfStmtPrinter if_stmt_printer(this);

  Action case_action = [&] {
    switch (action) {
      case Action::kAssign:
      case Action::kDef:
        return Action::kAssign;
      case Action::kReturn:
        return Action::kReturn;
    }
  }();

  for (const auto &scalar_type : variant_type.types()) {
    if (scalar_type->IsA<BuiltinType>()) {
      const auto &builtin_type = static_cast<const BuiltinType &>(*scalar_type);

      if_stmt_printer.PrintCase({
          [&] {
            // Print if-condition.
            PrintBuiltinJsonTypeCheck(builtin_type, rhs);
          },
          [&] {
            // Print if-body.
            PrintBuiltinFromJson(case_action, CheckJsonType::kNo, builtin_type,
                                 lhs, rhs);
          },
      });

    } else if (scalar_type->IsA<ClassType>()) {
      const auto &class_type = static_cast<const ClassType &>(*scalar_type);

      if_stmt_printer.PrintCase({
          [&] {
            // Print if-condition.
            PrintClassJsonTypeCheck(class_type, rhs);
          },
          [&] {
            // Print if-body.
            PrintClassFromJson(case_action, class_type, lhs, rhs, lang_name);
          },
      });

    } else {
      LOG(FATAL) << "Unreachable code.";
    }
  }

  const auto handle_invalid_type = UnIndentedSource(R"(
     else {
      auto result = absl::InvalidArgumentError("$rhs$ has invalid type.");
      result.SetPayload("json", absl::Cord{$json_variable$.dump()});
      result.SetPayload("json_element", absl::Cord{$rhs$.dump()});
      return result;
    }
  )");
  Println(handle_invalid_type);
}

void AstFromJsonPrinter::PrintListFromJson(Action action,
                                           const ListType &list_type,
                                           const Symbol &lhs, const Symbol &rhs,
                                           absl::string_view lang_name) {
  const Symbol lhs_element = lhs + "element";
  const Symbol rhs_element = rhs + "element";

  // Even if we are asked to assign to `lhs`, since the type of `lhs` is not
  // exactly list_type.CcType(), we need to define a new variable and assign it
  // to `lhs` at the end.
  const Symbol lhs_defined = [&] {
    switch (action) {
      case Action::kDef:
        return lhs;
      case Action::kAssign:
        return lhs + "value";
      case Action::kReturn:
        return lhs;
    }
  }();

  auto vars = WithVars({
      {"cc_type", list_type.CcType()},
      {"lhs", lhs.ToCcVarName()},
      {"rhs", rhs.ToCcVarName()},
      {"lhs_defined", lhs_defined.ToCcVarName()},
      {"lhs_element", lhs_element.ToCcVarName()},
      {"rhs_element", rhs_element.ToCcVarName()},
  });

  const auto check_json_type = UnIndentedSource(R"(
    if (!$rhs$.is_array()) {
      return absl::InvalidArgumentError("$rhs$ expected to be array.");
    }
  )");
  Println(check_json_type);
  Println();

  Println("$cc_type$ $lhs_defined$;");
  Println("for (const nlohmann::json& $rhs_element$ : $rhs$) {");
  {
    auto indent = WithIndent();
    PrintNullableFromJson(Action::kDef, list_type.element_type(),
                          list_type.element_maybe_null(), lhs_element,
                          rhs_element, lang_name);
    Println("$lhs_defined$.push_back(std::move($lhs_element$));");
  }
  Println("}");

  switch (action) {
    case Action::kDef:
      // Nothing here.
      break;
    case Action::kAssign:
      Println("$lhs$ = std::move($lhs_defined$);");
      break;
    case Action::kReturn:
      Println("return $lhs_defined$;");
      break;
  }
}

void AstFromJsonPrinter::PrintFromJson(Action action, const Type &type,
                                       const Symbol &lhs, const Symbol &rhs,
                                       absl::string_view lang_name) {
  switch (type.kind()) {
    case TypeKind::kList: {
      const auto &list_type = static_cast<const ListType &>(type);
      PrintListFromJson(action, list_type, lhs, rhs, lang_name);
      break;
    }

    case TypeKind::kVariant: {
      const auto &variant_type = static_cast<const VariantType &>(type);
      PrintVariantFromJson(action, variant_type, lhs, rhs, lang_name);
      break;
    }

    case TypeKind::kClass: {
      const auto &class_type = static_cast<const ClassType &>(type);
      PrintClassFromJson(action, class_type, lhs, rhs, lang_name);
      break;
    }

    case TypeKind::kEnum: {
      const auto &enum_type = static_cast<const EnumType &>(type);
      PrintEnumFromJson(action, enum_type, lhs, rhs, lang_name);
      break;
    }

    case TypeKind::kBuiltin: {
      const auto &builtin_type = static_cast<const BuiltinType &>(type);
      PrintBuiltinFromJson(action, CheckJsonType::kYes, builtin_type, lhs, rhs);
      break;
    }
  }
}

void AstFromJsonPrinter::PrintNullableFromJson(Action action, const Type &type,
                                               MaybeNull maybe_null,
                                               const Symbol &lhs,
                                               const Symbol &rhs,
                                               absl::string_view lang_name) {
  auto vars = WithVars({
      {"cc_type", type.CcType(maybe_null)},
      {"lhs", lhs.ToCcVarName()},
      {"rhs", rhs.ToCcVarName()},
  });

  switch (maybe_null) {
    case MaybeNull::kYes: {
      switch (action) {
        case Action::kDef:
          Print("$cc_type$ $lhs$;\n");
          ABSL_FALLTHROUGH_INTENDED;

        case Action::kAssign:
          Print("if (!$rhs$.is_null()) {\n");
          {
            auto indent = WithIndent();
            PrintFromJson(Action::kAssign, type, lhs, rhs, lang_name);
          }
          Print("}\n");
          break;

        case Action::kReturn: {
          const auto nullopt_on_null = UnIndentedSource(R"(
            if ($rhs$.is_null()) {
              return std::nullopt;
            }
          )");
          Println(nullopt_on_null);

          PrintFromJson(Action::kReturn, type, lhs, rhs, lang_name);
          break;
        }
      }

      break;
    }

    case MaybeNull::kNo: {
      Println("if ($rhs$.is_null()) {");
      Println("  return absl::InvalidArgumentError(\"$rhs$ is null.\");");
      Println("}");

      PrintFromJson(action, type, lhs, rhs, lang_name);
      break;
    }
  }
}

void AstFromJsonPrinter::PrintGetFieldFunction(const std::string &node_name,
                                               const FieldDef &field,
                                               absl::string_view lang_name) {
  const Symbol json_field_name = Symbol("json") + field.name();

  auto vars = WithVars({
      {"NodeType", (Symbol(lang_name) + node_name).ToPascalCase()},
      {"cc_type", CcType(field)},
      {"fieldName", field.name().ToCamelCase()},
      {"FieldName", field.name().ToPascalCase()},
      {"field_name", field.name().ToCcVarName()},
      {"field_name_it", (field.name() + "it").ToCcVarName()},
      {"json_variable", kJsonValueVariableName},
      {"json_field_name", json_field_name.ToCcVarName()},
  });

  Println("absl::StatusOr<$cc_type$>");
  Println(
      "$NodeType$::Get$FieldName$(const nlohmann::json& $json_variable$) {");
  {
    auto indent = WithIndent();
    const auto status_if_undefined = UnIndentedSource(R"cc(
      auto $field_name_it$ = $json_variable$.find("$fieldName$");
      if ($field_name_it$ == $json_variable$.end()) {
        return absl::InvalidArgumentError("`$fieldName$` is undefined.");
      }
      const nlohmann::json& $json_field_name$ = $field_name_it$.value();
    )cc");

    const auto nullopt_if_undefined = UnIndentedSource(R"cc(
      auto $field_name_it$ = $json_variable$.find("$fieldName$");
      if ($field_name_it$ == $json_variable$.end()) {
        return std::nullopt;
      }
      const nlohmann::json& $json_field_name$ = $field_name_it$.value();
    )cc");

    switch (field.optionalness()) {
      case OPTIONALNESS_UNSPECIFIED: {
        LOG(FATAL) << "Invalid Optionalness. Should be a bug.";
      }
      case OPTIONALNESS_REQUIRED: {
        Println(status_if_undefined);
        Println();

        PrintNullableFromJson(Action::kReturn, field.type(), MaybeNull::kNo,
                              /*lhs=*/field.name(), /*rhs=*/json_field_name,
                              lang_name);
        break;
      }
      case OPTIONALNESS_MAYBE_NULL: {
        Println(status_if_undefined);
        Println();

        PrintNullableFromJson(Action::kReturn, field.type(), MaybeNull::kYes,
                              /*lhs=*/field.name(), /*rhs=*/json_field_name,
                              lang_name);
        break;
      }
      case OPTIONALNESS_MAYBE_UNDEFINED: {
        Println(nullopt_if_undefined);
        Println();

        PrintNullableFromJson(Action::kReturn, field.type(), MaybeNull::kNo,
                              /*lhs=*/field.name(), /*rhs=*/json_field_name,
                              lang_name);
        break;
      }
    }
  }
  Println("}");
}

void AstFromJsonPrinter::PrintFromJsonFunction(const NodeDef &node,
                                               absl::string_view lang_name) {
  auto vars = WithVars({
      {"NodeType", (Symbol(lang_name) + node.name()).ToPascalCase()},
      {"json_variable", kJsonValueVariableName},
  });

  Println("absl::StatusOr<std::unique_ptr<$NodeType$>>");
  Println("$NodeType$::FromJson(const nlohmann::json& $json_variable$) {");
  {
    auto indent = WithIndent();

    const auto check_is_object = UnIndentedSource(R"cc(
      if (!$json_variable$.is_object()) {
        return absl::InvalidArgumentError("JSON is not an object.");
      }
    )cc");
    Println(check_is_object);
    Println();

    if (!node.children().empty()) {
      // This is a non-leaf type.
      // We get the `type` field and dispatch the corresponding FromJson()
      // function.

      Println(
          "MALDOCA_ASSIGN_OR_RETURN"
          "(std::string type, GetType($json_variable$));");
      Println();

      IfStmtPrinter if_stmt_printer(this);
      for (const NodeDef *descendent : node.descendants()) {
        auto vars = WithVars({
            {"DescendentType",
             (Symbol(lang_name) + descendent->name()).ToPascalCase()},
            {"DescendentTypeNoLangName", descendent->name()},
        });
        if_stmt_printer.PrintCase({
            [&] { Print("type == \"$DescendentTypeNoLangName$\""); },
            [&] {
              Println("return $DescendentType$::FromJson($json_variable$);");
            },
        });
      }
      Println();

      Print("return absl::InvalidArgumentError");
      Println(R"((absl::StrCat("Invalid type: ", type));)");

    } else {
      // This is a leaf type.
      // We get all the fields and call the constructor.

      struct NodeFieldPair {
        std::string node_name;
        Symbol field_name;
      };
      std::vector<NodeFieldPair> node_field_pairs;
      for (const NodeDef *ancestor : node.ancestors()) {
        for (const FieldDef &field : ancestor->fields()) {
          node_field_pairs.push_back({ancestor->name(), field.name()});
        }
      }
      for (const FieldDef &field : node.fields()) {
        node_field_pairs.push_back({node.name(), field.name()});
      }

      for (const NodeFieldPair &node_field_pair : node_field_pairs) {
        auto vars = WithVars({
            {"NodeType",
             (Symbol(lang_name) + node_field_pair.node_name).ToPascalCase()},
            {"field_name", node_field_pair.field_name.ToCcVarName()},
            {"FieldName", node_field_pair.field_name.ToPascalCase()},
        });
        Println(
            "MALDOCA_ASSIGN_OR_RETURN(auto $field_name$, "
            "$NodeType$::Get$FieldName$($json_variable$));");
      }

      Println();

      // Call the constructor.
      Print("return absl::make_unique<$NodeType$>(\n");
      {
        auto indent = WithIndent(4);
        TabPrinter tab_printer{{
            .print_separator = [this] { Print(",\n"); },
        }};
        for (const FieldDef *field : node.aggregated_fields()) {
          auto vars = WithVars({
              {"field_name", field->name().ToCcVarName()},
          });

          tab_printer.Print();
          Print("std::move($field_name$)");
        }
      }

      Println(");");
    }
  }
  Println("}");
}

std::string PrintAstFromJson(const AstDef &ast, absl::string_view cc_namespace,
                             absl::string_view ast_path) {
  std::string str;
  {
    google::protobuf::io::StringOutputStream os(&str);
    AstFromJsonPrinter printer(&os);
    printer.PrintAst(ast, cc_namespace, ast_path);
  }

  return str;
}

// =============================================================================
// IrTableGenPrinter
// =============================================================================

void IrTableGenPrinter::PrintAst(const AstDef &ast, absl::string_view ir_path) {
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
  for (const auto &import : imports) {
    Println(absl::StrCat("include \"", import, "\""));
  }
  Println();

  bool has_expr_region = false;
  bool has_exprs_region = false;
  for (const auto *node : ast.topological_sorted_nodes()) {
    for (const auto *field : node->aggregated_fields()) {
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

  for (const auto *node : ast.topological_sorted_nodes()) {
    if (!node->should_generate_ir_op()) {
      continue;
    }

    for (auto kind : node->aggregated_kinds()) {
      PrintNode(ast, *node, kind);
    }
  }

  PrintExitHeaderGuard(td_path);
}

void IrTableGenPrinter::PrintNode(const AstDef &ast, const NodeDef &node,
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
  for (const NodeDef *parent : node.parents()) {
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
  for (const FieldDef &field : node.fields()) {
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

      for (const Symbol &trait : traits) {
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
        for (const auto *field : node.aggregated_fields()) {
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
        for (const auto *field : node.aggregated_fields()) {
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

void IrTableGenPrinter::PrintArgument(const AstDef &ast, const NodeDef &node,
                                      const FieldDef &field) {
  auto vars = WithVars({
      {"type", field.type().TdType(field.optionalness(), field.kind())},
      {"name", field.name().ToCcVarName()},
  });
  Print("$type$: $$$name$");
}

void IrTableGenPrinter::PrintRegion(const AstDef &ast, const NodeDef &node,
                                    const FieldDef &field) {
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

std::string PrintIrTableGen(const AstDef &ast, absl::string_view ir_path) {
  std::string str;
  {
    google::protobuf::io::StringOutputStream os(&str);
    IrTableGenPrinter printer(&os);
    printer.PrintAst(ast, ir_path);
  }

  return str;
}

// =============================================================================
// AstToIrSourcePrinter
// =============================================================================

void AstToIrSourcePrinter::PrintAst(const AstDef &ast,
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
      absl::StrCat(ir_path, "/conversion/ast_to_", ast.lang_name(), "ir.h"));
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

  for (const auto *node : ast.topological_sorted_nodes()) {
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

static Symbol GetVisitor(const NodeDef &node, FieldKind kind) {
  auto visitor = Symbol("Visit") + node.name();
  if (kind == FIELD_KIND_ATTR) {
    visitor += "Attr";
  }
  if (kind == FIELD_KIND_LVAL) {
    visitor += "Ref";
  }
  return visitor;
}

void AstToIrSourcePrinter::PrintNonLeafNode(const AstDef &ast,
                                            const NodeDef &node,
                                            FieldKind kind) {
  auto ir_op_name = node.ir_op_name(ast.lang_name(), kind);
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
  auto ir_name = Symbol(absl::StrCat(ast.lang_name(), "ir"));
  auto visitor = GetVisitor(node, kind);

  auto vars = WithVars({
      {"Ret", return_type},
      {"Name", (Symbol(ast.lang_name()) + node.name()).ToPascalCase()},
      {"IrName", ir_name.ToPascalCase()},
      {"Visitor", visitor.ToPascalCase()},
  });

  Println("$Ret$ AstTo$IrName$::$Visitor$(const $Name$ *node) {");
  {
    auto indent = WithIndent();
    for (const NodeDef *leaf : node.leafs()) {
      auto vars = WithVars({
          {"LeafName", (Symbol(ast.lang_name()) + leaf->name()).ToPascalCase()},
          {"leaf_name", Symbol(leaf->name()).ToCcVarName()},
          {"LeafVisitor", GetVisitor(*leaf, kind).ToPascalCase()},
      });
      Println(
          "if (auto *$leaf_name$ = dynamic_cast<const $LeafName$ *>(node)) {");
      Println("  return $LeafVisitor$($leaf_name$);");
      Println("}");
    }

    Println("LOG(FATAL) << \"Unreachable code.\";");
  }
  Println("}");
  Println();
}

void AstToIrSourcePrinter::PrintLeafNode(const AstDef &ast, const NodeDef &node,
                                         FieldKind kind) {
  auto ir_op_name = node.ir_op_name(ast.lang_name(), kind).value();
  auto ir_name = Symbol(absl::StrCat(ast.lang_name(), "ir"));

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

  auto vars = WithVars({
      {"OpName", ir_op_name.ToPascalCase()},
      {"Name", (Symbol(ast.lang_name()) + node.name()).ToPascalCase()},
      {"IrName", ir_name.ToPascalCase()},
      {"Visitor", visitor.ToPascalCase()},
      {"Creator", creator.ToPascalCase()},
  });

  Println("$OpName$ AstTo$IrName$::$Visitor$(const $Name$ *node) {");
  {
    auto indent = WithIndent();

    for (const auto *field : node.aggregated_fields()) {
      if (!FieldIsArgument(field)) {
        continue;
      }

      PrintField(ast, node, *field);
    }

    bool has_regions = absl::c_any_of(node.aggregated_fields(), FieldIsRegion);
    if (has_regions) {
      Print("auto op = ");
    } else {
      Print("return ");
    }

    Print("$Creator$<$OpName$>(node");
    {
      auto indent = WithIndent(4);
      for (const auto *field : node.aggregated_fields()) {
        if (!FieldIsArgument(field)) {
          continue;
        }

        const auto mlir_field_name = (Symbol("mlir") + field->name());
        auto vars = WithVars({
            {"mlir_field_name", mlir_field_name.ToCcVarName()},
        });

        Print(", $mlir_field_name$");
      }
    }
    Println(");");

    if (has_regions) {
      for (const auto *field : node.aggregated_fields()) {
        if (FieldIsRegion(field)) {
          PrintRegion(ast, node, *field);
        }
      }

      Println("return op;");
    }
  }

  Println("}");
  Println();
}

void AstToIrSourcePrinter::PrintField(const AstDef &ast, const NodeDef &node,
                                      const FieldDef &field) {
  MaybeNull maybe_null = OptionalnessToMaybeNull(field.optionalness());

  auto lhs = Symbol("mlir") + field.name();
  auto rhs = absl::StrCat("node->", field.name().ToCcVarName(), "()");
  PrintNullableToIr(ast, Action::kDef, field.type(), maybe_null, RefOrVal::kRef,
                    field.kind(), lhs, rhs);
}

void AstToIrSourcePrinter::PrintRegion(const AstDef &ast, const NodeDef &node,
                                       const FieldDef &field) {
  MaybeNull maybe_null = OptionalnessToMaybeNull(field.optionalness());

  auto lhs = Symbol("mlir") + field.name();
  auto lhs_region = lhs + "region";
  auto rhs = absl::StrCat("node->", field.name().ToCcVarName(), "()");
  auto ir_name = Symbol(absl::StrCat(ast.lang_name(), "ir"));

  auto vars = WithVars({
      {"lhs", lhs.ToCcVarName()},
      {"lhs_region", lhs_region.ToCcVarName()},
      {"mlirGetter", field.name().ToMlirGetter()},
      {"rhs", rhs},
  });

  auto populate_region = [&] {
    Println("mlir::Region &$lhs_region$ = op.$mlirGetter$();");
    Println("AppendNewBlockAndPopulate($lhs_region$, [&] {");
    {
      auto indent = WithIndent();

      Action action = [&] {
        switch (field.kind()) {
          case FIELD_KIND_UNSPECIFIED:
            LOG(FATAL) << "Unspecified FieldKind.";
          case FIELD_KIND_ATTR:
            LOG(FATAL) << "Unsupported FieldKind: " << field.kind();
          case FIELD_KIND_RVAL:
          case FIELD_KIND_LVAL: {
            return Action::kDef;
          }
          case FIELD_KIND_STMT: {
            return Action::kCreate;
          }
        }
      }();

      Symbol region_end_op = GetRegionEndOp(ast, field);
      PrintToIr(ast, action, field.type(), RefOrVal::kRef, field.kind(), lhs,
                rhs);

      auto vars = WithVars({
          {"RegionEndOp", region_end_op.ToPascalCase()},
      });

      switch (action) {
        case Action::kAssign:
          LOG(FATAL) << "Unsupported Action: Assign.";
        case Action::kCreate:
          break;
        case Action::kDef: {
          Println("CreateStmt<$RegionEndOp$>(node, $lhs$);");
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

void AstToIrSourcePrinter::PrintBuiltinToIr(const AstDef &ast, Action action,
                                            const BuiltinType &type,
                                            const Symbol &lhs,
                                            const std::string &rhs) {
  auto vars = WithVars({
      {"mlir_type", type.CcMlirBuilderType(FIELD_KIND_ATTR)},
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
      Print("builder_.getBoolAttr($rhs$)");
      break;
    }
    case BuiltinTypeKind::kInt64: {
      Print("builder_.getI64IntegerAttr($rhs$)");
      break;
    }
    case BuiltinTypeKind::kString: {
      Print("builder_.getStringAttr($rhs$)");
      break;
    }
    case BuiltinTypeKind::kDouble: {
      Print("builder_.getF64FloatAttr($rhs$)");
      break;
    }
  }

  Println(";");
}

void AstToIrSourcePrinter::PrintClassToIr(const AstDef &ast, Action action,
                                          const ClassType &type, FieldKind kind,
                                          const Symbol &lhs,
                                          const std::string &rhs) {
  auto vars = WithVars({
      {"ClassName", type.name().ToPascalCase()},
      {"lhs", lhs.ToCcVarName()},
      {"rhs", rhs},
  });

  switch (action) {
    case Action::kDef: {
      auto vars = WithVars({
          {"cc_mlir_type", type.CcMlirBuilderType(kind)},
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
      Println("Visit$ClassName$Attr($rhs$);");
      break;
    case FIELD_KIND_RVAL:
    case FIELD_KIND_STMT: {
      Println("Visit$ClassName$($rhs$);");
      break;
    }
    case FIELD_KIND_LVAL: {
      Println("Visit$ClassName$Ref($rhs$);");
      break;
    }
  }
}

void AstToIrSourcePrinter::PrintClassToIr(const AstDef &ast, Action action,
                                          const ClassType &type,
                                          RefOrVal ref_or_val, FieldKind kind,
                                          const Symbol &lhs,
                                          const std::string &rhs) {
  switch (ref_or_val) {
    case RefOrVal::kRef:
      return PrintClassToIr(ast, action, type, kind, lhs, rhs);
    case RefOrVal::kVal:
      return PrintClassToIr(ast, action, type, kind, lhs,
                            absl::StrCat(rhs, ".get()"));
  }
}

void AstToIrSourcePrinter::PrintEnumToIr(const AstDef &ast, Action action,
                                         const EnumType &type,
                                         const Symbol &lhs,
                                         const std::string &rhs) {
  auto enum_name = (Symbol(ast.lang_name()) + type.name()).ToPascalCase();
  auto rhs_str = absl::StrCat(enum_name, "ToString(", rhs, ")");

  BuiltinType string_type{BuiltinTypeKind::kString, ast.lang_name()};
  return PrintBuiltinToIr(ast, action, string_type, lhs, rhs_str);
}

void AstToIrSourcePrinter::PrintVariantToIr(const AstDef &ast, Action action,
                                            const VariantType &type,
                                            RefOrVal ref_or_val, FieldKind kind,
                                            const Symbol &lhs,
                                            const std::string &rhs) {
  auto vars = WithVars({
      {"lhs", lhs.ToCcVarName()},
      {"rhs", rhs},
  });

  Action case_action;
  switch (action) {
    case Action::kDef: {
      auto vars = WithVars({
          {"cc_mlir_type", type.CcMlirBuilderType(kind)},
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
        const ScalarType &scalar_type = *type.types()[i];
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

void AstToIrSourcePrinter::PrintListToIr(const AstDef &ast, Action action,
                                         const ListType &type, FieldKind kind,
                                         const Symbol &lhs,
                                         const std::string &rhs) {
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
          Println("auto $lhs$ = builder_.getArrayAttr($lhs_data$);");
          break;
        }
        case Action::kAssign: {
          Println("$lhs$ = builder_.getArrayAttr($lhs_data$);");
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
              auto none_op =
                  Symbol(absl::StrCat(ast.lang_name(), "ir")) + "NoneOp";
              auto vars = WithVars({
                  {"NoneOp", none_op.ToPascalCase()},
              });

              Println("$lhs_element$ = CreateExpr<$NoneOp$>(node);");
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

void AstToIrSourcePrinter::PrintToIr(const AstDef &ast, Action action,
                                     const Type &type, RefOrVal ref_or_val,
                                     FieldKind kind, const Symbol &lhs,
                                     const std::string &rhs) {
  switch (type.kind()) {
    case TypeKind::kBuiltin: {
      const auto &builtin_type = static_cast<const BuiltinType &>(type);
      return PrintBuiltinToIr(ast, action, builtin_type, lhs, rhs);
    }

    case TypeKind::kClass: {
      const auto &class_type = static_cast<const ClassType &>(type);
      return PrintClassToIr(ast, action, class_type, ref_or_val, kind, lhs,
                            rhs);
    }

    case TypeKind::kEnum: {
      const auto &enum_type = static_cast<const EnumType &>(type);
      return PrintEnumToIr(ast, action, enum_type, lhs, rhs);
    }

    case TypeKind::kVariant: {
      const auto &variant_type = static_cast<const VariantType &>(type);
      return PrintVariantToIr(ast, action, variant_type, ref_or_val, kind, lhs,
                              rhs);
    }

    case TypeKind::kList: {
      const auto &list_type = static_cast<const ListType &>(type);
      CHECK(ref_or_val == RefOrVal::kRef);
      return PrintListToIr(ast, action, list_type, kind, lhs, rhs);
    }
  }
}

void AstToIrSourcePrinter::PrintNullableToIr(const AstDef &ast, Action action,
                                             const Type &type,
                                             MaybeNull maybe_null,
                                             RefOrVal ref_or_val,
                                             FieldKind kind, const Symbol &lhs,
                                             const std::string &rhs) {
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
              {"mlir_type", type.CcMlirBuilderType(kind)},
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

std::string PrintAstToIrSource(const AstDef &ast,
                               absl::string_view cc_namespace,
                               absl::string_view ast_path,
                               absl::string_view ir_path) {
  std::string str;
  {
    google::protobuf::io::StringOutputStream os(&str);
    AstToIrSourcePrinter printer(&os);
    printer.PrintAst(ast, cc_namespace, ast_path, ir_path);
  }

  return str;
}

// =============================================================================
// IrToAstSourcePrinter
// =============================================================================

void IrToAstSourcePrinter::PrintAst(const AstDef &ast,
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
      "maldoca/base/status_macros.h",
      std::string(ast_header_path),
      absl::StrCat(ir_path, "/ir.h"),
  });
  Println();

  PrintEnterNamespace(cc_namespace);
  Println();

  for (const auto *node : ast.topological_sorted_nodes()) {
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

void IrToAstSourcePrinter::PrintNonLeafNode(const AstDef &ast,
                                            const NodeDef &node,
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
      for (const NodeDef *leaf : node.leafs()) {
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

void IrToAstSourcePrinter::PrintLeafNode(const AstDef &ast, const NodeDef &node,
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
    for (const auto *field : node.aggregated_fields()) {
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

      for (const FieldDef *field : node.aggregated_fields()) {
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

void IrToAstSourcePrinter::PrintField(const AstDef &ast, const NodeDef &node,
                                      const FieldDef &field) {
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

  PrintNullableFromIr(ast, Action::kDef, field.type(), maybe_null, field.kind(),
                      /*lhs=*/field.name(), rhs, RhsKind::kFieldGetterResult);
}

void IrToAstSourcePrinter::PrintRegion(const AstDef &ast, const NodeDef &node,
                                       const FieldDef &field) {
  MaybeNull maybe_null = OptionalnessToMaybeNull(field.optionalness());

  auto vars = WithVars({
      {"lhs", field.name().ToCcVarName()},
      {"cc_type", field.type().CcType(maybe_null)},
      {"mlirGetter", field.name().ToMlirGetter()},
  });

  auto print_from_region = [&](Action action) {
    std::string rhs;
    std::string rhs_getter;
    RhsKind rhs_kind;
    switch (field.kind()) {
      case FIELD_KIND_UNSPECIFIED:
        LOG(FATAL) << "Unspecified FieldKind.";
      case FIELD_KIND_ATTR:
        LOG(FATAL) << "Unsupported FieldKind: " << field.kind();
      case FIELD_KIND_RVAL:
      case FIELD_KIND_LVAL: {
        if (field.type().IsA<ListType>()) {
          rhs = (Symbol("mlir") + field.name() + "values").ToCcVarName();
          rhs_getter = "GetExprsRegionValues";
          rhs_kind = RhsKind::kListElement;
        } else {
          rhs = (Symbol("mlir") + field.name() + "value").ToCcVarName();
          rhs_getter = "GetExprRegionValue";
          rhs_kind = RhsKind::kListElement;
        }
        break;
      }
      case FIELD_KIND_STMT: {
        if (field.type().IsA<ListType>()) {
          rhs = (Symbol("mlir") + field.name() + "block").ToCcVarName();
          rhs_getter = "GetStmtsRegionBlock";
          rhs_kind = RhsKind::kListElement;
        } else {
          rhs = (Symbol("mlir") + field.name() + "operation").ToCcVarName();
          rhs_getter = "GetStmtRegionOperation";
          rhs_kind = RhsKind::kFieldGetterResult;
        }
        break;
      }
    }

    auto vars = WithVars({
        {"rhs", rhs},
        {"RhsGetter", rhs_getter},
    });

    Println(
        "MALDOCA_ASSIGN_OR_RETURN"
        "(auto $rhs$, $RhsGetter$(op.$mlirGetter$()));");

    PrintFromIr(ast, action, field.type(), field.kind(), /*lhs=*/field.name(),
                rhs, rhs_kind);
  };

  switch (maybe_null) {
    case MaybeNull::kYes: {
      Println("$cc_type$ $lhs$;");
      Println("if (!op.$mlirGetter$().empty()) {");
      {
        auto indent = WithIndent();
        print_from_region(Action::kAssign);
      }
      Println("}");
      break;
    }
    case MaybeNull::kNo: {
      print_from_region(Action::kDef);
    }
  }
}

void IrToAstSourcePrinter::PrintNullableFromIr(
    const AstDef &ast, Action action, const Type &type, MaybeNull maybe_null,
    FieldKind kind, const Symbol &lhs, const std::string &rhs,
    RhsKind rhs_kind) {
  auto none_op = Symbol(absl::StrCat(ast.lang_name(), "ir")) + "NoneOp";
  auto vars = WithVars({
      {"type", type.CcType(maybe_null)},
      {"lhs", lhs.ToCcVarName()},
      {"rhs", rhs},
      {"NoneOp", none_op.ToPascalCase()},
  });

  switch (maybe_null) {
    case MaybeNull::kYes: {
      switch (action) {
        case Action::kDef:
          Println("$type$ $lhs$;");
          ABSL_FALLTHROUGH_INTENDED;

        case Action::kAssign: {
          // When a field is an mlir::Attribute, it can be null; but when a
          // field is an mlir::Value, it cannot be null (MLIR verification would
          // fail). Therefore, we need to define a specific <SomeIr>NoneOp
          // operation to represent null fields.
          bool should_use_none_op = false;
          switch (rhs_kind) {
            case RhsKind::kOp:
            case RhsKind::kFieldGetterResult:
              break;
            case RhsKind::kListElement:
              switch (kind) {
                case FIELD_KIND_UNSPECIFIED:
                case FIELD_KIND_STMT:
                case FIELD_KIND_ATTR:
                  break;
                case FIELD_KIND_RVAL:
                case FIELD_KIND_LVAL:
                  should_use_none_op = true;
              }
              break;
          }

          if (should_use_none_op) {
            Println("if (!llvm::isa<$NoneOp$>($rhs$.getDefiningOp())) {");
          } else {
            Println("if ($rhs$ != nullptr) {");
          }

          {
            auto indent = WithIndent();
            PrintFromIr(ast, Action::kAssign, type, kind, lhs, rhs, rhs_kind);
          }
          Println("}");
          break;
        }
      }
      break;
    }
    case MaybeNull::kNo:
      PrintFromIr(ast, action, type, kind, lhs, rhs, rhs_kind);
      break;
  }
}

void IrToAstSourcePrinter::PrintFromIr(const AstDef &ast, Action action,
                                       const Type &type, FieldKind kind,
                                       const Symbol &lhs,
                                       const std::string &rhs,
                                       RhsKind rhs_kind) {
  switch (type.kind()) {
    case TypeKind::kBuiltin: {
      const auto &builtin_type = static_cast<const BuiltinType &>(type);
      PrintBuiltinFromIr(ast, action, builtin_type, lhs, rhs, rhs_kind);
      break;
    }
    case TypeKind::kClass: {
      const auto &class_type = static_cast<const ClassType &>(type);
      PrintClassFromIr(ast, action, class_type, kind, lhs, rhs, rhs_kind);
      break;
    }
    case TypeKind::kVariant: {
      const auto &variant_type = static_cast<const VariantType &>(type);
      PrintVariantFromIr(ast, action, variant_type, kind, lhs, rhs, rhs_kind);
      break;
    }
    case TypeKind::kList: {
      const auto &list_type = static_cast<const ListType &>(type);
      PrintListFromIr(ast, action, list_type, kind, lhs, rhs);
      break;
    }
    case TypeKind::kEnum:
      const auto &enum_type = static_cast<const EnumType &>(type);
      PrintEnumFromIr(ast, action, enum_type, lhs, rhs);
      break;
  }
}

void IrToAstSourcePrinter::PrintBuiltinFromIr(const AstDef &ast, Action action,
                                              const BuiltinType &type,
                                              const Symbol &lhs,
                                              const std::string &rhs,
                                              RhsKind rhs_kind) {
  auto vars = WithVars({
      {"type", type.CcType()},
      {"lhs", lhs.ToCcVarName()},
      {"rhs", rhs},
      {"AttrName", type.CcMlirGetterType(FIELD_KIND_ATTR)},
  });

  std::string converted_rhs;
  switch (rhs_kind) {
    case RhsKind::kOp:
    case RhsKind::kFieldGetterResult:
      converted_rhs = rhs;
      break;
    case RhsKind::kListElement: {
      auto cast = UnIndentedSource(R"(
        auto $lhs$_attr = llvm::dyn_cast<$AttrName$>($rhs$);
        if ($lhs$_attr == nullptr) {
          return absl::InvalidArgumentError("Invalid attribute.");
        }
      )");
      Println(cast);

      converted_rhs = absl::StrCat(lhs.ToCcVarName(), "_attr");
      break;
    }
  }
  auto converted_rhs_var = WithVars({
      {"rhs", converted_rhs},
  });

  switch (action) {
    case Action::kDef:
      Print("$type$ ");
      ABSL_FALLTHROUGH_INTENDED;
    case Action::kAssign:
      Print("$lhs$ = ");
      break;
  }

  switch (type.builtin_kind()) {
    case BuiltinTypeKind::kBool: {
      Print("$rhs$.getValue()");
      break;
    }
    case BuiltinTypeKind::kInt64: {
      Print("$rhs$.getValue().getInt()");
      break;
    }
    case BuiltinTypeKind::kString: {
      Print("$rhs$.str()");
      break;
    }
    case BuiltinTypeKind::kDouble: {
      Print("$rhs$.getValueAsDouble()");
      break;
    }
  }

  Println(";");
}

void IrToAstSourcePrinter::PrintClassFromIr(const AstDef &ast, Action action,
                                            const ClassType &type,
                                            FieldKind kind, const Symbol &lhs,
                                            const std::string &rhs,
                                            RhsKind rhs_kind) {
  auto node_it = ast.nodes().find(type.name().ToPascalCase());
  CHECK(node_it != ast.nodes().end());
  auto op_name = node_it->second->ir_op_name(ast.lang_name(), kind);

  auto vars = WithVars({
      {"ClassName", type.name().ToPascalCase()},
      {"OpName",
       op_name.has_value() ? op_name->ToPascalCase() : "mlir::Operation*"},
      {"lhs", lhs.ToCcVarName()},
  });

  // Sometimes `rhs` is a mlir::Value, so we first need to call getDefiningOp().
  std::string converted_rhs;
  switch (rhs_kind) {
    case RhsKind::kOp:
      converted_rhs = rhs;
      break;
    case RhsKind::kFieldGetterResult:
    case RhsKind::kListElement: {
      switch (kind) {
        case FIELD_KIND_UNSPECIFIED:
        case FIELD_KIND_ATTR:
        case FIELD_KIND_STMT:
          converted_rhs = rhs;
          break;
        case FIELD_KIND_RVAL:
        case FIELD_KIND_LVAL:
          converted_rhs = absl::StrCat(rhs, ".getDefiningOp()");
          break;
      }
      break;
    }
  }
  auto converted_rhs_var = WithVars({
      {"rhs", converted_rhs},
  });

  switch (kind) {
    case FIELD_KIND_UNSPECIFIED:
    case FIELD_KIND_ATTR: {
      switch (rhs_kind) {
        case RhsKind::kOp:
        case RhsKind::kFieldGetterResult:
          // $rhs$ is a <ClassName>Attr now.
          break;
        case RhsKind::kListElement: {
          auto cast = UnIndentedSource(R"(
            auto $lhs$_attr = llvm::dyn_cast<$OpName$>($rhs$);
            if ($lhs$_attr == nullptr) {
              return absl::InvalidArgumentError("Invalid attribute.");
            }
          )");
          Println(cast);

          converted_rhs = absl::StrCat(lhs.ToCcVarName(), "_attr");

          break;
        }
      }
      break;
    }
    case FIELD_KIND_RVAL:
    case FIELD_KIND_LVAL: {
      switch (rhs_kind) {
        case RhsKind::kOp:
          // $rhs$ is a <ClassName>Op now.
          break;
        case RhsKind::kFieldGetterResult:
        case RhsKind::kListElement: {
          auto cast = UnIndentedSource(R"cc(
            auto $lhs$_op = llvm::dyn_cast<$OpName$>($rhs$);
            if ($lhs$_op == nullptr) {
              return absl::InvalidArgumentError(
                  absl::StrCat("Expected $OpName$, got ",
                               $rhs$->getName().getStringRef().str(), "."));
            }
          )cc");
          Println(cast);

          converted_rhs = absl::StrCat(lhs.ToCcVarName(), "_op");

          break;
        }
      }
      break;
    }

    case FIELD_KIND_STMT: {
      switch (rhs_kind) {
        case RhsKind::kOp:
          break;
        case RhsKind::kFieldGetterResult: {
          // $rhs$ is an mlir::Operation* now.
          auto cast = UnIndentedSource(R"cc(
            auto $lhs$_op = llvm::dyn_cast<$OpName$>($rhs$);
            if ($lhs$_op == nullptr) {
              return absl::InvalidArgumentError(
                  absl::StrCat("Expected $OpName$, got ",
                               $rhs$->getName().getStringRef().str(), "."));
            }
          )cc");
          Println(cast);

          converted_rhs = absl::StrCat(lhs.ToCcVarName(), "_op");

          break;
        }
        case RhsKind::kListElement: {
          auto cast = UnIndentedSource(R"(
            auto $lhs$_op = llvm::dyn_cast<$OpName$>($rhs$);
            if ($lhs$_op == nullptr) {
              continue;
            }
          )");
          Println(cast);

          converted_rhs = absl::StrCat(lhs.ToCcVarName(), "_op");

          break;
        }
      }
      break;
    }
  }

  auto further_converted_rhs_var = WithVars({
      {"rhs", converted_rhs},
  });

  Print("MALDOCA_ASSIGN_OR_RETURN(");

  switch (action) {
    case Action::kDef: {
      auto vars = WithVars({
          {"type", type.CcType()},
      });
      Print("$type$ ");
      ABSL_FALLTHROUGH_INTENDED;
    }
    case Action::kAssign:
      Print("$lhs$, ");
      break;
  }

  switch (kind) {
    case FIELD_KIND_UNSPECIFIED:
    case FIELD_KIND_ATTR:
      Println("Visit$ClassName$Attr($rhs$));");
      break;
    case FIELD_KIND_RVAL:
    case FIELD_KIND_STMT: {
      Println("Visit$ClassName$($rhs$));");
      break;
    }
    case FIELD_KIND_LVAL: {
      Println("Visit$ClassName$Ref($rhs$));");
      break;
    }
  }
}

void IrToAstSourcePrinter::PrintEnumFromIr(const AstDef &ast, Action action,
                                           const EnumType &type,
                                           const Symbol &lhs,
                                           const std::string &rhs) {
  auto enum_name = Symbol(ast.lang_name()) + type.name();

  absl::btree_map<std::string, std::string> vars = {
      {"Type", enum_name.ToPascalCase()},
      {"lhs", lhs.ToCcVarName()},
      {"rhs", rhs},
  };

  switch (action) {
    case Action::kDef:
      Println(vars,
              "MALDOCA_ASSIGN_OR_RETURN"
              "($Type$ $lhs$, StringTo$Type$($rhs$.str()));");
      break;
    case Action::kAssign:
      Println(vars,
              "MALDOCA_ASSIGN_OR_RETURN(lhs, StringTo$Type$($rhs$.str()));");
      break;
  }
}

void IrToAstSourcePrinter::PrintVariantFromIr(const AstDef &ast, Action action,
                                              const VariantType &type,
                                              FieldKind kind, const Symbol &lhs,
                                              const std::string &rhs,
                                              RhsKind rhs_kind) {
  auto mlir_lhs = (Symbol("mlir") + lhs).ToCcVarName();
  auto vars = WithVars({
      {"type", type.CcType()},
      {"lhs", lhs.ToCcVarName()},
      {"mlir_lhs", mlir_lhs},
  });

  switch (action) {
    case Action::kDef:
      Println("$type$ $lhs$;");
      break;
    case Action::kAssign:
      break;
  }

  std::string converted_rhs;
  switch (kind) {
    case FIELD_KIND_UNSPECIFIED:
    case FIELD_KIND_ATTR:
    case FIELD_KIND_STMT:
      converted_rhs = rhs;
      break;
    case FIELD_KIND_RVAL:
    case FIELD_KIND_LVAL:
      converted_rhs = absl::StrCat(rhs, ".getDefiningOp()");
      break;
  }
  auto rhs_var = WithVars({
      {"rhs", converted_rhs},
  });

  IfStmtPrinter if_stmt(this);
  for (const auto &scalar_type : type.types()) {
    if_stmt.PrintCase({
        [&] {
          if (scalar_type->IsA<ClassType>()) {
            const auto &class_type =
                static_cast<const ClassType &>(*scalar_type);

            auto node_it = ast.nodes().find(class_type.name().ToPascalCase());
            CHECK(node_it != ast.nodes().end());
            auto op_name = node_it->second->ir_op_name(ast.lang_name(), kind);
            if (!op_name.has_value()) {
              // There is no base op or attribute. Fallback to mlir::Attribute
              // or mlir::Operation*.

              Print("auto $mlir_lhs$ = $rhs$");
            }
            auto vars = WithVars({
                {"Op", op_name.has_value() ? op_name->ToPascalCase()
                                           : "mlir::Operation*"},
            });

            Print("auto $mlir_lhs$ = llvm::dyn_cast<$Op$>($rhs$)");
          } else if (scalar_type->IsA<BuiltinType>()) {
            const auto &builtin_type =
                static_cast<const BuiltinType &>(*scalar_type);

            auto attr_name = builtin_type.CcMlirGetterType(kind);
            auto vars = WithVars({
                {"Attr", attr_name},
            });

            Print("auto $mlir_lhs$ = llvm::dyn_cast<$Attr$>($rhs$)");
          }
        },
        [&] {
          // Body
          PrintFromIr(ast, Action::kAssign, *scalar_type, kind, lhs, mlir_lhs,
                      RhsKind::kOp);
        },
    });
  }

  switch (kind) {
    case FIELD_KIND_STMT: {
      if (rhs_kind == RhsKind::kListElement) {
        const auto handle_invalid_type = UnIndentedSource(R"(
         else {
          continue;
        }
      )");
        Println(handle_invalid_type);
        break;
      }

      ABSL_FALLTHROUGH_INTENDED;
    }
    case FIELD_KIND_UNSPECIFIED:
    case FIELD_KIND_RVAL:
    case FIELD_KIND_LVAL:
    case FIELD_KIND_ATTR: {
      const auto handle_invalid_type = UnIndentedSource(R"(
     else {
      return absl::InvalidArgumentError("$rhs$ has invalid type.");
    }
  )");
      Println(handle_invalid_type);
      break;
    }
  }
}

void IrToAstSourcePrinter::PrintListFromIr(const AstDef &ast, Action action,
                                           const ListType &type, FieldKind kind,
                                           const Symbol &lhs,
                                           const std::string &rhs) {
  const Symbol lhs_element = lhs + "element";
  const Symbol mlir_lhs_element = Symbol("mlir") + lhs + "element_unchecked";

  // Even if we are asked to assign to `lhs`, since the type of `lhs` is not
  // exactly type.CcType(), we need to define a new variable and assign it
  // to `lhs` at the end.
  const Symbol lhs_defined = [&] {
    switch (action) {
      case Action::kDef:
        return lhs;
      case Action::kAssign:
        return lhs + "value";
    }
  }();

  std::string converted_rhs = rhs;
  std::string rhs_element_type;
  switch (kind) {
    case FIELD_KIND_UNSPECIFIED:
      LOG(FATAL) << "FieldKind unspecified.";
    case FIELD_KIND_ATTR: {
      // rhs:            mlir::ArrayAttr;
      // rhs.getValue(): llvm::ArrayRef<mlir::Attribute>
      converted_rhs = absl::StrCat(rhs, ".getValue()");
      rhs_element_type = "mlir::Attribute";
      break;
    }
    case FIELD_KIND_RVAL:
    case FIELD_KIND_LVAL: {
      // rhs: mlir::OperandRange
      converted_rhs = rhs;
      rhs_element_type = "mlir::Value";
      break;
    }
    case FIELD_KIND_STMT: {
      // rhs: mlir::Block*
      converted_rhs = absl::StrCat("*", rhs);
      rhs_element_type = "mlir::Operation&";
      break;
    }
  }

  auto vars = WithVars({
      {"cc_type", type.CcType()},
      {"lhs", lhs.ToCcVarName()},
      {"rhs", converted_rhs},
      {"rhs_element_type", rhs_element_type},
      {"lhs_defined", lhs_defined.ToCcVarName()},
      {"lhs_element", lhs_element.ToCcVarName()},
      {"mlir_lhs_element", mlir_lhs_element.ToCcVarName()},
  });

  Println("$cc_type$ $lhs_defined$;");
  Println("for ($rhs_element_type$ $mlir_lhs_element$ : $rhs$) {");
  {
    auto indent = WithIndent();
    PrintNullableFromIr(ast, Action::kDef, type.element_type(),
                        type.element_maybe_null(), kind, lhs_element,
                        mlir_lhs_element.ToCcVarName(), RhsKind::kListElement);
    Println("$lhs_defined$.push_back(std::move($lhs_element$));");
  }
  Println("}");

  switch (action) {
    case Action::kDef:
      break;
    case Action::kAssign:
      Println("$lhs$ = std::move($lhs_defined$);");
      break;
  }
}

std::string PrintIrToAstSource(const AstDef &ast,
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
