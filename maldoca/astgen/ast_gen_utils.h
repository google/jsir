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

#ifndef MALDOCA_ASTGEN_AST_GEN_UTILS_H_
#define MALDOCA_ASTGEN_AST_GEN_UTILS_H_

#include <stddef.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "maldoca/astgen/ast_def.h"
#include "maldoca/astgen/ast_def.pb.h"
#include "maldoca/astgen/symbol.h"
#include "maldoca/astgen/type.h"
#include "maldoca/astgen/type.pb.h"
#include "maldoca/base/path.h"
#include "google/protobuf/io/printer.h"

namespace maldoca {

inline constexpr absl::string_view kJsonValueVariableName = "json";
inline constexpr absl::string_view kOsValueVariableName = "os";

inline std::string GetAstHeaderPath(absl::string_view ast_path) {
  return JoinPath(ast_path, "ast.generated.h");
}

inline std::string GetAstVisitorHeaderPath(absl::string_view ast_path) {
  return JoinPath(ast_path, "ast_visitor.generated.h");
}

inline std::string GetAstWalkerHeaderPath(absl::string_view ast_path) {
  return JoinPath(ast_path, "ast_walker.generated.h");
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
  explicit IfStmtPrinter(google::protobuf::io::Printer* printer)
      : is_first_(true), printer_(printer) {}

  struct IfStmtCase {
    std::function<void()> condition;
    std::function<void()> body;
  };

  void PrintCase(const IfStmtCase& kase) {
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
  google::protobuf::io::Printer* printer_;
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
inline std::string UnIndentedSource(absl::string_view source) {
  source = absl::StripTrailingAsciiWhitespace(source);

  std::vector<std::string> lines = absl::StrSplit(source, '\n');

  // Remove leading empty lines.
  lines.erase(lines.begin(), absl::c_find_if(lines, [](const auto& line) {
                return !line.empty();
              }));

  size_t min_indent = absl::c_accumulate(
      lines, std::numeric_limits<size_t>::max(),
      [](size_t current_min, const std::string& line) {
        size_t first_non_whitespace = line.find_first_not_of(' ');
        if (first_non_whitespace == std::string::npos) {
          return current_min;
        }
        return std::min(current_min, first_non_whitespace);
      });

  for (auto& line : lines) {
    if (line.size() >= min_indent) {
      line.erase(0, min_indent);
    }
  }

  return absl::StrJoin(lines, "\n");
}

// FieldIs{Argument,Region}:
//
// A field only defines something in the op if it exists on both sides of the
// AST/IR correspondence, since the op and the AST<->IR conversions are how that
// correspondence is materialized.
//
// Example: Node::start is AST_ONLY, so it does not lead to any argument/region
// in JSIR because we want to store the information in mlir::Location.
// Conversely, an IR_ONLY field (such as the trivia carried by an attribute) has
// no AST counterpart to convert from or to.
//
// If a field has enclose_in_region(), then it's an MLIR "region"; otherwise
// it's an MLIR "argument".
//
// An argument is either an mlir::Attribute or an mlir::Value;
// A region is an mlir::Region.
//
// See FieldDefPb::enclose_in_region for why we need to enclose certain fields
// in a region.
//
// A field split by the `ir_name`s of its alternatives (see
// FieldDef::ir_splits()) is held in several arguments/regions, so it can be
// both an argument and a region. A field that
// is derived_in_ir() is neither: it is not stored in the op, but recomputed
// when converting the op back to the AST.

// Whether the field takes part in the AST<->IR conversion: it is either stored
// in the op, or recomputed from it (derived_in_ir()).
inline bool FieldIsInIrOp(const FieldDef* field) {
  return field->in_ast() && field->in_ir();
}

// Whether the field is split across several operands/regions, one per
// `ir_name` of its alternatives. See FieldDef::ir_splits().
inline bool FieldIsSplit(const FieldDef* field) {
  return !field->ir_splits().empty();
}

// Whether the field is held either in a literal attribute or in an operand. See
// FieldIrMappingPb::literal_attr.
inline bool FieldHasLiteralAttr(const FieldDef* field) {
  return field->ir_mapping() != nullptr &&
         field->ir_mapping()->has_literal_attr();
}

inline bool FieldIsArgument(const FieldDef* field) {
  if (!FieldIsInIrOp(field) || field->derived_in_ir()) {
    return false;
  }
  if (FieldIsSplit(field)) {
    return absl::c_any_of(field->ir_splits(), [](const auto& split) {
      return split.has_operand();
    });
  }
  if (FieldHasLiteralAttr(field)) {
    return true;
  }
  return !field->enclose_in_region();
}

inline bool FieldIsRegion(const FieldDef* field) {
  if (!FieldIsInIrOp(field) || field->derived_in_ir()) {
    return false;
  }
  if (FieldIsSplit(field)) {
    return absl::c_any_of(field->ir_splits(),
                          [](const auto& split) { return split.has_region(); });
  }
  if (FieldHasLiteralAttr(field)) {
    return false;
  }
  return field->enclose_in_region();
}

// The name of the single argument/region that holds a field without a split or
// a literal attribute: `ir_mapping.operand`/`ir_mapping.region` if set,
// otherwise the field name.
inline Symbol FieldIrName(const FieldDef& field) {
  if (const FieldIrMappingPb* mapping = field.ir_mapping()) {
    if (mapping->has_operand()) {
      return Symbol(mapping->operand());
    }
    if (mapping->has_region()) {
      return Symbol(mapping->region());
    }
  }
  return field.name();
}

// The kind a field has in the op of `op_kind`: `ir_mapping.lval_op_kind` in an
// LVAL op, then `ir_mapping.kind`, then the field's own kind.
inline FieldKind FieldIrKind(const FieldDef& field, FieldKind op_kind) {
  if (const FieldIrMappingPb* mapping = field.ir_mapping()) {
    if (op_kind == FIELD_KIND_LVAL && mapping->has_lval_op_kind()) {
      return mapping->lval_op_kind();
    }
    if (mapping->has_kind()) {
      return mapping->kind();
    }
  }
  return field.kind();
}

// Builds the type `name` (a class) or `name1 | name2 | ...` (a variant).
inline std::unique_ptr<Type> MakeClassOrVariantType(
    absl::Span<const std::string> names, absl::string_view lang_name) {
  TypePb pb;
  if (names.size() == 1) {
    pb.set_class_(names.front());
  } else {
    for (const std::string& name : names) {
      pb.mutable_variant()->add_types()->set_class_(name);
    }
  }
  auto type = FromTypePb(pb, lang_name);
  CHECK_OK(type.status());
  return *std::move(type);
}

// If the op of `op_kind` holds a narrower type than the field's AST type,
// returns that type; otherwise returns nullptr.
//
// - In an LVAL op, `ir_mapping.lval_op_types` (if set) replaces the AST types.
// - For a variant field held as an rvalue/lvalue, members whose node doesn't
//   support that kind can't be held in the op. For example,
//   `ObjectProperty.value: Expression | Pattern` holds an `Expression` in the
//   `object_property` op and a `Pattern` in the `object_property_ref` op.
//
// The narrowed type must be a single class. When converting the AST to the IR,
// the field value is downcast to it; when converting back, the value is
// downcast to the first variant member it is an instance of.
inline std::unique_ptr<Type> NarrowedIrType(const AstDef& ast,
                                            const FieldDef& field,
                                            FieldKind op_kind) {
  if (FieldIsSplit(&field) || FieldHasLiteralAttr(&field)) {
    return nullptr;
  }
  if (const FieldIrMappingPb* mapping = field.ir_mapping();
      mapping != nullptr && op_kind == FIELD_KIND_LVAL &&
      !mapping->lval_op_types().empty()) {
    CHECK_EQ(mapping->lval_op_types().size(), 1)
        << "Only a single lval_op_types is supported.";
    std::vector<std::string> names{mapping->lval_op_types().begin(),
                                   mapping->lval_op_types().end()};
    return MakeClassOrVariantType(names, ast.lang_name());
  }

  FieldKind kind = FieldIrKind(field, op_kind);
  if (kind != FIELD_KIND_RVAL && kind != FIELD_KIND_LVAL) {
    return nullptr;
  }
  if (!field.type().IsA<VariantType>()) {
    return nullptr;
  }
  const auto& variant = static_cast<const VariantType&>(field.type());
  std::vector<std::string> supported;
  for (const auto& member : variant.types()) {
    if (!member->IsA<ClassType>()) {
      return nullptr;
    }
    std::string name =
        static_cast<const ClassType&>(*member).name().ToPascalCase();
    auto it = ast.nodes().find(name);
    if (it == ast.nodes().end() ||
        absl::c_linear_search(it->second->aggregated_kinds(), kind)) {
      supported.push_back(std::move(name));
    }
  }
  if (supported.size() == variant.types().size()) {
    return nullptr;
  }
  CHECK_EQ(supported.size(), 1u)
      << field.name().ToCcVarName()
      << ": Exactly one variant member must support the kind "
      << FieldKind_Name(kind) << ".";
  return MakeClassOrVariantType(supported, ast.lang_name());
}

// The index of the variant member (or 0 for a class type) that holds the AST
// node type `type_name`: either that member is `type_name` itself, or
// `type_name` descends from it. Returns nullopt if no member does.
//
// Sets `*exact` to whether the member is `type_name` itself.
inline std::optional<size_t> FindMemberHolding(const AstDef& ast,
                                               const Type& type,
                                               absl::string_view type_name,
                                               bool* exact) {
  std::vector<std::string> member_names;
  if (type.IsA<VariantType>()) {
    for (const auto& member : static_cast<const VariantType&>(type).types()) {
      if (member->IsA<ClassType>()) {
        member_names.push_back(
            static_cast<const ClassType&>(*member).name().ToPascalCase());
      } else {
        member_names.push_back("");
      }
    }
  } else if (type.IsA<ClassType>()) {
    member_names.push_back(
        static_cast<const ClassType&>(type).name().ToPascalCase());
  }

  for (size_t i = 0; i != member_names.size(); ++i) {
    if (member_names[i] == type_name) {
      *exact = true;
      return i;
    }
  }
  auto it = ast.nodes().find(type_name);
  if (it == ast.nodes().end()) {
    return std::nullopt;
  }
  for (size_t i = 0; i != member_names.size(); ++i) {
    if (absl::c_any_of(it->second->ancestors(), [&](const NodeDef* ancestor) {
          return ancestor->name() == member_names[i];
        })) {
      *exact = false;
      return i;
    }
  }
  return std::nullopt;
}

// The class names of the members of a variant type (or of a class type).
inline std::vector<std::string> MemberNames(const Type& type) {
  std::vector<std::string> names;
  if (type.IsA<VariantType>()) {
    for (const auto& member : static_cast<const VariantType&>(type).types()) {
      if (member->IsA<ClassType>()) {
        names.push_back(
            static_cast<const ClassType&>(*member).name().ToPascalCase());
      }
    }
  } else if (type.IsA<ClassType>()) {
    names.push_back(static_cast<const ClassType&>(type).name().ToPascalCase());
  }
  return names;
}

// Whether a class type refers to an AST node, i.e. is not marked
// `is_plain_object` in the AST definition. See NodeDefPb::is_plain_object.
inline bool IsAstNodeClass(const AstDef& ast, const ClassType& type) {
  auto it = ast.nodes().find(type.name().ToPascalCase());
  return it == ast.nodes().end() || !it->second->is_plain_object();
}

// Whether a variant holds at least one AST node type.
inline bool VariantHasAstNode(const AstDef& ast, const VariantType& variant) {
  for (const auto& scalar : variant.types()) {
    if (scalar->IsA<ClassType>() &&
        IsAstNodeClass(ast, static_cast<const ClassType&>(*scalar))) {
      return true;
    }
  }
  return false;
}

// Whether a field of this type holds a child AST node, i.e. whether traversal
// descends into it. Lists and variants of AST nodes count; plain-object
// classes do not.
inline bool IsAstNodeType(const AstDef& ast, const Type& type) {
  if (type.IsA<ClassType>()) {
    return IsAstNodeClass(ast, static_cast<const ClassType&>(type));
  }
  if (type.IsA<VariantType>()) {
    return VariantHasAstNode(ast, static_cast<const VariantType&>(type));
  }
  if (type.IsA<ListType>()) {
    const NonListType& element =
        static_cast<const ListType&>(type).element_type();
    if (element.IsA<ClassType>()) {
      return IsAstNodeClass(ast, static_cast<const ClassType&>(element));
    }
    if (element.IsA<VariantType>()) {
      return VariantHasAstNode(ast, static_cast<const VariantType&>(element));
    }
    return false;
  }
  return false;
}

}  // namespace maldoca

#endif  // MALDOCA_ASTGEN_AST_GEN_UTILS_H_
