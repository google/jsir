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

#ifndef MALDOCA_ASTGEN_AST_DEF_H_
#define MALDOCA_ASTGEN_AST_DEF_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "maldoca/astgen/ast_def.pb.h"
#include "maldoca/astgen/symbol.h"
#include "maldoca/astgen/type.h"
#include "maldoca/astgen/type.pb.h"

namespace maldoca {

class FieldDef;
class NodeDef;
class AstDef;

class EnumMemberDef {
 public:
  explicit EnumMemberDef(Symbol name, std::string string_value)
      : name_(std::move(name)), string_value_(std::move(string_value)) {}

  static absl::StatusOr<EnumMemberDef> FromEnumMemberDefPb(
      const EnumMemberDefPb &member_pb);

  const Symbol &name() const { return name_; }
  const std::string &string_value() const { return string_value_; }

 private:
  Symbol name_;
  std::string string_value_;
};

class EnumDef {
 public:
  explicit EnumDef(Symbol name, std::vector<EnumMemberDef> members)
      : name_(std::move(name)), members_(std::move(members)) {}

  static absl::StatusOr<EnumDef> FromEnumDefPb(const EnumDefPb &enum_pb);

  const Symbol &name() const { return name_; }
  absl::Span<const EnumMemberDef> members() const { return members_; }

 private:
  Symbol name_;
  std::vector<EnumMemberDef> members_;
};

// Definition of a field in a class.
class FieldDef {
 public:
  static absl::StatusOr<FieldDef> FromFieldDefPb(const FieldDefPb &field_pb,
                                                 absl::string_view lang_name);

  const Symbol &name() const { return name_; }
  Optionalness optionalness() const { return optionalness_; }
  const Type &type() const { return *type_; }
  Type &type() { return *type_; }
  FieldKind kind() const { return kind_; }
  bool ignore_in_ir() const { return ignore_in_ir_; }
  bool enclose_in_region() const { return enclose_in_region_; }

 private:
  // Only allows creation from proto.
  FieldDef() = default;

  Symbol name_;
  Optionalness optionalness_;
  std::unique_ptr<Type> type_;
  FieldKind kind_;
  bool ignore_in_ir_;
  bool enclose_in_region_;
};

// Definition of an AST node type.
// This corresponds to a C++ class.
class NodeDef {
 public:
  // Class name.
  const std::string &name() const { return name_; }

  // Type kind enum.
  //
  // In the JavaScript object version of the AST, a special "type" string
  // represents the kind of the node.
  //
  // interface BinaryExpression <: Expression {
  //   type: "BinaryExpression";  <============ This field.
  //   operator: BinaryOperator;
  //   left: Expression | PrivateName;
  //   right: Expression;
  // }
  //
  // The "type" string only has a concrete value in leaf types.
  //
  // interface Expression <: Node { }  <======= No "type" value defined.
  //
  // The existence of a concrete "type" value suggests that this is a leaf type.
  std::optional<absl::string_view> type() const { return type_; }

  // Fields in the class.
  //
  // This doesn't include fields in base classes.
  absl::Span<const FieldDef> fields() const { return fields_; }

  // The classes that this derives from.
  //
  // For example:
  //
  // interface Identifier <: Expression, Pattern {
  //   type: "Identifier";
  //   name: string;
  // }
  //
  // parents = { Expression, Pattern }
  absl::Span<const NodeDef *const> parents() const { return parents_; }

  // Topologically sorted: base comes before derived. Use the original
  // definition order to break tie.
  //
  // For example:
  //   interface Node;
  //   interface Expression <: Node
  //   interface Pattern <: Node
  //   interface Identifier <: Expression, Pattern
  //
  // ancestors: Node, Expression, Pattern
  absl::Span<const NodeDef *const> ancestors() const { return ancestors_; }

  // All fields, including those defined by ancestors.
  absl::Span<const FieldDef *const> aggregated_fields() const {
    return aggregated_fields_;
  }

  // Direct children of this class.
  absl::Span<const NodeDef *const> children() const { return children_; }

  // All types that directly or indirectly inherit this class.
  absl::Span<const NodeDef *const> descendants() const { return descendants_; }

  // All descendants that are leaf classes.
  absl::Span<const NodeDef *const> leafs() const { return leafs_; }

  std::optional<const EnumDef *> node_type_enum() const {
    if (node_type_enum_.has_value()) {
      return &node_type_enum_.value();
    } else {
      return std::nullopt;
    }
  }

  // Whether an IR op should be automatically generated.
  // If false, the op is expected to be manually written.
  bool should_generate_ir_op() const { return should_generate_ir_op_; }

  // The allowed FieldKinds for this node. Does not include those specified in
  // ancestors.
  //
  // For the meaning of FieldKind, see comments for the proto definition.
  //
  // For example:
  //
  // Expression {
  //   kinds: FIELD_KIND_RVAL
  // }
  // Identifier <: Expression {
  //   kinds: FIELD_KIND_LVAL
  // }
  //
  // For Identifier, kinds() returns [FIELD_KIND_LVAL].
  //
  // In practice, you most likely want aggregate_kinds(), which returns
  // [FIELD_KIND_RVAL, FIELD_KIND_LVAL].
  absl::Span<const FieldKind> kinds() const { return kinds_; }

  // The allowed FieldKinds for this node. Includes those specified in
  // ancestors.
  //
  // For the meaning of FieldKind, see comments for the proto definition.
  //
  // For example:
  //
  // Expression {
  //   kinds: FIELD_KIND_RVAL
  // }
  // Identifier <: Expression {
  //   kinds: FIELD_KIND_LVAL
  // }
  //
  // For Identifier, aggregate_kinds() returns
  // [FIELD_KIND_RVAL, FIELD_KIND_LVAL].
  //
  // You may also use kinds(), which returns [FIELD_KIND_RVAL]. However,
  // aggregated_kinds() is usually the one you want.
  absl::Span<const FieldKind> aggregated_kinds() const {
    return aggregated_kinds_;
  }

  // Whether this node has control-flow-related information.
  //
  // A node is considered to have control-flow-related information if it
  // contains some branch semantics.
  //
  // Example: IfStatement, BreakStatement.
  //
  // When this is true, we define two ops, one in HIR (high-level IR), one in
  // LIR (low-level IR).
  bool has_control_flow() const { return has_control_flow_; }

  // The MLIR op name (C++ class name).
  //
  // <IrName>:
  //   has_control_flow:  <LangName>hir
  //   !has_control_flow: <LangName>ir
  //
  // - Non-leaf type: "<IrName><ClassName>OpInterface"
  // - Leaf type:
  //   - RVal:        "<IrName><ClassName>Op"
  //   - LVal:        "<IrName><ClassName>RefOp"
  //
  // If a custom IR op name is specified (NodeDefPb::ir_op_name), returns that
  // instead.
  //
  // If a custom IR op name is specified for any of the descendants, returns
  // nullopt.
  std::optional<Symbol> ir_op_name(absl::string_view lang_name,
                                   FieldKind kind) const;

  // The stringified MLIR op name (without dialect name).
  //
  // - Non-leaf type: N/A
  // - Leaf type:
  //   - RVal:        "<class_name>"
  //   - LVal:        "<class_name>_ref"
  //
  // If a custom IR op name is specified, returns nullopt.
  //
  // If a custom IR op name is specified for any of the descendants, returns
  // nullopt.
  std::optional<Symbol> ir_op_mnemonic(FieldKind kind) const;

  bool has_fold() const { return has_fold_; }

  // Additional MLIR traits to add to the op definition in ODS.
  absl::Span<const MlirTrait> additional_mlir_traits() const {
    return additional_mlir_traits_;
  }

  // Additional MLIR traits to add to the op definition in ODS, including those
  // from ancestors.
  absl::Span<const MlirTrait> aggregated_additional_mlir_traits() const {
    return aggregated_additional_mlir_traits_;
  }

 private:
  // Only AstDef can create NodeDefs.
  NodeDef() = default;

  std::string name_;
  std::optional<std::string> type_;
  std::vector<FieldDef> fields_;
  std::vector<NodeDef *> parents_;
  std::vector<NodeDef *> ancestors_;
  std::vector<FieldDef *> aggregated_fields_;
  std::vector<NodeDef *> children_;
  std::vector<NodeDef *> descendants_;
  std::vector<NodeDef *> leafs_;
  std::optional<EnumDef> node_type_enum_;
  bool should_generate_ir_op_;
  std::vector<FieldKind> kinds_;
  std::vector<FieldKind> aggregated_kinds_;
  bool has_control_flow_;
  std::optional<std::string> ir_op_name_;
  bool has_fold_;
  std::vector<MlirTrait> additional_mlir_traits_;
  std::vector<MlirTrait> aggregated_additional_mlir_traits_;

  friend class AstDef;
};

// Definition of an AST.
class AstDef {
 public:
  // Creates an AST definition from a proto.
  // Also checks the validity of the proto.
  static absl::StatusOr<AstDef> FromProto(const AstDefPb &pb);

  absl::string_view lang_name() const { return lang_name_; }

  absl::Span<const EnumDef> enum_defs() const { return enum_defs_; }

  // Names of the nodes in the original order.
  absl::Span<const std::string> node_names() const { return node_names_; }

  // Node name => node definition.
  const absl::flat_hash_map<std::string, std::unique_ptr<NodeDef>> &nodes()
      const {
    return nodes_;
  }

  // Nodes listed in topological order.
  // This order ensures that dependencies (parent classes, field types) are
  // defined before each class.
  absl::Span<const NodeDef *const> topological_sorted_nodes() const {
    return topological_sorted_nodes_;
  }

 private:
  explicit AstDef(
      std::string lang_name, std::vector<EnumDef> enum_defs,
      std::vector<std::string> node_names,
      absl::flat_hash_map<std::string, std::unique_ptr<NodeDef>> nodes,
      std::vector<NodeDef *> topological_sorted_nodes)
      : lang_name_(std::move(lang_name)),
        enum_defs_(std::move(enum_defs)),
        node_names_(std::move(node_names)),
        nodes_(std::move(nodes)),
        topological_sorted_nodes_(std::move(topological_sorted_nodes)) {
    LOG(INFO) << "Created AstDef. node_names:";
    for (const std::string &node_name : node_names_) {
      LOG(INFO) << "  " << node_name;
    }
  }

  std::string lang_name_;
  std::vector<EnumDef> enum_defs_;
  std::vector<std::string> node_names_;
  absl::flat_hash_map<std::string, std::unique_ptr<NodeDef>> nodes_;
  std::vector<NodeDef *> topological_sorted_nodes_;

  static void ResolveClassType(
      Type &type, absl::Span<const NodeDef *const> topological_sorted_nodes);
};

}  // namespace maldoca

#endif  // MALDOCA_ASTGEN_AST_DEF_H_
