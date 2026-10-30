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

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
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

// Name of the well-known node that holds a node's source location, comments,
// and symbol references, and that lowers to the IR's `mlir::Location`
// attribute. Nodes with `ir_attr_has_loc` get an IR-only `loc` field of this
// type.
inline constexpr absl::string_view kTriviaNodeName = "Trivia";

class EnumMemberDef {
 public:
  explicit EnumMemberDef(Symbol name, std::string string_value)
      : name_(std::move(name)), string_value_(std::move(string_value)) {}

  static absl::StatusOr<EnumMemberDef> FromEnumMemberDefPb(
      const EnumMemberDefPb& member_pb);

  const Symbol& name() const { return name_; }
  const std::string& string_value() const { return string_value_; }

 private:
  Symbol name_;
  std::string string_value_;
};

class EnumDef {
 public:
  explicit EnumDef(Symbol name, std::vector<EnumMemberDef> members)
      : name_(std::move(name)), members_(std::move(members)) {}

  static absl::StatusOr<EnumDef> FromEnumDefPb(const EnumDefPb& enum_pb);

  const Symbol& name() const { return name_; }
  absl::Span<const EnumMemberDef> members() const { return members_; }

 private:
  Symbol name_;
  std::vector<EnumMemberDef> members_;
};

// Definition of a field in a class.
class FieldDef {
 public:
  static absl::StatusOr<FieldDef> FromFieldDefPb(const FieldDefPb& field_pb,
                                                 absl::string_view lang_name);

  const Symbol& name() const { return name_; }
  Optionalness optionalness() const { return optionalness_; }
  const Type& type() const { return *type_; }
  Type& type() { return *type_; }
  FieldKind kind() const { return kind_; }
  GenerationTarget generation_target() const { return generation_target_; }
  bool in_ast() const {
    return generation_target_ == GENERATION_TARGET_BOTH ||
           generation_target_ == GENERATION_TARGET_AST_ONLY;
  }
  bool in_ir() const {
    return generation_target_ == GENERATION_TARGET_BOTH ||
           generation_target_ == GENERATION_TARGET_IR_ONLY;
  }
  bool enclose_in_region() const { return enclose_in_region_; }

  // Whether the generated MLIR attribute parameter is wrapped in
  // `OptionalParameter<>`, which lets callers omit it.
  //
  // This is opt-in rather than derived from optionalness(): existing optional
  // class-typed fields are spelled unwrapped, and wrapping them would change
  // the builder APIs of the attributes that already ship.
  bool ir_attr_optional_parameter() const {
    return ir_attr_optional_parameter_;
  }

  // How the IR holds a field is resolved by AstDef::FromProto() from the
  // field's `ir_name` and `alternatives` (see FieldDefPb). By default, the IR
  // op of the field's node holds the field in one argument or region named
  // after it, with the field's type and kind. The accessors below describe how
  // a field departs from that.

  // One of the operands/regions of a field split across several of them: the
  // alternatives sharing an `ir_name`. See ir_splits().
  struct IrSplit {
    // Name of the optional operand or region.
    std::string name;

    // Whether `name` is a region (a `StmtRegion` for FIELD_KIND_STMT, an
    // `ExprRegion` for FIELD_KIND_RVAL/FIELD_KIND_LVAL) rather than an operand.
    bool is_region = false;

    // Kind of the members held in it: FIELD_KIND_STMT, FIELD_KIND_RVAL or
    // FIELD_KIND_LVAL.
    FieldKind kind = FIELD_KIND_UNSPECIFIED;

    // The variant members (node names) held in it.
    std::vector<std::string> types;
  };

  // The attribute that holds a field instead of its operand, depending on a
  // bool sibling field: the field's `unless` alternatives. See literal_attr().
  //
  // Example: a static member expression `a.b` holds its property in the
  // `literal_property` attribute, whereas a computed one `a[b]` holds it in the
  // `computed_property` operand (ir_name()).
  struct LiteralAttr {
    // Name of the optional attribute.
    std::string name;

    // The node types the attribute can hold, each with
    // `should_generate_ir_attr`. The AST -> IR conversion dispatches on the
    // field value's runtime type over them.
    std::vector<std::string> types;

    // Name of the required bool sibling field (in camelCase). When it's false,
    // the field is held in the attribute; when it's true, in the operand. It
    // isn't stored in the IR (see derived_in_ir()): the IR -> AST conversion
    // recomputes it from which of them is set.
    std::string discriminator;
  };

  // How the node's LVAL (`*RefOp`) op holds the field, when that differs from
  // its other ops: the field's `op_kind: FIELD_KIND_LVAL` alternative. See
  // lval_op_override().
  struct LvalOpOverride {
    // Kind of the field in the LVAL op.
    FieldKind kind = FIELD_KIND_UNSPECIFIED;

    // The class the LVAL op holds, for a class-typed field holding another
    // class there. The AST -> IR conversion falls back to the field's own type
    // (and emits an error) when the AST holds a node that isn't of this type,
    // so the IR -> AST conversion accepts both.
    //
    // Example: `ParenthesizedExpression.expression` is an `LVal` in the
    // `parenthesized_expression_ref` op.
    std::optional<std::string> type;
  };

  // Name of the single operand/region that holds the field (also the operand
  // of a field with a literal_attr()): its `ir_name` if set, otherwise its
  // name. Unused for a field with ir_splits().
  Symbol ir_name() const {
    return ir_name_.has_value() ? Symbol(*ir_name_) : name_;
  }

  // Whether the field is left out of the IR because it is recomputed from how
  // another field is represented, i.e. whether a sibling field's alternatives
  // name it in `when`/`unless` (it's that field's literal_attr()
  // discriminator).
  //
  // Example: `MemberExpression.computed` is true iff the IR holds `property`
  // in the `computed_property` operand rather than the `literal_property`
  // attribute.
  bool derived_in_ir() const { return derived_in_ir_; }

  // If the field's `alternatives` have `ir_name`s (and no selectors), the
  // field is split across several IR operands/regions: one per distinct
  // `ir_name`, in the order the names first appear, of which exactly one is
  // set (at most one if the field is optional). Empty otherwise.
  absl::Span<const IrSplit> ir_splits() const { return ir_splits_; }

  // If the field has `when`/`unless` alternatives, the attribute that holds it
  // unless the discriminator is true. nullptr otherwise.
  const LiteralAttr* literal_attr() const {
    return literal_attr_.has_value() ? &*literal_attr_ : nullptr;
  }

  // If the field has `op_kind` alternatives and the node's LVAL op holds it
  // with another kind or type than its other ops, how. nullptr otherwise.
  const LvalOpOverride* lval_op_override() const {
    return lval_op_override_.has_value() ? &*lval_op_override_ : nullptr;
  }

  // For a field with `alternatives` without `ir_name`s, the kind of the IR op
  // holding the variant member `member_name` if it differs from the field's
  // kind (FIELD_KIND_LVAL for an lvalue alternative of the FIELD_KIND_RVAL
  // field), or nullopt.
  std::optional<FieldKind> member_ir_kind(absl::string_view member_name) const {
    auto it = member_ir_kinds_.find(member_name);
    if (it == member_ir_kinds_.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  // Whether member_ir_kind() is set for some variant member.
  bool has_member_ir_kinds() const { return !member_ir_kinds_.empty(); }

 private:
  // An alternative of the field (see FieldAlternativePb).
  struct Alternative {
    // The class names of the alternative's type, or empty if it holds the
    // field's whole type.
    std::vector<std::string> member_names;
    std::optional<std::string> ir_name;
    std::optional<FieldKind> kind;
    std::optional<std::string> when;
    std::optional<std::string> unless;
    std::optional<FieldKind> op_kind;

    // How the alternative is named in error messages.
    std::string DebugName() const;
  };

  // Only allows creation from proto.
  FieldDef() = default;

  Symbol name_;
  Optionalness optionalness_;
  std::unique_ptr<Type> type_;
  FieldKind kind_;
  GenerationTarget generation_target_ = GENERATION_TARGET_BOTH;
  bool enclose_in_region_;
  bool ir_attr_optional_parameter_ = false;
  bool derived_in_ir_ = false;

  // As declared in the proto (empty if the field has no `alternatives`).
  std::vector<Alternative> alternatives_;

  // Resolved from `ir_name` and `alternatives_` by FromFieldDefPb() and
  // AstDef::ResolveAlternatives().
  std::optional<std::string> ir_name_;
  std::vector<IrSplit> ir_splits_;
  std::optional<LiteralAttr> literal_attr_;
  std::optional<LvalOpOverride> lval_op_override_;
  absl::flat_hash_map<std::string, FieldKind> member_ir_kinds_;

  friend class AstDef;
};

// Definition of an AST node type.
// This corresponds to a C++ class.
class NodeDef {
 public:
  // Class name.
  const std::string& name() const { return name_; }

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
  absl::Span<const NodeDef* const> parents() const { return parents_; }

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
  absl::Span<const NodeDef* const> ancestors() const { return ancestors_; }

  // All fields, including those defined by ancestors.
  absl::Span<const FieldDef* const> aggregated_fields() const {
    return aggregated_fields_;
  }

  bool has_ast_fields() const {
    for (const auto& field : fields_) {
      if (field.in_ast()) return true;
    }
    return false;
  }

  bool has_aggregated_ast_fields() const {
    for (const auto* field : aggregated_fields_) {
      if (field->in_ast()) return true;
    }
    return false;
  }

  // Direct children of this class.
  absl::Span<const NodeDef* const> children() const { return children_; }

  // All types that directly or indirectly inherit this class.
  absl::Span<const NodeDef* const> descendants() const { return descendants_; }

  // All descendants that are leaf classes.
  absl::Span<const NodeDef* const> leaves() const { return leaves_; }

  std::optional<const EnumDef*> node_type_enum() const {
    if (node_type_enum_.has_value()) {
      return &node_type_enum_.value();
    } else {
      return std::nullopt;
    }
  }

  // Whether an IR op should be automatically generated.
  // If false, the op is expected to be manually written.
  bool should_generate_ir_op() const { return should_generate_ir_op_; }

  // Whether IR dispatch code should be automatically generated for unions.
  bool should_generate_dispatch() const { return should_generate_dispatch_; }

  // Overrides for generated dispatch code.
  //
  // For a union type (e.g., `Expression`), the generator automatically
  // produces dispatch code (e.g., `dynamic_cast` chain in AST-to-IR, or
  // `TypeSwitch` in IR-to-AST) to route to the correct visitor for each member
  // (e.g., `VisitIdentifier`).
  //
  // `DispatchOverride` allows customizing this routing for specific members.
  struct DispatchOverride {
    // The name of the visitor function to call.
    // E.g., "VisitIdentifierRef" instead of the default "VisitIdentifier".
    std::string visitor;

    // The IR op name to use for this member.
    // E.g., "jsir.IdentifierRef" instead of "jsir.Identifier".
    std::optional<std::string> ir_op_name;
  };

  // Map from union member type name (e.g., "Identifier") to its dispatch
  // override.
  const absl::flat_hash_map<std::string, DispatchOverride>& dispatch_overrides()
      const {
    return dispatch_overrides_;
  }

  // Set of union member type names to skip in the generated dispatch code.
  //
  // If a member is skipped, it will not be included in the generated
  // dispatch methods, and must be handled manually if needed.
  //
  // E.g., {"InvalidExpression"} to skip generating dispatch for invalid nodes.
  const absl::flat_hash_set<std::string>& dispatch_skip() const {
    return dispatch_skip_;
  }

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

  // Whether the op of `kind` has a folder. See NodeDefPb::fold_kinds.
  bool has_fold(FieldKind kind) const {
    return has_fold_ &&
           (fold_kinds_.empty() || absl::c_linear_search(fold_kinds_, kind));
  }

  // Whether an IR attribute should be automatically generated.
  bool should_generate_ir_attr() const { return should_generate_ir_attr_; }

  // Whether this attribute has a "$loc: JsirTriviaAttr" parameter.
  //
  // This is sugar for an IR-only `loc` field of type `Trivia`, which is
  // synthesized in AstDef::FromProto. Prefer in_ir()/aggregated_fields() over
  // this accessor when generating code.
  bool ir_attr_has_loc() const { return ir_attr_has_loc_; }

  // Target representation for code generation of this node.
  GenerationTarget generation_target() const { return generation_target_; }

  // Whether this node is generated in the AST.
  bool in_ast() const {
    return generation_target_ == GENERATION_TARGET_BOTH ||
           generation_target_ == GENERATION_TARGET_AST_ONLY;
  }

  // Whether this node is generated in the IR.
  bool in_ir() const {
    return generation_target_ == GENERATION_TARGET_BOTH ||
           generation_target_ == GENERATION_TARGET_IR_ONLY;
  }

  // Whether the generated attribute is a `LocationAttrDef` rather than a plain
  // `AttrDef`, i.e. a custom `::mlir::LocationAttr` subclass.
  bool ir_attr_is_location() const { return ir_attr_is_location_; }

  // Whether this class is plain data attached to the AST (e.g. source
  // locations, comments) rather than a node carrying information necessary to
  // represent the program's structure. See NodeDefPb::is_plain_object.
  bool is_plain_object() const { return is_plain_object_; }

  // Additional traits/interfaces to add to the AttrDef definition.
  absl::Span<const std::string> additional_attr_traits() const {
    return additional_attr_traits_;
  }

  // The MLIR attribute name (C++ class name): "<LangName>ir<ClassName>Attr"
  Symbol ir_attr_name(absl::string_view lang_name) const;

  // The stringified MLIR attribute mnemonic (without dialect name):
  // "<class_name>"
  Symbol ir_attr_mnemonic() const;

  // Additional MLIR traits to add to the op definition in ODS.
  absl::Span<const MlirTrait> additional_mlir_traits() const {
    return additional_mlir_traits_;
  }

  // Additional MLIR traits to add to the op definition in ODS, including those
  // from ancestors, minus the node's excluded ones (see
  // NodeDefPb::excluded_mlir_traits).
  absl::Span<const MlirTrait> aggregated_additional_mlir_traits() const {
    return aggregated_additional_mlir_traits_;
  }

 private:
  // Only AstDef can create NodeDefs.
  NodeDef() = default;

  std::string name_;
  std::optional<std::string> type_;
  std::vector<FieldDef> fields_;
  std::vector<NodeDef*> parents_;
  std::vector<NodeDef*> ancestors_;
  std::vector<FieldDef*> aggregated_fields_;
  std::vector<NodeDef*> children_;
  std::vector<NodeDef*> descendants_;
  std::vector<NodeDef*> leaves_;
  std::optional<EnumDef> node_type_enum_;
  bool should_generate_ir_op_;
  bool should_generate_ir_attr_ = false;
  bool ir_attr_has_loc_ = false;
  GenerationTarget generation_target_ = GENERATION_TARGET_BOTH;
  bool ir_attr_is_location_ = false;
  bool is_plain_object_ = false;
  std::vector<std::string> additional_attr_traits_;
  std::vector<FieldKind> kinds_;
  std::vector<FieldKind> aggregated_kinds_;
  bool has_control_flow_;
  std::optional<std::string> ir_op_name_;
  bool should_generate_dispatch_ = true;
  absl::flat_hash_map<std::string, DispatchOverride> dispatch_overrides_;
  absl::flat_hash_set<std::string> dispatch_skip_;
  bool has_fold_;
  std::vector<FieldKind> fold_kinds_;
  std::vector<MlirTrait> additional_mlir_traits_;
  std::vector<MlirTrait> excluded_mlir_traits_;
  std::vector<MlirTrait> aggregated_additional_mlir_traits_;

  friend class AstDef;
};

// Definition of an AST.
class AstDef {
 public:
  // Creates an AST definition from a proto.
  // Also checks the validity of the proto.
  static absl::StatusOr<AstDef> FromProto(const AstDefPb& pb);

  absl::string_view lang_name() const { return lang_name_; }

  absl::Span<const EnumDef> enum_defs() const { return enum_defs_; }

  // Names of the nodes in the original order.
  absl::Span<const std::string> node_names() const { return node_names_; }

  // Node name => node definition.
  const absl::flat_hash_map<std::string, std::unique_ptr<NodeDef>>& nodes()
      const {
    return nodes_;
  }

  // Nodes listed in topological order.
  // This order ensures that dependencies (parent classes, field types) are
  // defined before each class.
  absl::Span<const NodeDef* const> topological_sorted_nodes() const {
    return topological_sorted_nodes_;
  }

 private:
  explicit AstDef(
      std::string lang_name, std::vector<EnumDef> enum_defs,
      std::vector<std::string> node_names,
      absl::flat_hash_map<std::string, std::unique_ptr<NodeDef>> nodes,
      std::vector<NodeDef*> topological_sorted_nodes)
      : lang_name_(std::move(lang_name)),
        enum_defs_(std::move(enum_defs)),
        node_names_(std::move(node_names)),
        nodes_(std::move(nodes)),
        topological_sorted_nodes_(std::move(topological_sorted_nodes)) {
    LOG(INFO) << "Created AstDef. node_names:";
    for (const std::string& node_name : node_names_) {
      LOG(INFO) << "  " << node_name;
    }
  }

  std::string lang_name_;
  std::vector<EnumDef> enum_defs_;
  std::vector<std::string> node_names_;
  absl::flat_hash_map<std::string, std::unique_ptr<NodeDef>> nodes_;
  std::vector<NodeDef*> topological_sorted_nodes_;

  static void ResolveClassType(
      Type& type, absl::Span<const NodeDef* const> topological_sorted_nodes);

  // Infers the kinds of the field's `alternatives`, validates them against the
  // rest of `node`, and resolves the field's kind(), ir_name(), ir_splits(),
  // literal_attr(), lval_op_override() and member_ir_kind()s. Marks the bool
  // sibling named by `when`/`unless` (if any) as derived_in_ir().
  static absl::Status ResolveAlternatives(
      const absl::flat_hash_map<std::string, std::unique_ptr<NodeDef>>& nodes,
      NodeDef& node, FieldDef& field);

  // ResolveAlternatives() for alternatives with `when`/`unless`.
  static absl::Status ResolveSiblingAlternatives(
      const absl::flat_hash_map<std::string, std::unique_ptr<NodeDef>>& nodes,
      NodeDef& node, FieldDef& field, absl::Span<const FieldKind> kinds);

  // ResolveAlternatives() for alternatives with `op_kind`.
  static absl::Status ResolveOpKindAlternatives(
      const absl::flat_hash_map<std::string, std::unique_ptr<NodeDef>>& nodes,
      const NodeDef& node, FieldDef& field, absl::Span<const FieldKind> kinds);
};

}  // namespace maldoca

#endif  // MALDOCA_ASTGEN_AST_DEF_H_
