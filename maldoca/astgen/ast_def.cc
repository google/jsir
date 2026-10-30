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

#include "maldoca/astgen/ast_def.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "maldoca/astgen/ast_def.pb.h"
#include "maldoca/astgen/symbol.h"
#include "maldoca/astgen/type.h"

namespace maldoca {
namespace {

// NOTE: DO NOT USE this internal function directly.
// Use TopologicalSortDependencies(node, get_dependencies) instead.
//
// - get_dependencies: A function that returns the dependencies of a node.
//
// - pre_order_visited: Internal state.
//   See comments within the function.
//
// - sorted_dependencies: Output vector.
//   Topologically sorted dependencies are appended here.
//   See comments within the function.
void TopologicalSortDependencies(
    NodeDef* node,
    std::function<std::vector<NodeDef*>(NodeDef*)> get_dependencies,
    absl::flat_hash_set<NodeDef*>* pre_order_visited,
    std::vector<NodeDef*>* sorted_dependencies) {
  // We run a DFS to perform topological sort.
  //
  // We maintain two sets:
  // - sorted_dependencies: The result vector being constructed.
  // - pre_order_visited: nodes in the recursion stack.
  //
  // Each node in the graph can either:
  // 1) Appear in neither.
  // 2) Appear in "pre_order_visited";
  // 3) Appear in "sorted_dependencies".
  //
  // Each node is inserted to "pre_order_visited" pre-order; moved to
  // "sorted_dependencies" post-order. If a node is already in
  // "sorted_dependencies", skip this node (typical DFS); If a node is already
  // in "pre_order_visited", this means the graph has cycle!
  std::vector<NodeDef*> dependencies = get_dependencies(node);
  for (NodeDef* dependency : dependencies) {
    CHECK(!pre_order_visited->contains(dependency)) << "Graph has cycle!";
    if (absl::c_linear_search(*sorted_dependencies, dependency)) {
      continue;
    }

    pre_order_visited->insert(dependency);
    TopologicalSortDependencies(dependency, get_dependencies, pre_order_visited,
                                sorted_dependencies);
    pre_order_visited->erase(dependency);
    sorted_dependencies->push_back(dependency);
  }
}

// Performs a topological sort on all the (transitive) dependencies of `node`.
//
// For example: (A <: B means A depends on B)
//
// Input graph:
//   CatDog <: Cat, Dog
//   Cat <: Animal
//   Dog <: Animal
//
// TopologicalSortDependencies(CatDog):
//   Animal, Cat, Dog
//
// Note: We use the original order of dependencies to break tie. For example,
// Cat appears before Dog and this is preserved.
std::vector<NodeDef*> TopologicalSortDependencies(
    NodeDef* node,
    std::function<std::vector<NodeDef*>(NodeDef*)> get_dependencies) {
  absl::flat_hash_set<NodeDef*> pre_order_visited;
  std::vector<NodeDef*> sorted_dependencies;
  TopologicalSortDependencies(node, get_dependencies, &pre_order_visited,
                              &sorted_dependencies);
  return sorted_dependencies;
}

// Gets the dependency nodes of a given type.
//
// In the generated C++ code, these nodes must be defined before the type is
// used.
//
// - nodes: All nodes in the AST.
// - dependencies: Output vector. Dependencies are appended to this vector.
void GetDependencies(
    const Type& type,
    const absl::flat_hash_map<std::string, std::unique_ptr<NodeDef>>& nodes,
    std::vector<NodeDef*>* dependencies) {
  switch (type.kind()) {
    case TypeKind::kBuiltin:
    case TypeKind::kEnum:
      // No dependencies.
      break;

    case TypeKind::kClass: {
      const auto& class_type = static_cast<const ClassType&>(type);
      auto it = nodes.find(class_type.name().ToPascalCase());
      CHECK(it != nodes.end())
          << class_type.name().ToPascalCase() << " undefined.";
      dependencies->push_back(it->second.get());
      break;
    }

    case TypeKind::kList: {
      const auto& list_type = static_cast<const ListType&>(type);
      GetDependencies(list_type.element_type(), nodes, dependencies);
      break;
    }

    case TypeKind::kVariant: {
      const auto& variant_type = static_cast<const VariantType&>(type);
      for (const auto& type : variant_type.types()) {
        GetDependencies(*type, nodes, dependencies);
      }
      break;
    }
  }
}

}  // namespace

absl::StatusOr<EnumMemberDef> EnumMemberDef::FromEnumMemberDefPb(
    const EnumMemberDefPb& member_pb) {
  Symbol name{member_pb.name()};
  if (name.ToPascalCase() != member_pb.name()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The enum member name '", member_pb.name(), "' is not in PascalCase."));
  }

  return EnumMemberDef{std::move(name), std::move(member_pb.string_value())};
}

absl::StatusOr<EnumDef> EnumDef::FromEnumDefPb(const EnumDefPb& enum_pb) {
  Symbol name{enum_pb.name()};
  if (name.ToPascalCase() != enum_pb.name()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The enum type name '", enum_pb.name(), "' is not in PascalCase."));
  }

  std::vector<EnumMemberDef> members;
  for (const EnumMemberDefPb& member_pb : enum_pb.members()) {
    ABSL_ASSIGN_OR_RETURN(auto member,
                          EnumMemberDef::FromEnumMemberDefPb(member_pb));
    members.push_back(std::move(member));
  }

  return EnumDef{std::move(name), std::move(members)};
}

absl::StatusOr<FieldDef> FieldDef::FromFieldDefPb(const FieldDefPb& field_pb,
                                                  absl::string_view lang_name) {
  FieldDef field;
  field.name_ = Symbol(field_pb.name());

  // Check that the name is in camelCase.
  if (field.name().ToCamelCase() != field_pb.name()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Field '", field_pb.name(), "' is not in camelCase."));
  }

  auto field_error = [&](absl::string_view message) {
    return absl::InvalidArgumentError(
        absl::StrCat("Field '", field_pb.name(), "': ", message));
  };

  // A field with `alternatives` without `type` has the variant of their types.
  // Their IR settings are checked against the nodes in
  // AstDef::ResolveAlternatives().
  TypePb type_pb = field_pb.type();
  if (!field_pb.alternatives().empty()) {
    if (field_pb.kind() != FIELD_KIND_UNSPECIFIED) {
      return field_error(
          "kind can't be set with alternatives: each alternative has its own "
          "kind.");
    }
    if (field_pb.has_ir_name() &&
        !absl::c_all_of(field_pb.alternatives(),
                        [](const FieldAlternativePb& alternative_pb) {
                          return alternative_pb.has_op_kind();
                        })) {
      return field_error(
          "ir_name can't be combined with alternatives, except with op_kind "
          "ones: set the ir_name of each alternative.");
    }

    size_t num_ir_names = 0;
    size_t num_selectors = 0;
    size_t num_op_kinds = 0;
    size_t num_members = 0;
    bool has_typeless = false;
    for (const FieldAlternativePb& alternative_pb : field_pb.alternatives()) {
      Alternative alternative;
      if (alternative_pb.has_type()) {
        const NonListTypePb& alt_type = alternative_pb.type();
        if (alt_type.has_class_()) {
          alternative.member_names.push_back(alt_type.class_());
        } else if (alt_type.has_variant()) {
          for (const ScalarTypePb& member : alt_type.variant().types()) {
            if (!member.has_class_()) {
              return field_error(
                  "Each alternative's type must be a class or a variant of "
                  "classes.");
            }
            alternative.member_names.push_back(member.class_());
          }
          if (alternative.member_names.empty()) {
            return field_error("An alternative has an empty variant type.");
          }
        } else {
          return field_error(
              "Each alternative's type must be a class or a variant of "
              "classes.");
        }
        for (const std::string& member_name : alternative.member_names) {
          *type_pb.mutable_variant()->add_types()->mutable_class_() =
              member_name;
        }
        num_members += alternative.member_names.size();
      } else {
        has_typeless = true;
        ++num_members;
      }
      if (alternative_pb.has_ir_name()) {
        if (alternative_pb.ir_name().empty()) {
          return field_error(absl::StrCat("Alternative ",
                                          alternative.DebugName(),
                                          " has an empty ir_name."));
        }
        alternative.ir_name = alternative_pb.ir_name();
        ++num_ir_names;
      }
      if (alternative_pb.has_kind()) {
        alternative.kind = alternative_pb.kind();
      }
      if (alternative_pb.has_when() && alternative_pb.has_unless()) {
        return field_error(absl::StrCat("Alternative ", alternative.DebugName(),
                                        " can't have both when and unless."));
      }
      if (alternative_pb.has_when()) {
        alternative.when = alternative_pb.when();
      }
      if (alternative_pb.has_unless()) {
        alternative.unless = alternative_pb.unless();
      }
      if (alternative.when.has_value() || alternative.unless.has_value()) {
        ++num_selectors;
      }
      if (alternative_pb.has_op_kind()) {
        alternative.op_kind = alternative_pb.op_kind();
        ++num_op_kinds;
      }
      field.alternatives_.push_back(std::move(alternative));
    }

    size_t num_alternatives = field.alternatives_.size();
    if (num_members < 2) {
      return field_error("alternatives requires at least two alternatives.");
    }
    if (num_ir_names != 0 && num_ir_names != num_alternatives) {
      return field_error(
          "Either all alternatives must have an ir_name or none.");
    }
    if (num_selectors != 0 && num_op_kinds != 0) {
      return field_error(
          "Alternatives can't mix when/unless with op_kind selectors.");
    }
    bool has_selectors = num_selectors != 0 || num_op_kinds != 0;
    if (has_selectors && num_selectors + num_op_kinds != num_alternatives) {
      return field_error(
          "Either all alternatives must have a selector (when, unless or "
          "op_kind) or none.");
    }
    if (!has_selectors && field_pb.has_type()) {
      return field_error(
          "Exactly one of type and alternatives can be set, unless the "
          "alternatives have selectors (when, unless or op_kind).");
    }
    if (has_typeless && !field_pb.has_type()) {
      return field_error(
          "An alternative without a type holds the field's type, so the field "
          "must have a type.");
    }
    if (field_pb.has_type()) {
      type_pb = field_pb.type();
    }
  }

  ABSL_ASSIGN_OR_RETURN(field.type_, FromTypePb(type_pb, lang_name));

  if (field_pb.optionalness() == OPTIONALNESS_UNSPECIFIED) {
    return absl::InvalidArgumentError(
        absl::StrCat("Field '", field_pb.name(),
                     "' has OPTIONALNESS_UNSPECIFIED. This should be a bug, as "
                     "the default value is already OPTIONALNESS_REQUIRED."));
  }
  field.optionalness_ = field_pb.optionalness();

  field.kind_ = field_pb.kind();
  if (field_pb.generation_target() == GENERATION_TARGET_UNSPECIFIED) {
    field.generation_target_ = GENERATION_TARGET_BOTH;
  } else {
    field.generation_target_ = field_pb.generation_target();
  }
  field.ir_attr_optional_parameter_ = field_pb.ir_attr_optional_parameter();
  field.enclose_in_region_ = field_pb.enclose_in_region();

  if (field_pb.has_ir_name()) {
    if (field_pb.ir_name().empty()) {
      return field_error("ir_name can't be empty.");
    }
    if (!field.in_ast() || !field.in_ir()) {
      return field_error(
          "ir_name requires a field that is both in the AST and in the IR.");
    }
    field.ir_name_ = field_pb.ir_name();
  }

  if (!field.alternatives_.empty() && !field.in_ir()) {
    return field_error(
        "alternatives can't be set on a field that isn't in the IR "
        "(GENERATION_TARGET_AST_ONLY): use type.");
  }

  return field;
}

std::string FieldDef::Alternative::DebugName() const {
  if (member_names.empty()) {
    return "<field type>";
  }
  return absl::StrCat("\"", absl::StrJoin(member_names, " | "), "\"");
}

std::optional<Symbol> NodeDef::ir_op_name(absl::string_view lang_name,
                                          FieldKind kind) const {
  // If there's a custom IR op name, return it.
  if (ir_op_name_.has_value()) {
    return Symbol(*ir_op_name_);
  }

  // If any descendent has a custom IR op name, then we fallback to mlir::Value.
  if (absl::c_any_of(descendants(), [](const NodeDef* descendent) {
        return descendent->ir_op_name_.has_value();
      })) {
    return std::nullopt;
  }

  auto ir_name = absl::StrCat(lang_name, has_control_flow() ? "hir" : "ir");

  Symbol result{ir_name};

  result += name();

  switch (kind) {
    case FIELD_KIND_UNSPECIFIED:
      LOG(FATAL) << "Invalid FieldKind.";
    case FIELD_KIND_ATTR:
      result += "Attr";
      break;
    case FIELD_KIND_RVAL:
    case FIELD_KIND_STMT:
      result += "Op";
      break;
    case FIELD_KIND_LVAL:
      result += "RefOp";
      break;
  }

  if (!children().empty()) {
    result += "Interface";
  }

  return result;
}

std::optional<Symbol> NodeDef::ir_op_mnemonic(FieldKind kind) const {
  // If there's a custom IR op name, give up (we won't need mnemonic since we
  // won't generate an op).
  if (ir_op_name_.has_value()) {
    return std::nullopt;
  }

  // If any descendent has a custom IR op name, then we fallback to mlir::Value.
  if (absl::c_any_of(descendants(), [](const NodeDef* descendent) {
        return descendent->ir_op_name_.has_value();
      })) {
    return std::nullopt;
  }

  switch (kind) {
    case FIELD_KIND_UNSPECIFIED:
      LOG(FATAL) << "Invalid FieldKind.";
    case FIELD_KIND_ATTR:
      LOG(FATAL) << "Unsupported FieldKind: " << kind;
    case FIELD_KIND_LVAL:
      return Symbol(name()) + "ref";
    case FIELD_KIND_RVAL:
    case FIELD_KIND_STMT:
      return Symbol(name());
  }
}

Symbol NodeDef::ir_attr_name(absl::string_view lang_name) const {
  return Symbol(absl::StrCat(lang_name, "ir")) + name() + "Attr";
}

Symbol NodeDef::ir_attr_mnemonic() const { return Symbol(name()); }

/*static*/
absl::StatusOr<AstDef> AstDef::FromProto(const AstDefPb& pb) {
  std::vector<EnumDef> enum_defs;
  for (const EnumDefPb& enum_def_pb : pb.enums()) {
    ABSL_ASSIGN_OR_RETURN(EnumDef enum_def,
                          EnumDef::FromEnumDefPb(enum_def_pb));
    enum_defs.push_back(std::move(enum_def));
  }

  std::vector<std::string> node_names;
  absl::flat_hash_map<std::string, std::unique_ptr<NodeDef>> nodes;

  for (const NodeDefPb& node_pb : pb.nodes()) {
    if (nodes.contains(node_pb.name())) {
      return absl::InvalidArgumentError(
          absl::StrCat(node_pb.name(), " already exists!"));
    }

    auto node = absl::WrapUnique(new NodeDef());

    node->name_ = node_pb.name();

    if (node_pb.has_type()) {
      node->type_ = node_pb.type();
    }

    for (const FieldDefPb& field_pb : node_pb.fields()) {
      ABSL_ASSIGN_OR_RETURN(FieldDef field,
                            FieldDef::FromFieldDefPb(field_pb, pb.lang_name()));
      node->fields_.push_back(std::move(field));
    }

    node->has_control_flow_ = node_pb.has_control_flow();

    if (node_pb.has_ir_op_name()) {
      node->ir_op_name_ = node_pb.ir_op_name();
    }

    node->should_generate_ir_op_ = node_pb.should_generate_ir_op();
    node->should_generate_ir_attr_ = node_pb.should_generate_ir_attr();
    node->ir_attr_has_loc_ = node_pb.ir_attr_has_loc();
    if (node_pb.generation_target() == GENERATION_TARGET_UNSPECIFIED) {
      node->generation_target_ = GENERATION_TARGET_BOTH;
    } else {
      node->generation_target_ = node_pb.generation_target();
    }
    node->ir_attr_is_location_ = node_pb.ir_attr_is_location();
    node->is_plain_object_ = node_pb.is_plain_object();

    // `ir_attr_has_loc` is sugar for an IR-only `loc` field holding the node's
    // trivia. Desugaring it here rather than special-casing it in each IR
    // printer keeps `loc` an ordinary field: it participates in the node
    // dependency graph (so `Trivia` is emitted before its users) and flows
    // through the shared argument-printing logic.
    if (node_pb.ir_attr_has_loc()) {
      FieldDefPb loc_pb;
      loc_pb.set_name("loc");
      loc_pb.set_optionalness(OPTIONALNESS_MAYBE_UNDEFINED);
      loc_pb.mutable_type()->set_class_(kTriviaNodeName);
      loc_pb.set_generation_target(GENERATION_TARGET_IR_ONLY);
      loc_pb.set_ir_attr_optional_parameter(true);

      ABSL_ASSIGN_OR_RETURN(FieldDef loc_field,
                            FieldDef::FromFieldDefPb(loc_pb, pb.lang_name()));

      // The trivia is conventionally the first attribute parameter.
      node->fields_.insert(node->fields_.begin(), std::move(loc_field));
    }

    for (const auto& trait : node_pb.additional_attr_traits()) {
      node->additional_attr_traits_.push_back(trait);
    }

    node->has_fold_ = node_pb.has_fold();

    for (auto kind : node_pb.kinds()) {
      node->kinds_.push_back(static_cast<FieldKind>(kind));
    }

    for (auto mlir_trait : node_pb.additional_mlir_traits()) {
      node->additional_mlir_traits_.push_back(
          static_cast<MlirTrait>(mlir_trait));
    }

    for (auto mlir_trait : node_pb.excluded_mlir_traits()) {
      node->excluded_mlir_traits_.push_back(static_cast<MlirTrait>(mlir_trait));
    }

    if (!node_pb.fold_kinds().empty() && !node_pb.has_fold()) {
      return absl::InvalidArgumentError(
          absl::StrCat(node_pb.name(), ": fold_kinds requires has_fold."));
    }
    for (auto kind : node_pb.fold_kinds()) {
      node->fold_kinds_.push_back(static_cast<FieldKind>(kind));
    }

    node_names.push_back(node_pb.name());
    nodes.emplace(node_pb.name(), std::move(node));
  }

  // Set parent pointers.
  for (const NodeDefPb& node_pb : pb.nodes()) {
    NodeDef& node = *nodes.at(node_pb.name());

    for (absl::string_view parent_name : node_pb.parents()) {
      auto it = nodes.find(parent_name);
      if (it == nodes.end()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Parent ", parent_name, " doesn't exist!"));
      }
      NodeDef* parent = it->second.get();
      node.parents_.push_back(parent);
    }
  }

  // For union types, create a node to represent each one and add that node as
  // a parent of the specified types.
  for (const UnionTypePb& union_type_pb : pb.union_types()) {
    auto union_type_node = absl::WrapUnique(new NodeDef());
    union_type_node->name_ = union_type_pb.name();
    union_type_node->should_generate_ir_op_ =
        union_type_pb.should_generate_ir_op();
    for (auto kind : union_type_pb.kinds()) {
      union_type_node->kinds_.push_back(static_cast<FieldKind>(kind));
    }
    if (union_type_pb.has_ir_op_name()) {
      union_type_node->ir_op_name_ = union_type_pb.ir_op_name();
    }
    if (union_type_pb.has_should_generate_dispatch()) {
      union_type_node->should_generate_dispatch_ =
          union_type_pb.should_generate_dispatch();
    }
    for (const auto& o : union_type_pb.dispatch_overrides()) {
      std::optional<std::string> ir_op_name;
      if (o.has_ir_op_name()) {
        ir_op_name = o.ir_op_name();
      }
      union_type_node->dispatch_overrides_.emplace(
          o.type(), NodeDef::DispatchOverride{o.visitor(), ir_op_name});
    }
    for (const auto& s : union_type_pb.dispatch_skip()) {
      union_type_node->dispatch_skip_.insert(s);
    }
    if (nodes.contains(union_type_pb.name())) {
      return absl::InvalidArgumentError(
          absl::StrCat(union_type_pb.name(), " already exists!"));
    }
    node_names.push_back(union_type_pb.name());
    nodes.emplace(union_type_pb.name(), std::move(union_type_node));
  }

  for (const UnionTypePb& union_type_pb : pb.union_types()) {
    auto union_type_node = nodes.at(union_type_pb.name()).get();
    for (const std::string& type : union_type_pb.types()) {
      auto child_node = nodes.find(type);
      if (child_node == nodes.end()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Union type ", union_type_pb.name(), ": member ", type,
                         " doesn't exist!"));
      }
      child_node->second->parents_.push_back(union_type_node);
    }

    for (absl::string_view parent_name : union_type_pb.parents()) {
      auto it = nodes.find(parent_name);
      if (it == nodes.end()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Parent ", parent_name, " doesn't exist!"));
      }
      NodeDef* parent = it->second.get();
      union_type_node->parents_.push_back(parent);
    }
  }

  // NOTE: In the code below, we traverse `node_names` instead of `nodes`.
  // `node_names` preserves the original order of definitions.
  // This makes sure that the algorithm is always deterministic.

  // Set ancestors vector.
  for (const std::string& name : node_names) {
    NodeDef& node = *nodes.at(name);

    node.ancestors_ = TopologicalSortDependencies(
        &node, [](NodeDef* node) { return node->parents_; });
  }

  // Set aggregated_fields vector.
  for (const std::string& name : node_names) {
    NodeDef& node = *nodes.at(name);

    for (NodeDef* ancestor : node.ancestors_) {
      for (FieldDef& field : ancestor->fields_) {
        node.aggregated_fields_.push_back(&field);
      }
    }
    for (FieldDef& field : node.fields_) {
      node.aggregated_fields_.push_back(&field);
    }
  }

  // Set children vector.
  for (const std::string& name : node_names) {
    NodeDef& node = *nodes.at(name);

    for (NodeDef* parent : node.parents_) {
      parent->children_.push_back(&node);
    }
  }

  // Set descendants vector.
  for (const std::string& name : node_names) {
    NodeDef& node = *nodes.at(name);

    node.descendants_ = TopologicalSortDependencies(
        &node, [](NodeDef* node) { return node->children_; });
  }

  // Set leaves vector.
  for (const std::string& name : node_names) {
    NodeDef& node = *nodes.at(name);

    for (NodeDef* descendent : node.descendants_) {
      if (!descendent->children().empty()) {
        continue;
      }
      node.leaves_.push_back(descendent);
    }
  }

  // Set aggregated_kinds vector.
  for (const std::string& name : node_names) {
    NodeDef& node = *nodes.at(name);

    absl::btree_set<FieldKind> aggregated_kinds;

    for (NodeDef* ancestor : node.ancestors_) {
      for (FieldKind kind : ancestor->kinds_) {
        aggregated_kinds.insert(kind);
      }
    }
    for (FieldKind kind : node.kinds_) {
      aggregated_kinds.insert(kind);
    }

    node.aggregated_kinds_ = {aggregated_kinds.begin(), aggregated_kinds.end()};
  }

  // Set the aggregated_additional_mlir_traits vector.
  for (const std::string& name : node_names) {
    NodeDef& node = *nodes.at(name);

    absl::btree_set<MlirTrait> aggregated_additional_mlir_traits;

    for (NodeDef* ancestor : node.ancestors_) {
      for (MlirTrait trait : ancestor->additional_mlir_traits()) {
        aggregated_additional_mlir_traits.insert(trait);
      }
    }
    for (MlirTrait trait : node.additional_mlir_traits()) {
      aggregated_additional_mlir_traits.insert(trait);
    }
    for (MlirTrait trait : node.excluded_mlir_traits_) {
      aggregated_additional_mlir_traits.erase(trait);
    }

    node.aggregated_additional_mlir_traits_ = {
        aggregated_additional_mlir_traits.begin(),
        aggregated_additional_mlir_traits.end(),
    };
  }

  // Reorder the node definitions so that dependencies always come first.
  std::vector<NodeDef*> topological_sorted_nodes;
  absl::flat_hash_set<NodeDef*> preorder_visited_nodes;
  for (const std::string& name : node_names) {
    NodeDef& node = *nodes.at(name);

    TopologicalSortDependencies(
        &node,
        [&nodes](NodeDef* node) {
          std::vector<NodeDef*> dependencies;
          dependencies.insert(dependencies.end(), node->parents_.begin(),
                              node->parents_.end());
          for (const FieldDef& field : node->fields()) {
            // The topological order is shared by the AST and the IR printers,
            // so a field contributes a dependency edge if it appears in either
            // representation. In particular, an IR-only field still requires
            // its type to be defined before the node that uses it.
            if (!field.in_ast() && !field.in_ir()) {
              continue;
            }
            GetDependencies(field.type(), nodes, &dependencies);
          }
          return dependencies;
        },
        &preorder_visited_nodes, &topological_sorted_nodes);
    if (!absl::c_linear_search(topological_sorted_nodes, &node)) {
      topological_sorted_nodes.push_back(&node);
    }
  }

  // For each root node, add an enum field to represent the leaf type.
  for (NodeDef* node : topological_sorted_nodes) {
    if (!node->parents().empty()) {
      continue;
    }
    if (node->children().empty()) {
      continue;
    }

    std::vector<EnumMemberDef> type_enum_members;
    for (const NodeDef* leaf : node->leaves()) {
      EnumMemberDef member{Symbol(leaf->name()), leaf->name()};
      type_enum_members.push_back(std::move(member));
    }

    node->node_type_enum_ = EnumDef{
        Symbol{node->name()} + "Type",
        std::move(type_enum_members),
    };
  }

  // For each ClassType, if it resolves to a NodeDef, store a reference to it.
  for (NodeDef* node : topological_sorted_nodes) {
    for (FieldDef& field : node->fields_) {
      ResolveClassType(field.type(), topological_sorted_nodes);
    }
  }

  for (const std::string& name : node_names) {
    NodeDef& node = *nodes.at(name);
    for (FieldDef& field : node.fields_) {
      ABSL_RETURN_IF_ERROR(ResolveAlternatives(nodes, node, field));
    }
  }

  return AstDef{pb.lang_name(), std::move(enum_defs), std::move(node_names),
                std::move(nodes), std::move(topological_sorted_nodes)};
}

namespace {

// The class names of a class type, or of the members of a variant of classes.
// nullopt for any other type.
std::optional<std::vector<std::string>> ClassNames(const Type& type) {
  if (type.IsA<ClassType>()) {
    return std::vector<std::string>{
        static_cast<const ClassType&>(type).name().ToPascalCase()};
  }
  if (!type.IsA<VariantType>()) {
    return std::nullopt;
  }
  std::vector<std::string> names;
  for (const auto& member : static_cast<const VariantType&>(type).types()) {
    if (!member->IsA<ClassType>()) {
      return std::nullopt;
    }
    names.push_back(
        static_cast<const ClassType&>(*member).name().ToPascalCase());
  }
  return names;
}

bool SupportsKind(const NodeDef& node, FieldKind kind) {
  return absl::c_linear_search(node.aggregated_kinds(), kind);
}

// The leaf nodes that are `node` or descend from it.
absl::flat_hash_set<const NodeDef*> LeavesOf(const NodeDef& node) {
  if (node.leaves().empty()) {
    return {&node};
  }
  return {node.leaves().begin(), node.leaves().end()};
}

}  // namespace

absl::Status AstDef::ResolveAlternatives(
    const absl::flat_hash_map<std::string, std::unique_ptr<NodeDef>>& nodes,
    NodeDef& node, FieldDef& field) {
  if (field.alternatives_.empty()) {
    return absl::OkStatus();
  }

  auto error = [&](absl::string_view message) {
    return absl::InvalidArgumentError(absl::StrCat(
        node.name(), "::", field.name().ToCcVarName(), ": ", message));
  };

  std::optional<std::vector<std::string>> field_members =
      ClassNames(field.type());
  if (!field_members.has_value()) {
    return error(
        "A field with alternatives must have a class type or a variant of "
        "classes.");
  }

  // The class names an alternative holds: its own, or the field's.
  auto names_of = [&](const FieldDef::Alternative& alternative)
      -> absl::Span<const std::string> {
    if (alternative.member_names.empty()) {
      return *field_members;
    }
    return alternative.member_names;
  };

  // The kind of each alternative: its `kind` if set, otherwise the one of
  // FIELD_KIND_STMT, FIELD_KIND_RVAL and FIELD_KIND_LVAL all its types
  // support.
  bool has_sibling_selectors = false;
  bool has_op_kinds = false;
  std::vector<FieldKind> kinds;
  for (const FieldDef::Alternative& alternative : field.alternatives_) {
    has_sibling_selectors |=
        alternative.when.has_value() || alternative.unless.has_value();
    has_op_kinds |= alternative.op_kind.has_value();

    std::vector<const NodeDef*> members;
    for (const std::string& member_name : names_of(alternative)) {
      auto it = nodes.find(member_name);
      if (it == nodes.end()) {
        return error(absl::StrCat("Alternative \"", member_name,
                                  "\" doesn't name a node."));
      }
      members.push_back(it->second.get());
    }

    if (alternative.kind.has_value()) {
      FieldKind kind = *alternative.kind;
      if (kind == FIELD_KIND_ATTR) {
        // Validated by ResolveSiblingAlternatives().
        if (!alternative.unless.has_value()) {
          return error(absl::StrCat(
              "Alternative ", alternative.DebugName(),
              " has kind FIELD_KIND_ATTR, which is only supported for "
              "unless alternatives."));
        }
        kinds.push_back(kind);
        continue;
      }
      if (kind != FIELD_KIND_STMT && kind != FIELD_KIND_RVAL &&
          kind != FIELD_KIND_LVAL) {
        return error(absl::StrCat(
            "Alternative ", alternative.DebugName(), " has kind ",
            FieldKind_Name(kind),
            ", but must be FIELD_KIND_ATTR, FIELD_KIND_STMT, FIELD_KIND_RVAL "
            "or FIELD_KIND_LVAL."));
      }
      for (const NodeDef* member : members) {
        if (!SupportsKind(*member, kind)) {
          return error(absl::StrCat("Alternative \"", member->name(),
                                    "\" doesn't support kind ",
                                    FieldKind_Name(kind), "."));
        }
      }
      kinds.push_back(kind);
      continue;
    }

    std::optional<FieldKind> inferred;
    for (const NodeDef* member : members) {
      std::vector<FieldKind> supported;
      for (FieldKind kind :
           {FIELD_KIND_STMT, FIELD_KIND_RVAL, FIELD_KIND_LVAL}) {
        if (SupportsKind(*member, kind)) {
          supported.push_back(kind);
        }
      }
      if (supported.size() != 1) {
        std::vector<std::string> names;
        for (FieldKind kind : supported) {
          names.push_back(FieldKind_Name(kind));
        }
        return error(absl::StrCat(
            "Alternative \"", member->name(), "\" supports ",
            names.empty() ? "none" : absl::StrJoin(names, ", "),
            " of FIELD_KIND_STMT, FIELD_KIND_RVAL and FIELD_KIND_LVAL: set "
            "its kind."));
      }
      if (inferred.has_value() && *inferred != supported.front()) {
        return error(absl::StrCat("The types of alternative ",
                                  alternative.DebugName(),
                                  " support different kinds: set its kind."));
      }
      inferred = supported.front();
    }
    kinds.push_back(*inferred);
  }

  if (has_sibling_selectors) {
    return ResolveSiblingAlternatives(nodes, node, field, kinds);
  }
  if (has_op_kinds) {
    return ResolveOpKindAlternatives(nodes, node, field, kinds);
  }

  // FieldDef::FromFieldDefPb() checked that either all alternatives have an
  // ir_name or none does.
  if (!field.alternatives_.front().ir_name.has_value()) {
    // Without ir_name, the alternatives are held in the field's operand, as
    // rvalues or lvalues. This doesn't require the node to have
    // should_generate_ir_op: it also describes handwritten ops.
    for (size_t i = 0; i != kinds.size(); ++i) {
      if (kinds[i] == FIELD_KIND_STMT) {
        return error(absl::StrCat(
            "Alternative ", field.alternatives_[i].DebugName(),
            " is FIELD_KIND_STMT, which requires ir_names (a statement is "
            "held in its own region)."));
      }
    }
    if (static_cast<size_t>(absl::c_count(kinds, kinds.front())) ==
        kinds.size()) {
      return error(absl::StrCat(
          "All alternatives are ", FieldKind_Name(kinds.front()),
          " and have no ir_name: use type and kind instead of alternatives."));
    }
    // The field is held as an rvalue, except for its lvalue alternatives.
    field.kind_ = FIELD_KIND_RVAL;
    for (size_t i = 0; i != kinds.size(); ++i) {
      if (kinds[i] == FIELD_KIND_LVAL) {
        for (const std::string& member_name :
             field.alternatives_[i].member_names) {
          field.member_ir_kinds_[member_name] = FIELD_KIND_LVAL;
        }
      }
    }
    return absl::OkStatus();
  }

  if (!node.should_generate_ir_op()) {
    return error(
        "Alternatives with ir_names require the node to have "
        "should_generate_ir_op.");
  }
  for (size_t i = 0; i != kinds.size(); ++i) {
    const FieldDef::Alternative& alternative = field.alternatives_[i];
    FieldKind kind = kinds[i];
    // A statement is always held in a region.
    bool is_region = kind == FIELD_KIND_STMT || field.enclose_in_region();

    auto it =
        absl::c_find_if(field.ir_splits_, [&](const FieldDef::IrSplit& split) {
          return split.name == *alternative.ir_name;
        });
    if (it == field.ir_splits_.end()) {
      field.ir_splits_.push_back(FieldDef::IrSplit{
          .name = *alternative.ir_name,
          .is_region = is_region,
          .kind = kind,
          .types = alternative.member_names,
      });
      continue;
    }
    if (it->kind != kind || it->is_region != is_region) {
      return error(absl::StrCat(
          "The alternatives with ir_name \"", *alternative.ir_name,
          "\" must have the same kind and all be held in a region or all in "
          "an operand, but \"",
          it->types.front(), "\" is ", FieldKind_Name(it->kind),
          it->is_region ? " (region)" : " (operand)", " and ",
          alternative.DebugName(), " is ", FieldKind_Name(kind),
          is_region ? " (region)." : " (operand)."));
    }
    it->types.insert(it->types.end(), alternative.member_names.begin(),
                     alternative.member_names.end());
  }
  return absl::OkStatus();
}

absl::Status AstDef::ResolveSiblingAlternatives(
    const absl::flat_hash_map<std::string, std::unique_ptr<NodeDef>>& nodes,
    NodeDef& node, FieldDef& field, absl::Span<const FieldKind> kinds) {
  auto error = [&](absl::string_view message) {
    return absl::InvalidArgumentError(absl::StrCat(
        node.name(), "::", field.name().ToCcVarName(), ": ", message));
  };

  std::string sibling_name;
  std::optional<size_t> when_index;
  std::vector<size_t> unless_indices;
  for (size_t i = 0; i != field.alternatives_.size(); ++i) {
    const FieldDef::Alternative& alternative = field.alternatives_[i];
    const std::string& name =
        alternative.when.has_value() ? *alternative.when : *alternative.unless;
    if (sibling_name.empty()) {
      sibling_name = name;
    } else if (name != sibling_name) {
      return error(
          absl::StrCat("All when/unless must name the same sibling, "
                       "but got \"",
                       sibling_name, "\" and \"", name, "\"."));
    }
    if (!alternative.ir_name.has_value()) {
      return error("Alternatives with when/unless require ir_names.");
    }
    if (alternative.when.has_value()) {
      if (when_index.has_value()) {
        return error("Exactly one alternative must have when.");
      }
      when_index = i;
    } else {
      unless_indices.push_back(i);
    }
  }
  if (!when_index.has_value()) {
    return error("Exactly one alternative must have when.");
  }
  if (unless_indices.empty()) {
    return error("At least one alternative must have unless.");
  }

  // The `when` alternative: the field's whole type, in an operand.
  const FieldDef::Alternative& when_alternative =
      field.alternatives_[*when_index];
  if (!when_alternative.member_names.empty()) {
    return error(
        "The when alternative holds the field's whole type: it can't have a "
        "type.");
  }
  FieldKind when_kind = kinds[*when_index];
  if (when_kind != FIELD_KIND_RVAL && when_kind != FIELD_KIND_LVAL) {
    return error(
        absl::StrCat("The when alternative must be FIELD_KIND_RVAL or "
                     "FIELD_KIND_LVAL, but is ",
                     FieldKind_Name(when_kind), "."));
  }

  // The `unless` alternatives: attributes of some of the field's types.
  const std::string& literal_attr =
      *field.alternatives_[unless_indices.front()].ir_name;
  if (literal_attr == *when_alternative.ir_name) {
    return error(
        absl::StrCat("The when and unless alternatives must have "
                     "different ir_names, but both are \"",
                     literal_attr, "\"."));
  }
  std::vector<std::string> field_members = *ClassNames(field.type());
  std::vector<std::string> literal_attr_types;
  for (size_t i : unless_indices) {
    const FieldDef::Alternative& alternative = field.alternatives_[i];
    if (kinds[i] != FIELD_KIND_ATTR) {
      return error(absl::StrCat("The unless alternative ",
                                alternative.DebugName(),
                                " must have kind FIELD_KIND_ATTR."));
    }
    if (*alternative.ir_name != literal_attr) {
      return error(absl::StrCat(
          "The unless alternatives must share an ir_name, but got \"",
          literal_attr, "\" and \"", *alternative.ir_name, "\"."));
    }
    if (alternative.member_names.empty()) {
      return error("An unless alternative must have a type.");
    }
    for (const std::string& member_name : alternative.member_names) {
      const NodeDef& member = *nodes.at(member_name);
      if (!member.should_generate_ir_attr()) {
        return error(absl::StrCat("The unless alternative \"", member_name,
                                  "\" must have should_generate_ir_attr."));
      }
      // The type must be held by a member of the field's type.
      bool held = absl::c_any_of(field_members, [&](const std::string& name) {
        return name == member_name ||
               absl::c_any_of(member.ancestors(), [&](const NodeDef* ancestor) {
                 return ancestor->name() == name;
               });
      });
      if (!held) {
        return error(absl::StrCat("The unless alternative \"", member_name,
                                  "\" isn't a type of the field."));
      }
      literal_attr_types.push_back(member_name);
    }
  }

  if (field.optionalness() != OPTIONALNESS_REQUIRED) {
    return error("Alternatives with when/unless require a required field.");
  }
  if (field.enclose_in_region()) {
    return error(
        "Alternatives with when/unless can't be combined with "
        "enclose_in_region.");
  }

  FieldDef* sibling = nullptr;
  for (FieldDef& other : node.fields_) {
    if (other.name().ToCamelCase() == sibling_name) {
      sibling = &other;
      break;
    }
  }
  if (sibling == nullptr) {
    return error(absl::StrCat("when/unless \"", sibling_name,
                              "\" doesn't name a sibling field."));
  }
  if (sibling == &field || !sibling->type().IsA<BuiltinType>() ||
      static_cast<const BuiltinType&>(sibling->type()).builtin_kind() !=
          BuiltinTypeKind::kBool ||
      sibling->optionalness() != OPTIONALNESS_REQUIRED) {
    return error(absl::StrCat("when/unless \"", sibling_name,
                              "\" must name a required bool field."));
  }
  sibling->derived_in_ir_ = true;

  field.kind_ = when_kind;
  field.ir_name_ = *when_alternative.ir_name;
  field.literal_attr_ = FieldDef::LiteralAttr{
      .name = literal_attr,
      .types = std::move(literal_attr_types),
      .discriminator = sibling_name,
  };
  return absl::OkStatus();
}

absl::Status AstDef::ResolveOpKindAlternatives(
    const absl::flat_hash_map<std::string, std::unique_ptr<NodeDef>>& nodes,
    const NodeDef& node, FieldDef& field, absl::Span<const FieldKind> kinds) {
  auto error = [&](absl::string_view message) {
    return absl::InvalidArgumentError(absl::StrCat(
        node.name(), "::", field.name().ToCcVarName(), ": ", message));
  };

  std::optional<size_t> rval_index;
  std::optional<size_t> lval_index;
  for (size_t i = 0; i != field.alternatives_.size(); ++i) {
    const FieldDef::Alternative& alternative = field.alternatives_[i];
    FieldKind op_kind = *alternative.op_kind;
    if (op_kind != FIELD_KIND_RVAL && op_kind != FIELD_KIND_LVAL) {
      return error(
          absl::StrCat("op_kind must be FIELD_KIND_RVAL or "
                       "FIELD_KIND_LVAL, but is ",
                       FieldKind_Name(op_kind), "."));
    }
    std::optional<size_t>& index =
        op_kind == FIELD_KIND_RVAL ? rval_index : lval_index;
    if (index.has_value()) {
      return error(absl::StrCat("Exactly one alternative must have op_kind ",
                                FieldKind_Name(op_kind), "."));
    }
    index = i;
    if (!SupportsKind(node, op_kind)) {
      return error(absl::StrCat("op_kind ", FieldKind_Name(op_kind),
                                " requires the node to support that kind."));
    }
    if (alternative.ir_name.has_value()) {
      return error(
          "Alternatives with op_kind can't have an ir_name: set the field's "
          "ir_name.");
    }
    if (kinds[i] != FIELD_KIND_RVAL && kinds[i] != FIELD_KIND_LVAL) {
      return error(
          absl::StrCat("Alternative ", alternative.DebugName(),
                       " must be FIELD_KIND_RVAL or FIELD_KIND_LVAL, but is ",
                       FieldKind_Name(kinds[i]), "."));
    }
  }
  if (!rval_index.has_value() || !lval_index.has_value()) {
    return error(
        "Alternatives with op_kind require exactly one for each of "
        "FIELD_KIND_RVAL and FIELD_KIND_LVAL.");
  }

  std::vector<std::string> field_members = *ClassNames(field.type());
  bool is_variant = field.type().IsA<VariantType>();

  // The members the op holds for a field held with `kind`: for a variant
  // field, only the members that support the kind (see NarrowedIrType()).
  auto implicit_members = [&](FieldKind kind) -> std::vector<std::string> {
    if (!is_variant) {
      return field_members;
    }
    std::vector<std::string> supported;
    for (const std::string& name : field_members) {
      if (SupportsKind(*nodes.at(name), kind)) {
        supported.push_back(name);
      }
    }
    return supported;
  };

  bool narrowed = false;
  // Checks that the alternative of `op_kind` holds what the op of that kind
  // holds by default. Returns its types.
  auto check_implicit = [&](size_t index, FieldKind op_kind) -> absl::Status {
    const FieldDef::Alternative& alternative = field.alternatives_[index];
    std::vector<std::string> expected = implicit_members(kinds[index]);
    if (expected.empty() ||
        (expected.size() != 1 && expected.size() != field_members.size())) {
      return error(absl::StrCat(
          "In the ", FieldKind_Name(op_kind),
          " op, either one or all members of the field must support ",
          FieldKind_Name(kinds[index]), "."));
    }
    std::vector<std::string> actual = alternative.member_names.empty()
                                          ? field_members
                                          : alternative.member_names;
    if (actual != expected) {
      return error(absl::StrCat("The ", FieldKind_Name(op_kind),
                                " alternative ", alternative.DebugName(),
                                " must hold \"", absl::StrJoin(expected, " | "),
                                "\": the members of the field that "
                                "support ",
                                FieldKind_Name(kinds[index]), "."));
    }
    narrowed |= expected.size() != field_members.size();
    return absl::OkStatus();
  };

  ABSL_RETURN_IF_ERROR(check_implicit(*rval_index, FIELD_KIND_RVAL));
  field.kind_ = kinds[*rval_index];

  FieldKind lval_kind = kinds[*lval_index];
  // The class the LVAL op holds, for a class field holding another class.
  std::optional<std::string> lval_type;

  const FieldDef::Alternative& lval_alternative =
      field.alternatives_[*lval_index];
  if (is_variant || lval_alternative.member_names.empty() ||
      lval_alternative.member_names == field_members) {
    ABSL_RETURN_IF_ERROR(check_implicit(*lval_index, FIELD_KIND_LVAL));
  } else {
    // A class field holding another class in the LVAL op.
    if (lval_alternative.member_names.size() != 1) {
      return error(absl::StrCat(
          "The FIELD_KIND_LVAL alternative of a class field must have a "
          "single class, but has ",
          lval_alternative.DebugName(), "."));
    }
    const std::string& member = lval_alternative.member_names.front();
    absl::flat_hash_set<const NodeDef*> lval_leaves =
        LeavesOf(*nodes.at(member));
    if (!absl::c_any_of(
            LeavesOf(*nodes.at(field_members.front())),
            [&](const NodeDef* leaf) { return lval_leaves.contains(leaf); })) {
      return error(absl::StrCat("The FIELD_KIND_LVAL alternative \"", member,
                                "\" has no node in common with the field's "
                                "type \"",
                                field_members.front(), "\"."));
    }
    lval_type = member;
    narrowed = true;
  }

  if (narrowed && field.optionalness() != OPTIONALNESS_REQUIRED) {
    return error(
        "An op holding only some of the field's types requires a required "
        "field.");
  }
  if (lval_kind != field.kind_ || lval_type.has_value()) {
    field.lval_op_override_ = FieldDef::LvalOpOverride{
        .kind = lval_kind,
        .type = std::move(lval_type),
    };
  }
  return absl::OkStatus();
}

void AstDef::ResolveClassType(
    Type& type, absl::Span<const NodeDef* const> topological_sorted_nodes) {
  switch (type.kind()) {
    case TypeKind::kBuiltin:
    case TypeKind::kEnum: {
      break;
    }
    case TypeKind::kClass: {
      auto& class_type = static_cast<ClassType&>(type);
      for (const NodeDef* node : topological_sorted_nodes) {
        if (node->name() == class_type.name().ToPascalCase()) {
          LOG(INFO) << "Resolved class " << node->name();
          class_type.node_def_ = node;
          break;
        }
      }
      break;
    }
    case TypeKind::kList: {
      auto& list_type = static_cast<ListType&>(type);
      ResolveClassType(list_type.element_type(), topological_sorted_nodes);
      break;
    }
    case TypeKind::kVariant: {
      auto& variant_type = static_cast<VariantType&>(type);
      for (auto& type : variant_type.types()) {
        ResolveClassType(*type, topological_sorted_nodes);
      }
      break;
    }
  }
}

}  // namespace maldoca
