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

  // A field with `alternatives` has the variant of their types. Their IR
  // settings are checked against the nodes in AstDef::ValidateIrMapping().
  TypePb type_pb = field_pb.type();
  if (!field_pb.alternatives().empty()) {
    if (field_pb.has_type()) {
      return field_error("Exactly one of type and alternatives can be set.");
    }
    if (field_pb.alternatives_size() < 2) {
      return field_error("alternatives requires at least two alternatives.");
    }
    if (field_pb.kind() != FIELD_KIND_UNSPECIFIED) {
      return field_error(
          "kind can't be set with alternatives: each alternative has its own "
          "kind.");
    }
    size_t num_ir_names = 0;
    for (const FieldAlternativePb& alternative_pb : field_pb.alternatives()) {
      if (!alternative_pb.type().has_class_()) {
        return field_error("Each alternative's type must be a class.");
      }
      *type_pb.mutable_variant()->add_types() = alternative_pb.type();
      Alternative alternative{.member_name = alternative_pb.type().class_()};
      if (alternative_pb.has_ir_name()) {
        if (alternative_pb.ir_name().empty()) {
          return field_error(absl::StrCat("Alternative \"",
                                          alternative.member_name,
                                          "\" has an empty ir_name."));
        }
        alternative.ir_name = alternative_pb.ir_name();
        ++num_ir_names;
      }
      if (alternative_pb.has_kind()) {
        alternative.kind = alternative_pb.kind();
      }
      field.alternatives_.push_back(std::move(alternative));
    }
    if (num_ir_names != 0 && num_ir_names != field.alternatives_.size()) {
      return field_error(
          "Either all alternatives must have an ir_name or none.");
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
  if (field_pb.has_ir_mapping()) {
    field.ir_mapping_ = field_pb.ir_mapping();
  }

  if (!field.alternatives_.empty() && !field.in_ir()) {
    return field_error(
        "alternatives can't be set on a field that isn't in the IR "
        "(GENERATION_TARGET_AST_ONLY): use type.");
  }

  return field;
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
      ABSL_RETURN_IF_ERROR(ValidateIrMapping(nodes, node, field));
    }
  }

  return AstDef{pb.lang_name(), std::move(enum_defs), std::move(node_names),
                std::move(nodes), std::move(topological_sorted_nodes)};
}

absl::Status AstDef::ValidateIrMapping(
    const absl::flat_hash_map<std::string, std::unique_ptr<NodeDef>>& nodes,
    NodeDef& node, FieldDef& field) {
  ABSL_RETURN_IF_ERROR(ResolveAlternatives(nodes, node, field));

  const FieldIrMappingPb* mapping = field.ir_mapping();
  if (mapping == nullptr) {
    return absl::OkStatus();
  }

  auto error = [&](absl::string_view message) {
    return absl::InvalidArgumentError(
        absl::StrCat(node.name(), "::", field.name().ToCcVarName(),
                     ": ir_mapping: ", message));
  };

  auto check_node_names = [&](absl::string_view what,
                              const auto& names) -> absl::Status {
    for (const std::string& type_name : names) {
      if (!nodes.contains(type_name)) {
        return error(
            absl::StrCat(what, " \"", type_name, "\" doesn't name a node."));
      }
    }
    return absl::OkStatus();
  };

  if (mapping->has_operand() && mapping->has_region()) {
    return error("At most one of operand and region can be set.");
  }
  if (mapping->has_literal_attr() && !mapping->has_operand()) {
    return error("literal_attr requires operand.");
  }
  if (mapping->has_literal_attr() != !mapping->literal_attr_types().empty()) {
    return error("literal_attr_types must be set iff literal_attr is set.");
  }
  ABSL_RETURN_IF_ERROR(
      check_node_names("literal_attr_types", mapping->literal_attr_types()));
  for (const std::string& type_name : mapping->literal_attr_types()) {
    if (!nodes.at(type_name)->should_generate_ir_attr()) {
      return error(absl::StrCat("literal_attr_types \"", type_name,
                                "\" must have should_generate_ir_attr."));
    }
  }
  ABSL_RETURN_IF_ERROR(
      check_node_names("lval_op_types", mapping->lval_op_types()));

  if (!field.ir_splits().empty() &&
      (mapping->has_operand() || mapping->has_region() ||
       mapping->has_literal_attr() || mapping->has_kind() ||
       mapping->has_lval_op_kind() || !mapping->lval_op_types().empty())) {
    return error(
        "A field split by the ir_names of its alternatives cannot have "
        "operand, region, literal_attr, kind, lval_op_kind or lval_op_types.");
  }
  if (field.has_member_ir_kinds() &&
      (mapping->has_kind() || mapping->has_lval_op_kind() ||
       !mapping->lval_op_types().empty() || mapping->has_literal_attr())) {
    return error(
        "Alternatives of different kinds cannot be combined with kind, "
        "lval_op_kind, lval_op_types or literal_attr.");
  }

  if (mapping->has_literal_unless()) {
    if (!mapping->has_literal_attr() || !mapping->has_operand()) {
      return error("literal_unless requires literal_attr and operand.");
    }
    if (field.optionalness() != OPTIONALNESS_REQUIRED) {
      return error("literal_unless requires a required field.");
    }
    FieldDef* sibling = nullptr;
    for (FieldDef& other : node.fields_) {
      if (other.name().ToCamelCase() == mapping->literal_unless()) {
        sibling = &other;
        break;
      }
    }
    if (sibling == nullptr) {
      return error(absl::StrCat("literal_unless \"", mapping->literal_unless(),
                                "\" doesn't name a sibling field."));
    }
    if (!sibling->type().IsA<BuiltinType>() ||
        static_cast<const BuiltinType&>(sibling->type()).builtin_kind() !=
            BuiltinTypeKind::kBool ||
        sibling->optionalness() != OPTIONALNESS_REQUIRED) {
      return error(absl::StrCat("literal_unless \"", mapping->literal_unless(),
                                "\" must name a required bool field."));
    }
    sibling->derived_in_ir_ = true;
  }

  return absl::OkStatus();
}

absl::Status AstDef::ResolveAlternatives(
    const absl::flat_hash_map<std::string, std::unique_ptr<NodeDef>>& nodes,
    const NodeDef& node, FieldDef& field) {
  if (field.alternatives_.empty()) {
    return absl::OkStatus();
  }

  auto error = [&](absl::string_view message) {
    return absl::InvalidArgumentError(absl::StrCat(
        node.name(), "::", field.name().ToCcVarName(), ": ", message));
  };
  auto supports_kind = [&](const NodeDef& member, FieldKind kind) {
    return absl::c_linear_search(member.aggregated_kinds(), kind);
  };

  // The kind of each alternative: its `kind` if set, otherwise the one of
  // FIELD_KIND_STMT, FIELD_KIND_RVAL and FIELD_KIND_LVAL its node supports.
  std::vector<FieldKind> kinds;
  for (const FieldDef::Alternative& alternative : field.alternatives_) {
    auto it = nodes.find(alternative.member_name);
    if (it == nodes.end()) {
      return error(absl::StrCat("Alternative \"", alternative.member_name,
                                "\" doesn't name a node."));
    }
    const NodeDef& member = *it->second;
    if (alternative.kind.has_value()) {
      FieldKind kind = *alternative.kind;
      if (kind != FIELD_KIND_STMT && kind != FIELD_KIND_RVAL &&
          kind != FIELD_KIND_LVAL) {
        return error(absl::StrCat(
            "Alternative \"", alternative.member_name, "\" has kind ",
            FieldKind_Name(kind),
            ", but must be FIELD_KIND_STMT, FIELD_KIND_RVAL or "
            "FIELD_KIND_LVAL."));
      }
      if (!supports_kind(member, kind)) {
        return error(absl::StrCat("Alternative \"", alternative.member_name,
                                  "\" doesn't support kind ",
                                  FieldKind_Name(kind), "."));
      }
      kinds.push_back(kind);
      continue;
    }
    std::vector<FieldKind> supported;
    for (FieldKind kind :
         {FIELD_KIND_STMT, FIELD_KIND_RVAL, FIELD_KIND_LVAL}) {
      if (supports_kind(member, kind)) {
        supported.push_back(kind);
      }
    }
    if (supported.size() != 1) {
      std::vector<std::string> names;
      for (FieldKind kind : supported) {
        names.push_back(FieldKind_Name(kind));
      }
      return error(absl::StrCat(
          "Alternative \"", alternative.member_name, "\" supports ",
          names.empty() ? "none" : absl::StrJoin(names, ", "),
          " of FIELD_KIND_STMT, FIELD_KIND_RVAL and FIELD_KIND_LVAL: set its "
          "kind."));
    }
    kinds.push_back(supported.front());
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
            "Alternative \"", field.alternatives_[i].member_name,
            "\" is FIELD_KIND_STMT, which requires ir_names (a statement is "
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
        field.member_ir_kinds_[field.alternatives_[i].member_name] =
            FIELD_KIND_LVAL;
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

    auto it = absl::c_find_if(field.ir_splits_, [&](const FieldIrSplitPb& s) {
      return s.operand() == *alternative.ir_name ||
             s.region() == *alternative.ir_name;
    });
    if (it == field.ir_splits_.end()) {
      FieldIrSplitPb& split = field.ir_splits_.emplace_back();
      split.add_types(alternative.member_name);
      if (is_region) {
        split.set_region(*alternative.ir_name);
      } else {
        split.set_operand(*alternative.ir_name);
      }
      split.set_kind(kind);
      continue;
    }
    if (it->kind() != kind || it->has_region() != is_region) {
      return error(absl::StrCat(
          "The alternatives with ir_name \"", *alternative.ir_name,
          "\" must have the same kind and all be held in a region or all in "
          "an operand, but \"",
          it->types(0), "\" is ", FieldKind_Name(it->kind()),
          it->has_region() ? " (region)" : " (operand)", " and \"",
          alternative.member_name, "\" is ", FieldKind_Name(kind),
          is_region ? " (region)." : " (operand)."));
    }
    it->add_types(alternative.member_name);
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
