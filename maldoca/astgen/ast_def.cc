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
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "maldoca/astgen/ast_def.pb.h"
#include "maldoca/astgen/symbol.h"
#include "maldoca/astgen/type.h"
#include "maldoca/base/status_macros.h"

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
    NodeDef *node,
    std::function<std::vector<NodeDef *>(NodeDef *)> get_dependencies,
    absl::flat_hash_set<NodeDef *> *pre_order_visited,
    std::vector<NodeDef *> *sorted_dependencies) {
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
  std::vector<NodeDef *> dependencies = get_dependencies(node);
  for (NodeDef *dependency : dependencies) {
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
std::vector<NodeDef *> TopologicalSortDependencies(
    NodeDef *node,
    std::function<std::vector<NodeDef *>(NodeDef *)> get_dependencies) {
  absl::flat_hash_set<NodeDef *> pre_order_visited;
  std::vector<NodeDef *> sorted_dependencies;
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
    const Type &type,
    const absl::flat_hash_map<std::string, std::unique_ptr<NodeDef>> &nodes,
    std::vector<NodeDef *> *dependencies) {
  switch (type.kind()) {
    case TypeKind::kBuiltin:
    case TypeKind::kEnum:
      // No dependencies.
      break;

    case TypeKind::kClass: {
      const auto &class_type = static_cast<const ClassType &>(type);
      auto it = nodes.find(class_type.name().ToPascalCase());
      CHECK(it != nodes.end())
          << class_type.name().ToPascalCase() << " undefined.";
      dependencies->push_back(it->second.get());
      break;
    }

    case TypeKind::kList: {
      const auto &list_type = static_cast<const ListType &>(type);
      GetDependencies(list_type.element_type(), nodes, dependencies);
      break;
    }

    case TypeKind::kVariant: {
      const auto &variant_type = static_cast<const VariantType &>(type);
      for (const auto &type : variant_type.types()) {
        GetDependencies(*type, nodes, dependencies);
      }
      break;
    }
  }
}

}  // namespace

absl::StatusOr<EnumMemberDef> EnumMemberDef::FromEnumMemberDefPb(
    const EnumMemberDefPb &member_pb) {
  Symbol name{member_pb.name()};
  if (name.ToPascalCase() != member_pb.name()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The enum member name '", member_pb.name(), "' is not in PascalCase."));
  }

  return EnumMemberDef{std::move(name), std::move(member_pb.string_value())};
}

absl::StatusOr<EnumDef> EnumDef::FromEnumDefPb(const EnumDefPb &enum_pb) {
  Symbol name{enum_pb.name()};
  if (name.ToPascalCase() != enum_pb.name()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The enum type name '", enum_pb.name(), "' is not in PascalCase."));
  }

  std::vector<EnumMemberDef> members;
  for (const EnumMemberDefPb &member_pb : enum_pb.members()) {
    MALDOCA_ASSIGN_OR_RETURN(auto member,
                             EnumMemberDef::FromEnumMemberDefPb(member_pb));
    members.push_back(std::move(member));
  }

  return EnumDef{std::move(name), std::move(members)};
}

absl::StatusOr<FieldDef> FieldDef::FromFieldDefPb(const FieldDefPb &field_pb,
                                                  absl::string_view lang_name) {
  FieldDef field;
  field.name_ = Symbol(field_pb.name());

  // Check that the name is in camelCase.
  if (field.name().ToCamelCase() != field_pb.name()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Field '", field_pb.name(), "' is not in camelCase."));
  }

  MALDOCA_ASSIGN_OR_RETURN(field.type_, FromTypePb(field_pb.type(), lang_name));

  if (field_pb.optionalness() == OPTIONALNESS_UNSPECIFIED) {
    return absl::InvalidArgumentError(
        absl::StrCat("Field '", field_pb.name(),
                     "' has OPTIONALNESS_UNSPECIFIED. This should be a bug, as "
                     "the default value is already OPTIONALNESS_REQUIRED."));
  }
  field.optionalness_ = field_pb.optionalness();

  field.kind_ = field_pb.kind();
  field.ignore_in_ir_ = field_pb.ignore_in_ir();
  field.enclose_in_region_ = field_pb.enclose_in_region();

  return field;
}

std::optional<Symbol> NodeDef::ir_op_name(absl::string_view lang_name,
                                          FieldKind kind) const {
  // If there's a custom IR op name, return it.
  if (ir_op_name_.has_value()) {
    return Symbol(*ir_op_name_);
  }

  // If any descendent has a custom IR op name, then we fallback to mlir::Value.
  if (absl::c_any_of(descendants(), [](const NodeDef *descendent) {
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
  if (absl::c_any_of(descendants(), [](const NodeDef *descendent) {
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

/*static*/
absl::StatusOr<AstDef> AstDef::FromProto(const AstDefPb &pb) {
  std::vector<EnumDef> enum_defs;
  for (const EnumDefPb &enum_def_pb : pb.enums()) {
    MALDOCA_ASSIGN_OR_RETURN(EnumDef enum_def,
                             EnumDef::FromEnumDefPb(enum_def_pb));
    enum_defs.push_back(std::move(enum_def));
  }

  std::vector<std::string> node_names;
  absl::flat_hash_map<std::string, std::unique_ptr<NodeDef>> nodes;

  for (const NodeDefPb &node_pb : pb.nodes()) {
    if (nodes.contains(node_pb.name())) {
      return absl::InvalidArgumentError(
          absl::StrCat(node_pb.name(), " already exists!"));
    }

    auto node = absl::WrapUnique(new NodeDef());

    node->name_ = node_pb.name();

    if (node_pb.has_type()) {
      node->type_ = node_pb.type();
    }

    for (const FieldDefPb &field_pb : node_pb.fields()) {
      MALDOCA_ASSIGN_OR_RETURN(
          FieldDef field, FieldDef::FromFieldDefPb(field_pb, pb.lang_name()));
      node->fields_.push_back(std::move(field));
    }

    node->has_control_flow_ = node_pb.has_control_flow();

    if (node_pb.has_ir_op_name()) {
      node->ir_op_name_ = node_pb.ir_op_name();
    }

    node->should_generate_ir_op_ = node_pb.should_generate_ir_op();

    node->has_fold_ = node_pb.has_fold();

    for (auto kind : node_pb.kinds()) {
      node->kinds_.push_back(static_cast<FieldKind>(kind));
    }

    for (auto mlir_trait : node_pb.additional_mlir_traits()) {
      node->additional_mlir_traits_.push_back(
          static_cast<MlirTrait>(mlir_trait));
    }

    node_names.push_back(node_pb.name());
    nodes.emplace(node_pb.name(), std::move(node));
  }

  // Set parent pointers.
  for (const NodeDefPb &node_pb : pb.nodes()) {
    NodeDef &node = *nodes.at(node_pb.name());

    for (absl::string_view parent_name : node_pb.parents()) {
      auto it = nodes.find(parent_name);
      if (it == nodes.end()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Parent ", parent_name, " doesn't exist!"));
      }
      NodeDef *parent = it->second.get();
      node.parents_.push_back(parent);
    }
  }

  // For union types, create a node to represent each one and add that node as
  // a parent of the specified types.
  for (const UnionTypePb &union_type_pb : pb.union_types()) {
    auto union_type_node = absl::WrapUnique(new NodeDef());
    union_type_node->name_ = union_type_pb.name();
    if (nodes.contains(union_type_pb.name())) {
      return absl::InvalidArgumentError(
          absl::StrCat(union_type_pb.name(), " already exists!"));
    }
    node_names.push_back(union_type_pb.name());
    nodes.emplace(union_type_pb.name(), std::move(union_type_node));
  }

  for (const UnionTypePb &union_type_pb : pb.union_types()) {
    auto union_type_node = nodes.at(union_type_pb.name()).get();
    for (const std::string &type : union_type_pb.types()) {
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
      NodeDef *parent = it->second.get();
      union_type_node->parents_.push_back(parent);
    }
  }

  // NOTE: In the code below, we traverse `node_names` instead of `nodes`.
  // `node_names` preserves the original order of definitions.
  // This makes sure that the algorithm is always deterministic.

  // Set ancestors vector.
  for (const std::string &name : node_names) {
    NodeDef &node = *nodes.at(name);

    node.ancestors_ = TopologicalSortDependencies(
        &node, [](NodeDef *node) { return node->parents_; });
  }

  // Set aggregated_fields vector.
  for (const std::string &name : node_names) {
    NodeDef &node = *nodes.at(name);

    for (NodeDef *ancestor : node.ancestors_) {
      for (FieldDef &field : ancestor->fields_) {
        node.aggregated_fields_.push_back(&field);
      }
    }
    for (FieldDef &field : node.fields_) {
      node.aggregated_fields_.push_back(&field);
    }
  }

  // Set children vector.
  for (const std::string &name : node_names) {
    NodeDef &node = *nodes.at(name);

    for (NodeDef *parent : node.parents_) {
      parent->children_.push_back(&node);
    }
  }

  // Set descendants vector.
  for (const std::string &name : node_names) {
    NodeDef &node = *nodes.at(name);

    node.descendants_ = TopologicalSortDependencies(
        &node, [](NodeDef *node) { return node->children_; });
  }

  // Set leafs vector.
  for (const std::string &name : node_names) {
    NodeDef &node = *nodes.at(name);

    for (NodeDef *descendent : node.descendants_) {
      if (!descendent->children().empty()) {
        continue;
      }
      node.leafs_.push_back(descendent);
    }
  }

  // Set aggregated_kinds vector.
  for (const std::string &name : node_names) {
    NodeDef &node = *nodes.at(name);

    absl::btree_set<FieldKind> aggregated_kinds;

    for (NodeDef *ancestor : node.ancestors_) {
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
  for (const std::string &name : node_names) {
    NodeDef &node = *nodes.at(name);

    absl::btree_set<MlirTrait> aggregated_additional_mlir_traits;

    for (NodeDef *ancestor : node.ancestors_) {
      for (MlirTrait trait : ancestor->additional_mlir_traits()) {
        aggregated_additional_mlir_traits.insert(trait);
      }
    }
    for (MlirTrait trait : node.additional_mlir_traits()) {
      aggregated_additional_mlir_traits.insert(trait);
    }

    node.aggregated_additional_mlir_traits_ = {
        aggregated_additional_mlir_traits.begin(),
        aggregated_additional_mlir_traits.end(),
    };
  }

  // Reorder the node definitions so that dependencies always come first.
  std::vector<NodeDef *> topological_sorted_nodes;
  absl::flat_hash_set<NodeDef *> preorder_visited_nodes;
  for (const std::string &name : node_names) {
    NodeDef &node = *nodes.at(name);

    TopologicalSortDependencies(
        &node,
        [&nodes](NodeDef *node) {
          std::vector<NodeDef *> dependencies;
          dependencies.insert(dependencies.end(), node->parents_.begin(),
                              node->parents_.end());
          for (const FieldDef &field : node->fields()) {
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
  for (NodeDef *node : topological_sorted_nodes) {
    if (!node->parents().empty()) {
      continue;
    }
    if (node->children().empty()) {
      continue;
    }

    std::vector<EnumMemberDef> type_enum_members;
    for (const NodeDef *leaf : node->leafs()) {
      EnumMemberDef member{Symbol(leaf->name()), leaf->name()};
      type_enum_members.push_back(std::move(member));
    }

    node->node_type_enum_ = EnumDef{
        Symbol{node->name()} + "Type",
        std::move(type_enum_members),
    };
  }

  // For each ClassType, if it resolves to a NodeDef, store a reference to it.
  for (NodeDef *node : topological_sorted_nodes) {
    for (FieldDef &field : node->fields_) {
      ResolveClassType(field.type(), topological_sorted_nodes);
    }
  }

  return AstDef{pb.lang_name(), std::move(enum_defs), std::move(node_names),
                std::move(nodes), std::move(topological_sorted_nodes)};
}

void AstDef::ResolveClassType(
    Type &type, absl::Span<const NodeDef *const> topological_sorted_nodes) {
  switch (type.kind()) {
    case TypeKind::kBuiltin:
    case TypeKind::kEnum: {
      break;
    }
    case TypeKind::kClass: {
      auto &class_type = static_cast<ClassType &>(type);
      for (const NodeDef *node : topological_sorted_nodes) {
        if (node->name() == class_type.name().ToPascalCase()) {
          LOG(INFO) << "Resolved class " << node->name();
          class_type.node_def_ = node;
          break;
        }
      }
      break;
    }
    case TypeKind::kList: {
      auto &list_type = static_cast<ListType &>(type);
      ResolveClassType(list_type.element_type(), topological_sorted_nodes);
      break;
    }
    case TypeKind::kVariant: {
      auto &variant_type = static_cast<VariantType &>(type);
      for (auto &type : variant_type.types()) {
        ResolveClassType(*type, topological_sorted_nodes);
      }
      break;
    }
  }
}

}  // namespace maldoca
