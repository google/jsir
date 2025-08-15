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

#include "maldoca/js/ir/analyses/dynamic_constant_propagation/analysis.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <iostream>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/DebugStringHelper.h"
#include "absl/algorithm/container.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/ir/analyses/constant_propagation/analysis.h"
#include "maldoca/js/ir/analyses/dataflow_analysis.h"
#include "maldoca/js/ir/analyses/dynamic_constant_propagation/attrs.h"
#include "maldoca/js/ir/analyses/dynamic_constant_propagation/dynamic_prelude.h"
#include "maldoca/js/ir/analyses/dynamic_constant_propagation/symbol_mutation_info.h"
#include "maldoca/js/ir/analyses/scope.h"
#include "maldoca/js/ir/ir.h"
#include "maldoca/js/ir/jsir_utils.h"
#include "maldoca/js/quickjs/quickjs.h"
#include "quickjs/quickjs.h"

namespace maldoca {

#define MALDOCA_CAT_(a, b) a##b
#define MALDOCA_CAT(a, b) MALDOCA_CAT_(a, b)
#define OPT_ASSIGN_OR_RETURN(var, expression)            \
  auto MALDOCA_CAT(optional_, __LINE__) = (expression);  \
  if (!(MALDOCA_CAT(optional_, __LINE__)).has_value()) { \
    return std::nullopt;                                 \
  }                                                      \
  var = std::move(*MALDOCA_CAT(optional_, __LINE__));

#define OPT_RETURN_IF_FAILED(expression) \
  if (!(expression).has_value()) {       \
    return std::nullopt;                 \
  }

// Common info from `JsirFunctionExpressionOp` and `JsirFunctionDeclarationOp`.
// We only support inlining of functions with a single return statement.
struct JsirInlineFunctionInfo {
  std::vector<JsSymbolId> param_symbols;
  mlir::Operation *return_value;

  static std::optional<JsirInlineFunctionInfo> Create(const BabelScopes &scopes,
                                                      mlir::Region &params,
                                                      mlir::Region &body);

  static std::optional<JsirInlineFunctionInfo> Create(
      const BabelScopes &scopes, JsirFunctionExpressionOp op) {
    return Create(scopes, op.getParams(), op.getBody());
  }

  static std::optional<JsirInlineFunctionInfo> Create(
      const BabelScopes &scopes, JsirFunctionDeclarationOp op) {
    return Create(scopes, op.getParams(), op.getBody());
  }
};

std::optional<JsirInlineFunctionInfo> JsirInlineFunctionInfo::Create(
    const BabelScopes &scopes, mlir::Region &params, mlir::Region &body) {
  absl::StatusOr<mlir::Block *> body_block = GetStmtsRegionBlock(body);
  if (!body_block.ok()) {
    return std::nullopt;
  }

  auto return_op_range = FilterBlockOps<JsirReturnStatementOp>(**body_block);
  if (!llvm::hasNItems(return_op_range, 1)) {
    return std::nullopt;
  }
  JsirReturnStatementOp return_op = *return_op_range.begin();

  if (return_op.getArgument() == nullptr) {
    return std::nullopt;
  }

  mlir::Operation *return_value = return_op.getArgument().getDefiningOp();
  if (return_value == nullptr) {
    return std::nullopt;
  }

  absl::StatusOr<mlir::ValueRange> param_values = GetExprsRegionValues(params);
  if (!param_values.ok()) {
    return std::nullopt;
  }

  std::vector<JsSymbolId> param_symbols;
  for (mlir::Value param_value : *param_values) {
    auto param_op = param_value.getDefiningOp<JsirIdentifierRefOp>();
    if (param_op == nullptr) {
      return std::nullopt;
    }
    param_symbols.push_back(GetSymbolId(scopes, param_op));
  }

  return JsirInlineFunctionInfo{
      .param_symbols = std::move(param_symbols),
      .return_value = return_value,
  };
}

typedef std::optional<mlir::Attribute> (*BuiltinFunc)(
    mlir::MLIRContext *context, absl::Span<const mlir::Attribute> args);

std::optional<mlir::Attribute> BuiltinAtob(
    mlir::MLIRContext *context, absl::Span<const mlir::Attribute> args) {
  if (args.size() != 1) {
    return std::nullopt;
  }
  std::string ascii_string =
      llvm::dyn_cast<mlir::StringAttr>(args[0]).strref().str();

  // Perform the ascii to binary conversion in C++ for now
  std::string binary_string;
  absl::Base64Unescape(ascii_string, &binary_string);
  return mlir::StringAttr::get(context, binary_string);
}

std::optional<mlir::Attribute> BuiltinBtoa(
    mlir::MLIRContext *context, absl::Span<const mlir::Attribute> args) {
  if (args.size() != 1) {
    return std::nullopt;
  }
  std::string binary_string =
      llvm::dyn_cast<mlir::StringAttr>(args[0]).strref().str();

  // Perform the binary to ascii conversion in C++ for now
  std::string ascii_string;
  absl::Base64Escape(binary_string, &ascii_string);
  return mlir::StringAttr::get(context, ascii_string);
}

std::set<std::string> kEncodeURIComponentReservedChars = {
    "-", "_", ".", "!", "~", "*", "'", "(", ")", "*", ",", ";", ":", "@", "&", "=", "+", "$", "/", "?", "#"
};

std::optional<mlir::Attribute> BuiltinDecodeURIComponent(
    mlir::MLIRContext *context, absl::Span<const mlir::Attribute> args) {
  if (args.size() != 1) {
    return std::nullopt;
  }
  std::string encoded_str =
      llvm::dyn_cast<mlir::StringAttr>(args[0]).strref().str();

  std::string decoded;
    char hex_char;
    for (size_t i = 0; i < encoded_str.length(); ++i) {
        if (encoded_str[i] == '%' && i + 2 < encoded_str.length()) {
            std::string hex_str = encoded_str.substr(i + 1, 2);
            char decoded_char = static_cast<char>(std::stoi(hex_str, nullptr, 16));
            decoded += decoded_char;
            i += 2;
        } else {
            decoded += encoded_str[i];
        }
    }
    return mlir::StringAttr::get(context, decoded);
}

std::optional<mlir::Attribute> BuiltinEncodeURIComponent(
    mlir::MLIRContext *context, absl::Span<const mlir::Attribute> args) {
  if (args.size() != 1) {
    return std::nullopt;
  }
  std::string str =
      llvm::dyn_cast<mlir::StringAttr>(args[0]).strref().str();

  std::string encoded;
  for (char c : str) {
    if (isalnum(c) || kEncodeURIComponentReservedChars.count(std::string(1, c))) {
      encoded += c;
    } else {
      encoded += absl::StrFormat("%%%02X", static_cast<unsigned char>(c));
    }
  }
  return mlir::StringAttr::get(context, encoded);
}



static const auto *kBuiltins =
    new absl::flat_hash_map<std::string, BuiltinFunc>{
        {"atob", &BuiltinAtob},
        {"btoa", &BuiltinBtoa},
        {"decodeURIComponent", &BuiltinDecodeURIComponent},
        {"encodeURIComponent", &BuiltinEncodeURIComponent},
    };

static std::string InlineExprToString(mlir::Attribute expr, size_t indent = 0) {
  return llvm::TypeSwitch<mlir::Attribute, std::string>(expr)
      .Case([&](mlir::FloatAttr attr) {
        return absl::StrCat(attr.getValueAsDouble());
      })
      .Case([&](JsirInlineExpressionBinaryExpressionAttr attr) {
        return absl::StrCat("(", InlineExprToString(attr.getLeft()), " ",
                            attr.getOp().str(), " ",
                            InlineExprToString(attr.getRight()), ")");
      })
      .Case([&](JsirSymbolIdAttr attr) {
        JsSymbolId symbol{attr.getName().str(), attr.getDefScopeId()};
        return absl::StrCat(symbol);
      })
      .Case([&](JsirInlineExpressionMemberExpressionAttr attr) {
        return absl::StrCat(InlineExprToString(attr.getObject()), "[",
                            InlineExprToString(attr.getComputedKey()), "]");
      })
      .Case([&](JsirInlineExpressionObjectAttr attr) {
        // /* <n> properties */ {
        //   'key': <value>,
        //   ...
        // }
        std::string str = absl::StrFormat("/* %zu properties */ {\n",
                                          attr.getProperties().size());
        for (JsirInlineExpressionPropertyAttr property : attr.getProperties()) {
          absl::StrAppend(&str, std::string(indent + 4, ' '),
                          InlineExprToString(property), ",\n");
        }
        absl::StrAppend(&str, std::string(indent, ' '), "}");
        return str;
      })
      .Case([&](JsirInlineExpressionPropertyAttr attr) {
        return absl::StrCat("[", InlineExprToString(attr.getKey()),
                            "]: ", InlineExprToString(attr.getValue()));
      })
      .Case([&](JsirStringLiteralAttr attr) {
        return absl::StrCat("\"", attr.getValue().str(), "\"");
      })
      .Case([&](JsirInlineExpressionUnaryExpressionAttr attr) {
        return absl::StrCat(attr.getOp().str(),
                            InlineExprToString(attr.getArgument()));
      })
      .Case([&](JsirInlineExpressionCallAttr attr) {
        std::vector<std::string> arguments;
        arguments.reserve(attr.getArguments().size());
        for (mlir::Attribute argument : attr.getArguments()) {
          arguments.push_back(InlineExprToString(argument));
        }
        return absl::StrCat(InlineExprToString(attr.getCallee()), "(",
                            absl::StrJoin(arguments, ", "), ")");
      })
      .Case([&](JsirInlineExpressionFunctionAttr attr) {
        std::vector<std::string> params;
        params.reserve(attr.getParams().size());
        for (mlir::Attribute param : attr.getParams()) {
          params.push_back(InlineExprToString(param));
        }
        return absl::StrCat("(", absl::StrJoin(params, ", "), ") => { ",
                            InlineExprToString(attr.getBody()), " }");
      })
      .Default([&](mlir::Attribute attr) { return mlir::debugString(attr); });
}

void PrintBindings(
    llvm::raw_ostream &os,
    const absl::flat_hash_map<JsSymbolId, mlir::Attribute> &bindings) {
  std::vector<JsSymbolId> sorted_symbols;
  for (const auto &[symbol, value] : bindings) {
    sorted_symbols.push_back(symbol);
  }
  absl::c_sort(sorted_symbols);

  for (const JsSymbolId &symbol : sorted_symbols) {
    os << symbol << ":\n";
    os << "    " << InlineExprToString(bindings.at(symbol), /*indent=*/4);
    os << "\n";
  }

  os.flush();
}

// =============================================================================
// GetConstBindings
// =============================================================================
//
// Here we build a lookup table from symbols (`JsSymbolId`s) to "inline
// expressions" (`mlir::Attribute`s). The idea is that these expressions have
// the possibility of being evaluated into a value (`mlir::Attribute`) after
// inline.
//
// The difference between an "inline expression" and a "value" is that, a
// "value" cannot contain any unresolved symbol references.
//
// Example 1:
// ----------
//
// For the code:
//
// ```
// const a = 0;
// const b = a;
// ```
//
// We will build the initial lookup table:
//
// ```
// 'a': NumericLiteral{0}
// 'b': Identifier{'a'}
// ```
//
// Then, when evaluating `Identifier{'a'}`, since we have access to the lookup
// table, we will get the result `NumericLiteral{0}`.
//
// Example 2:
// ----------
//
// For the code:
//
// ```
// function foo(x) { return x + 1; }
// const y = foo(1);
// ```
//
// We will build the initial lookup table:
//
// ```
// 'foo': Function{BinaryExpression{'+', Param#0, NumericLiteral{1}}}
// 'y':   CallExpression{Identifier{'foo'}, [NumericLiteral{1}]}
// ```
//
// Then, when evaluating the `CallExpression`, since we have access to the
// lookup table, we will get the result `NumericLiteral{2}`.

std::optional<mlir::Attribute> GetInlineExpr(const BabelScopes &scopes,
                                             mlir::Value value);

std::optional<mlir::Attribute> GetInlineExpr(const BabelScopes &scopes,
                                             mlir::Operation *op);

std::optional<mlir::Attribute> GetInlineExpr(const BabelScopes &scopes,
                                             JsirIdentifierOp op);

std::optional<JsirInlineExpressionPropertyAttr> GetInlineExpr(
    const BabelScopes &scopes, JsirObjectPropertyOp op);

std::optional<JsirInlineExpressionFunctionAttr> GetInlineExpr(
    const BabelScopes &scopes, const JsirInlineFunctionInfo &func_info);

void GetSymbolDependencies(mlir::Attribute attr,
                           absl::flat_hash_set<JsSymbolId> &dependencies);

struct SymbolDependencyNode {
  JsSymbolId symbol;
  std::vector<SymbolDependencyNode *> dependencies;

  explicit SymbolDependencyNode(JsSymbolId symbol) : symbol(symbol) {}
};

struct SymbolDependencyGraph {
  std::vector<SymbolDependencyNode> nodes_vector;
  absl::flat_hash_map<JsSymbolId, SymbolDependencyNode *> nodes_map;

  SymbolDependencyNode *getEntryNode() { return &nodes_vector.back(); }

  static SymbolDependencyGraph Create(
      const absl::flat_hash_map<JsSymbolId, mlir::Attribute> &bindings);
};

SymbolDependencyGraph SymbolDependencyGraph::Create(
    const absl::flat_hash_map<JsSymbolId, mlir::Attribute> &bindings) {
  std::vector<SymbolDependencyNode> nodes_vec;
  for (const auto &[symbol_id, _] : bindings) {
    nodes_vec.push_back(SymbolDependencyNode(symbol_id));
  }
  nodes_vec.push_back(SymbolDependencyNode(JsSymbolId{"", 0}));

  absl::flat_hash_map<JsSymbolId, SymbolDependencyNode *> nodes_map;
  for (SymbolDependencyNode &node : nodes_vec) {
    nodes_map[node.symbol] = &node;
  }

  for (const auto &[symbol_id, _] : bindings) {
    nodes_vec.back().dependencies.push_back(nodes_map.at(symbol_id));
  }

  for (const auto &[symbol_id, attribute] : bindings) {
    absl::flat_hash_set<JsSymbolId> dependencies;
    GetSymbolDependencies(attribute, dependencies);

    for (const JsSymbolId &dependency : dependencies) {
      if (auto it = nodes_map.find(dependency); it != nodes_map.end()) {
        nodes_map[symbol_id]->dependencies.push_back(it->second);
      }
    }
  }

  SymbolDependencyGraph graph{.nodes_vector = std::move(nodes_vec),
                              .nodes_map = std::move(nodes_map)};

  return graph;
};

}  // namespace maldoca

namespace llvm {
template <>
struct GraphTraits<maldoca::SymbolDependencyNode *> {
  using NodeRef = maldoca::SymbolDependencyNode *;

  using ChildIteratorType =
      std::vector<maldoca::SymbolDependencyNode *>::iterator;

  static ChildIteratorType child_begin(NodeRef N) {
    return N->dependencies.begin();
  }

  static ChildIteratorType child_end(NodeRef N) {
    return N->dependencies.end();
  }
};

template <>
struct GraphTraits<maldoca::SymbolDependencyGraph *>
    : public GraphTraits<maldoca::SymbolDependencyNode *> {
  using nodes_iterator = std::vector<maldoca::SymbolDependencyNode>::iterator;

  static nodes_iterator nodes_begin(maldoca::SymbolDependencyGraph *G) {
    return G->nodes_vector.begin();
  }

  static nodes_iterator nodes_end(maldoca::SymbolDependencyGraph *G) {
    return G->nodes_vector.end();
  }

  static NodeRef getEntryNode(maldoca::SymbolDependencyGraph *G) {
    return G->getEntryNode();
  }
};

}  // namespace llvm

namespace maldoca {

absl::flat_hash_map<JsSymbolId, mlir::Attribute> GetConstBindings(
    const BabelScopes &scopes, mlir::Operation *root) {
  absl::flat_hash_map<JsSymbolId, mlir::Attribute> bindings;

  root->walk([&](JsirFunctionDeclarationOp op) {
    // Defining a lambda and executing it immediately so that we can use
    // OPT_ASSIGN_OR_RETURN.
    //
    // OPT_ASSIGN_OR_RETURN returns std::nullopt on failure, but the return type
    // of the walk() callback is void.
    [&] {
      OPT_ASSIGN_OR_RETURN(auto func_info,
                           JsirInlineFunctionInfo::Create(scopes, op));

      OPT_ASSIGN_OR_RETURN(JsirInlineExpressionFunctionAttr attr,
                           GetInlineExpr(scopes, func_info));

      OPT_ASSIGN_OR_RETURN(JsirIdentifierAttr id_attr, op.getId());
      JsSymbolId symbol = GetSymbolId(scopes, id_attr);

      bindings.insert({symbol, attr});

      return std::nullopt;
    }();
  });

  root->walk([&](JsirVariableDeclaratorOp op) {
    auto id_ref_op = op.getId().getDefiningOp<JsirIdentifierRefOp>();
    if (id_ref_op == nullptr) {
      return;
    }
    JsSymbolId def_symbol = GetSymbolId(scopes, id_ref_op);

    if (op.getInit() == nullptr) {
      return;
    }

    std::optional<mlir::Attribute> expr = GetInlineExpr(scopes, op.getInit());

    if (expr.has_value()) {
      bindings.insert({def_symbol, *expr});
    }
  });

  absl::flat_hash_map<JsSymbolId, SymbolMutationInfo> infos =
      GetSymbolMutationInfos(scopes, root);

  for (const auto &[symbol, info] : infos) {
    if (info.num_assignments + info.num_mutations != 1) {
      bindings.erase(symbol);
    }
  }

  SymbolDependencyGraph symbol_dependency_graph =
      SymbolDependencyGraph::Create(bindings);

  for (auto it = llvm::scc_begin(&symbol_dependency_graph);
       it != ::llvm::scc_end(&symbol_dependency_graph); ++it) {
    if (it.hasCycle()) {
      for (SymbolDependencyNode *node : *it) {
        bindings.erase(node->symbol);
      }
    }
  }

  return bindings;
}

void GetSymbolDependencies(mlir::Attribute attr,
                           absl::flat_hash_set<JsSymbolId> &dependencies) {
  llvm::TypeSwitch<mlir::Attribute, void>(attr)
      .Case([&](JsirInlineExpressionBinaryExpressionAttr attr) {
        GetSymbolDependencies(attr.getLeft(), dependencies);
        GetSymbolDependencies(attr.getRight(), dependencies);
      })
      .Case([&](JsirSymbolIdAttr attr) {
        dependencies.insert(
            JsSymbolId{attr.getName().str(), attr.getDefScopeId()});
      })
      .Case([&](JsirInlineExpressionMemberExpressionAttr attr) {
        GetSymbolDependencies(attr.getObject(), dependencies);
        GetSymbolDependencies(attr.getComputedKey(), dependencies);
      })
      .Case([&](JsirInlineExpressionObjectAttr attr) {
        for (JsirInlineExpressionPropertyAttr property : attr.getProperties()) {
          GetSymbolDependencies(property, dependencies);
        }
      })
      .Case([&](JsirInlineExpressionPropertyAttr attr) {
        GetSymbolDependencies(attr.getKey(), dependencies);
        GetSymbolDependencies(attr.getValue(), dependencies);
      })
      .Case([&](JsirInlineExpressionUnaryExpressionAttr attr) {
        GetSymbolDependencies(attr.getArgument(), dependencies);
      })
      .Case([&](JsirInlineExpressionCallAttr attr) {
        GetSymbolDependencies(attr.getCallee(), dependencies);
        for (mlir::Attribute argument : attr.getArguments()) {
          GetSymbolDependencies(argument, dependencies);
        }
      })
      .Case([&](JsirInlineExpressionFunctionAttr attr) {
        GetSymbolDependencies(attr.getBody(), dependencies);
      })
      .Default([&](mlir::Attribute attr) {});
}

// Normalizes object keys into strings. JavaScript spec says that object keys
// are all converted to strings: https://tc39.es/ecma262/#sec-object-type.
std::optional<mlir::StringAttr> NormalizeLiteralKey(mlir::Attribute attr) {
  using Ret = std::optional<mlir::StringAttr>;

  return llvm::TypeSwitch<mlir::Attribute, Ret>(attr)
      .Case([&](mlir::StringAttr attr) { return attr; })
      .Case([&](mlir::FloatAttr attr) {
        mlir::MLIRContext *context = attr.getContext();
        return mlir::StringAttr::get(context,
                                     absl::StrCat(attr.getValueAsDouble()));
      })
      .Case([&](JsirIdentifierAttr attr) { return attr.getName(); })
      .Case([&](JsirStringLiteralAttr attr) { return attr.getValue(); })
      .Case([&](JsirNumericLiteralAttr attr) {
        return NormalizeLiteralKey(attr.getValue());
      })
      .Case([&](JsirBigIntLiteralAttr attr) { return attr.getValue(); })
      .Default([&](mlir::Attribute attr) { return std::nullopt; });
}

std::optional<mlir::Attribute> GetInlineExpr(const BabelScopes &scopes,
                                             mlir::Value value) {
  CHECK(value != nullptr);
  if (value.getDefiningOp() == nullptr) {
    return std::nullopt;
  }
  return GetInlineExpr(scopes, value.getDefiningOp());
}

std::optional<mlir::Attribute> GetInlineExpr(const BabelScopes &scopes,
                                             mlir::Operation *op) {
  mlir::MLIRContext *mlir_context = op->getContext();
  using Ret = std::optional<mlir::Attribute>;

  return llvm::TypeSwitch<mlir::Operation *, Ret>(op)
      .Case([&](JsirBinaryExpressionOp op) -> Ret {
        OPT_ASSIGN_OR_RETURN(mlir::Attribute left,
                             GetInlineExpr(scopes, op.getLeft()));

        OPT_ASSIGN_OR_RETURN(mlir::Attribute right,
                             GetInlineExpr(scopes, op.getRight()));

        return JsirInlineExpressionBinaryExpressionAttr::get(
            mlir_context, left, right, op.getOperator_Attr());
      })

      .Case([&](JsirCallExpressionOp op) -> Ret {
        OPT_ASSIGN_OR_RETURN(mlir::Attribute callee,
                             GetInlineExpr(scopes, op.getCallee()));

        std::vector<mlir::Attribute> arguments;
        for (mlir::Value argument_value : op.getArguments()) {
          OPT_ASSIGN_OR_RETURN(mlir::Attribute argument,
                               GetInlineExpr(scopes, argument_value));
          arguments.push_back(argument);
        }

        return JsirInlineExpressionCallAttr::get(mlir_context, callee,
                                                 arguments);
      })

      .Case([&](JsirFunctionExpressionOp op) -> Ret {
        OPT_ASSIGN_OR_RETURN(auto func_info,
                             JsirInlineFunctionInfo::Create(scopes, op));

        return GetInlineExpr(scopes, func_info);
      })

      .Case(
          [&](JsirIdentifierOp op) -> Ret { return GetInlineExpr(scopes, op); })

      .Case([&](JsirMemberExpressionOp op) -> Ret {
        OPT_ASSIGN_OR_RETURN(mlir::Attribute object,
                             GetInlineExpr(scopes, op.getObject()));

        mlir::Attribute key;
        if (mlir::Value computed_key = op.getComputedProperty();
            computed_key != nullptr) {
          OPT_ASSIGN_OR_RETURN(key, GetInlineExpr(scopes, computed_key));
        } else if (mlir::Attribute literal_key = op.getLiteralPropertyAttr();
                   literal_key != nullptr) {
          OPT_ASSIGN_OR_RETURN(key, NormalizeLiteralKey(literal_key));
        } else {
          return std::nullopt;
        }

        return JsirInlineExpressionMemberExpressionAttr::get(mlir_context,
                                                             object, key);
      })

      .Case([&](JsirObjectExpressionOp op) -> Ret {
        absl::StatusOr<mlir::ValueRange> property_values =
            GetExprsRegionValues(op.getRegion());
        if (!property_values.ok()) {
          LOG(ERROR) << "Failed to get property values.";
          return std::nullopt;
        }

        std::vector<JsirInlineExpressionPropertyAttr> properties;
        for (mlir::Value property_value : *property_values) {
          auto property_op =
              property_value.getDefiningOp<JsirObjectPropertyOp>();
          if (property_op == nullptr) {
            return std::nullopt;
          }

          std::optional<JsirInlineExpressionPropertyAttr> property =
              GetInlineExpr(scopes, property_op);
          if (!property.has_value()) {
            // It's okay to have some properties not inlined.
            continue;
          }

          properties.push_back(*property);
        }

        return JsirInlineExpressionObjectAttr::get(mlir_context, properties);
      })

      .Case([&](JsirObjectPropertyOp op) -> Ret {
        return GetInlineExpr(scopes, op);
      })

      .Case([&](JsirParenthesizedExpressionOp op) -> Ret {
        return GetInlineExpr(scopes, op.getOperand());
      })

      .Case([&](JsirStringLiteralOp op) -> Ret { return op.getValueAttr(); })

      .Case([&](JsirNumericLiteralOp op) -> Ret { return op.getValueAttr(); })

      .Case([&](JsirUnaryExpressionOp op) -> Ret {
        OPT_ASSIGN_OR_RETURN(mlir::Attribute argument,
                             GetInlineExpr(scopes, op.getArgument()));

        return JsirInlineExpressionUnaryExpressionAttr::get(
            mlir_context, argument, op.getOperator_Attr());
      })

      .Default([&](mlir::Operation *op) -> Ret { return std::nullopt; });
}

std::optional<mlir::Attribute> GetInlineExpr(const BabelScopes &scopes,
                                             JsirIdentifierOp op) {
  mlir::MLIRContext *mlir_context = op.getContext();

  JsSymbolId symbol = GetSymbolId(scopes, op);

  return JsirSymbolIdAttr::get(mlir_context, op.getNameAttr(),
                               symbol.def_scope_uid());
}

std::optional<mlir::StringAttr> GetInlineExprFromKey(
    mlir::Value computed_key, mlir::Attribute literal_key) {
  using Ret = std::optional<mlir::StringAttr>;
  if (computed_key != nullptr) {
    mlir::Operation *computed_key_op = computed_key.getDefiningOp();

    return llvm::TypeSwitch<mlir::Operation *, Ret>(computed_key_op)
        .Case([&](JsirStringLiteralOp op) { return op.getValueAttr(); })
        .Case([&](JsirNumericLiteralOp op) {
          mlir::MLIRContext *context = op.getContext();
          mlir::FloatAttr value = op.getValueAttr();
          return mlir::StringAttr::get(context,
                                       absl::StrCat(value.getValueAsDouble()));
        })
        .Case([&](JsirBigIntLiteralOp op) { return op.getValueAttr(); })
        .Default([&](mlir::Operation *op) { return std::nullopt; });
  }

  return NormalizeLiteralKey(literal_key);
}

std::optional<JsirInlineExpressionPropertyAttr> GetInlineExpr(
    const BabelScopes &scopes, JsirObjectPropertyOp op) {
  OPT_ASSIGN_OR_RETURN(
      mlir::StringAttr key,
      GetInlineExprFromKey(op.getComputedKey(), op.getLiteralKeyAttr()));

  OPT_ASSIGN_OR_RETURN(mlir::Attribute value,
                       GetInlineExpr(scopes, op.getValue()));

  return JsirInlineExpressionPropertyAttr::get(op.getContext(), key, value);
}

std::optional<JsirInlineExpressionFunctionAttr> GetInlineExpr(
    const BabelScopes &scopes, const JsirInlineFunctionInfo &func_info) {
  OPT_ASSIGN_OR_RETURN(mlir::Attribute return_value,
                       GetInlineExpr(scopes, func_info.return_value));

  mlir::MLIRContext *context = func_info.return_value->getContext();
  std::vector<JsirSymbolIdAttr> params;
  params.reserve(func_info.param_symbols.size());
  for (const JsSymbolId &symbol : func_info.param_symbols) {
    auto param = JsirSymbolIdAttr::get(
        context, mlir::StringAttr::get(context, symbol.name()),
        symbol.def_scope_uid());
    params.push_back(param);
  }

  return JsirInlineExpressionFunctionAttr::get(context, params, return_value);
}

void JsirDynamicConstantPropagationAnalysis::VisitOp(
    mlir::Operation *op,
    llvm::ArrayRef<const JsirConstantPropagationValue *> operands,
    const JsirConstantPropagationState *before,
    llvm::MutableArrayRef<JsirStateRef<JsirConstantPropagationValue>> results,
    JsirStateRef<JsirConstantPropagationState> after) {
  op->getContext()->getOrLoadDialect<JsirBuiltinDialect>();

  llvm::TypeSwitch<mlir::Operation *, void>(op)
      .Case([&](JsirIdentifierOp op) {
        assert(results.size() == 1);
        auto &result = results[0];
        VisitIdentifier(op, operands, before, result);
        after.Join(*before);
      })
      .Case([&](JsirCallExpressionOp op) {
        return VisitCallExpression(op, operands, before, results, after);
      })
      .Case([&](JsirMemberExpressionOp op) {
        return VisitMemberExpression(op, operands, before, results, after);
      })
      .Default([&](mlir::Operation *op) {
        return JsirConstantPropagationAnalysis::VisitOp(op, operands, before,
                                                        results, after);
      });
}

// =============================================================================

void JsirDynamicConstantPropagationAnalysis::VisitIdentifier(
    JsirIdentifierOp op, OperandStates<JsirIdentifierOp> operands,
    const JsirConstantPropagationState *before,
    JsirStateRef<JsirConstantPropagationValue> result) {
  JsSymbolId symbol_id = GetSymbolId(scopes_, op);

  const JsirConstantPropagationValue &const_prop_value = before->Get(symbol_id);
  if (!const_prop_value.IsUninitialized() && !const_prop_value.IsUnknown()) {
    result.Join(const_prop_value);
    return;
  }

  std::optional<mlir::Attribute> inline_result =
      EvalIdentifier(op.getNameAttr(), symbol_id.def_scope_uid(), {});

  if (!symbol_id.def_scope_uid().has_value() &&
      kBuiltins->contains(op.getNameAttr().str())) {
    inline_result =
        JsirBuiltinFunctionAttr::get(op.getContext(), op.getNameAttr());
  }

  if (inline_result.has_value()) {
    result.Join(JsirConstantPropagationValue{*inline_result});
    return;
  }

  result.Join(const_prop_value);
}

std::optional<mlir::Attribute>
JsirDynamicConstantPropagationAnalysis::EvalIdentifier(
    mlir::StringAttr name, std::optional<int64_t> def_scope_id,
    const absl::flat_hash_map<JsSymbolId, mlir::Attribute> &bindings) {
  JsSymbolId symbol_id{name.str(), def_scope_id};

  if (dynamic_prelude_->GetFunction(symbol_id).has_value()) {
    return JsirBuiltinFunctionAttr::get(name.getContext(), name);
  };

  if (auto it = bindings.find(symbol_id); it != bindings.end()) {
    return Eval(it->second, bindings);
  }

  if (auto it = const_bindings_.find(symbol_id); it != const_bindings_.end()) {
    return Eval(it->second, bindings);
  }

  return std::nullopt;
}

std::optional<mlir::Attribute>
JsirDynamicConstantPropagationAnalysis::EvalCallExpression(
    mlir::Attribute callee, std::vector<mlir::Attribute> arguments) {
  JSContext *qjs_context = dynamic_prelude_->GetQjsContext();
  mlir::MLIRContext *mlir_context = callee.getContext();

  using Ret = std::optional<mlir::Attribute>;

  return llvm::TypeSwitch<mlir::Attribute, Ret>(callee)
      .Case([&](JsirBuiltinFunctionAttr callee) -> Ret {
        if (kBuiltins->contains(callee.getName().str())) {
          return kBuiltins->at(callee.getName().str())(mlir_context, arguments);
        }

        QjsValue qjs_global{qjs_context, JS_GetGlobalObject(qjs_context)};

        OPT_ASSIGN_OR_RETURN(
            QjsValue qjs_function,
            dynamic_prelude_->GetFunction(callee.getName().strref()));

        std::vector<QjsValue> qjs_arguments;
        std::vector<JSValue> qjs_argument_refs;
        qjs_arguments.reserve(arguments.size());
        qjs_argument_refs.reserve(arguments.size());
        for (const auto &argument : arguments) {
          OPT_ASSIGN_OR_RETURN(
              QjsValue qjs_argument,
              MlirAttributeToQuickJsValue(qjs_context, argument));

          qjs_arguments.push_back(qjs_argument);
          qjs_argument_refs.push_back(qjs_argument.get());
        }

        QjsValue qjs_result{
            qjs_context,
            JS_Call(qjs_context, /*func_obj=*/qjs_function.get(),
                    /*this_obj=*/qjs_global.get(), qjs_argument_refs.size(),
                    qjs_argument_refs.data()),
        };

        return QuickJsValueToMlirAttribute(qjs_context, mlir_context,
                                           qjs_result.get());
      })

      .Case([&](JsirInlineExpressionFunctionAttr callee) -> Ret {
        if (callee.getParams().size() != arguments.size()) {
          return std::nullopt;
        }

        absl::flat_hash_map<JsSymbolId, mlir::Attribute> bindings;
        for (auto [param, argument] :
             llvm::zip(callee.getParams(), arguments)) {
          JsSymbolId symbol_id{param.getName().str(), param.getDefScopeId()};
          bindings[symbol_id] = argument;
        }

        auto result = Eval(callee.getBody(), bindings);
        return result;
      })

      .Default([&](mlir::Attribute callee) { return std::nullopt; });
}

std::optional<mlir::Attribute> JsirDynamicConstantPropagationAnalysis::Eval(
    mlir::Attribute expr,
    const absl::flat_hash_map<JsSymbolId, mlir::Attribute> &bindings) {
  JSContext *qjs_context = dynamic_prelude_->GetQjsContext();
  mlir::MLIRContext *mlir_context = expr.getContext();

  using Ret = std::optional<mlir::Attribute>;

  return llvm::TypeSwitch<mlir::Attribute, Ret>(expr)
      .Case([&](mlir::FloatAttr expr) { return expr; })

      .Case([&](JsirNumericLiteralAttr expr) { return expr; })

      .Case([&](JsirInlineExpressionBinaryExpressionAttr expr) -> Ret {
        std::string op = expr.getOp().str();
        OPT_ASSIGN_OR_RETURN(mlir::Attribute left,
                             Eval(expr.getLeft(), bindings));
        OPT_ASSIGN_OR_RETURN(QjsValue qjs_left,
                             MlirAttributeToQuickJsValue(qjs_context, left));

        OPT_ASSIGN_OR_RETURN(mlir::Attribute right,
                             Eval(expr.getRight(), bindings));
        OPT_ASSIGN_OR_RETURN(QjsValue qjs_right,
                             MlirAttributeToQuickJsValue(qjs_context, right));

        QjsValue qjs_res = EmulateBinOp(qjs_context, op, qjs_left, qjs_right);

        return QuickJsValueToMlirAttribute(qjs_context, mlir_context,
                                           qjs_res.get());
      })

      .Case([&](JsirSymbolIdAttr expr) -> Ret {
        return EvalIdentifier(expr.getName(), expr.getDefScopeId(), bindings);
      })

      .Case([&](JsirInlineExpressionMemberExpressionAttr expr) -> Ret {
        OPT_ASSIGN_OR_RETURN(mlir::Attribute object_unchecked,
                             Eval(expr.getObject(), bindings));
        auto object =
            llvm::dyn_cast<JsirInlineExpressionObjectAttr>(object_unchecked);
        if (object == nullptr) {
          return std::nullopt;
        }

        OPT_ASSIGN_OR_RETURN(mlir::Attribute key,
                             Eval(expr.getComputedKey(), bindings));

        OPT_ASSIGN_OR_RETURN(mlir::StringAttr normalized_key,
                             NormalizeLiteralKey(key));

        for (JsirInlineExpressionPropertyAttr property :
             object.getProperties()) {
          if (property.getKey().strref() != normalized_key.strref()) {
            continue;
          }

          return Eval(property.getValue(), bindings);
        }

        return std::nullopt;
      })

      .Case([&](JsirInlineExpressionObjectAttr expr) { return expr; })

      .Case([&](JsirInlineExpressionPropertyAttr expr) { return expr; })

      .Case([&](JsirStringLiteralAttr expr) { return expr; })

      .Case([&](mlir::StringAttr expr) -> Ret { return expr; })

      .Case([&](JsirInlineExpressionUnaryExpressionAttr expr) -> Ret {
        std::string op = expr.getOp().str();

        OPT_ASSIGN_OR_RETURN(mlir::Attribute operand,
                             Eval(expr.getArgument(), bindings));
        OPT_ASSIGN_OR_RETURN(QjsValue qjs_operand,
                             MlirAttributeToQuickJsValue(qjs_context, operand));

        QjsValue qjs_res = EmulateUnaryOp(qjs_context, op, qjs_operand);

        return QuickJsValueToMlirAttribute(qjs_context, mlir_context,
                                           qjs_res.get());
      })

      .Case([&](JsirInlineExpressionCallAttr expr) -> Ret {
        OPT_ASSIGN_OR_RETURN(mlir::Attribute callee,
                             Eval(expr.getCallee(), bindings));

        std::vector<mlir::Attribute> new_arguments;
        for (mlir::Attribute argument : expr.getArguments()) {
          OPT_ASSIGN_OR_RETURN(mlir::Attribute new_argument,
                               Eval(argument, bindings));
          new_arguments.push_back(new_argument);
        }

        return EvalCallExpression(callee, new_arguments);
      })

      .Case([&](JsirInlineExpressionFunctionAttr expr) { return expr; })

      .Default([&](mlir::Attribute expr) { return std::nullopt; });
}

void JsirDynamicConstantPropagationAnalysis::VisitCallExpression(
    JsirCallExpressionOp op, OperandStates<JsirCallExpressionOp> operands,
    const JsirConstantPropagationState *before,
    llvm::MutableArrayRef<JsirStateRef<JsirConstantPropagationValue>> results,
    JsirStateRef<JsirConstantPropagationState> after) {
  assert(results.size() == 1);
  auto &result = results[0];

  // Propagate dense state.
  after.Join(*before);

  // If any of the arguments is uninitialized, that means states haven't
  // "flowed" into this CallExpression yet, and we are visiting it during
  // initialization, so we will leave the result as Uninitialized.
  const JsirConstantPropagationValue &callee_lattice = *operands.getCallee();
  if (callee_lattice.IsUninitialized()) {
    return;
  }
  for (const auto *argument_lattice : operands.getArguments()) {
    if (argument_lattice->IsUninitialized()) {
      return;
    }
  }

  // At this point, we're committed to evaluate the transfer function. From
  // now on, at any point, if we determine that we can't compute a definite
  // value, we set the result to Unknown.
  auto set_result_to_unknown = absl::MakeCleanup(
      [&] { result.Join(JsirConstantPropagationValue::Unknown()); });

  // If any of the arguments is unknown, the result must be unknown too.
  std::vector<mlir::Attribute> arguments;
  for (auto *argument_lattice : operands.getArguments()) {
    if (argument_lattice->IsUnknown()) {
      return;
    }
    mlir::Attribute argument = ***argument_lattice;
    arguments.push_back(argument);
  }

  // Check if the callee is a builtin function or property function.
  if (callee_lattice.IsUnknown()) {
    return;
  }

  mlir::Attribute callee = **callee_lattice;
  std::optional<mlir::Attribute> result_attr =
      EvalCallExpression(callee, arguments);
  if (!result_attr.has_value()) {
    return;
  }

  // Now that we successfully computed the result, we no longer want to set
  // it to Unknown. We cancel the assignment here.
  std::move(set_result_to_unknown).Cancel();
  result.Join(JsirConstantPropagationValue(*result_attr));
}

// VisitMemberExpression inline when target is a string/numeric literal.
// If inline target is a function call or operations, we inline it in
// VisitCallExpression.
void JsirDynamicConstantPropagationAnalysis::VisitMemberExpression(
    JsirMemberExpressionOp op, OperandStates<JsirMemberExpressionOp> operands,
    const JsirConstantPropagationState *before,
    llvm::MutableArrayRef<JsirStateRef<JsirConstantPropagationValue>> results,
    JsirStateRef<JsirConstantPropagationState> after) {
  assert(results.size() == 1);
  auto &result = results[0];

  // Propagate dense state.
  after.Join(*before);

  // By default, we will use the MLIR folder result.
  absl::Cleanup fallback_to_folder = [&] {
    JsirConstantPropagationAnalysis::VisitOpDefault(op, operands.getOperands(),
                                                    before, results, after);
  };

  const JsirConstantPropagationValue *object_lattice = operands.getObject();
  if (object_lattice->IsUninitialized() || object_lattice->IsUnknown()) {
    return;
  }

  auto object =
      llvm::dyn_cast<JsirInlineExpressionObjectAttr>(***object_lattice);
  if (object == nullptr) {
    return;
  }

  auto target_key_attr = [&]() -> std::optional<mlir::Attribute> {
    if (mlir::Value property = op.getComputedProperty(); property != nullptr) {
      const JsirConstantPropagationValue *key_lattice =
          operands.getComputedProperty();
      if (key_lattice->IsUninitialized() || key_lattice->IsUnknown()) {
        return std::nullopt;
      }
      return ***key_lattice;
    }

    return op.getLiteralProperty();
  }();

  if (!target_key_attr.has_value()) {
    return;
  }

  auto normalized_target_key = NormalizeLiteralKey(*target_key_attr);
  if (!normalized_target_key.has_value()) {
    return;
  }

  auto result_attr = [&]() -> std::optional<mlir::Attribute> {
    for (JsirInlineExpressionPropertyAttr property : object.getProperties()) {
      if (property.getKey().str() != normalized_target_key->str()) {
        continue;
      }

      return Eval(property.getValue(), /*bindings=*/{});
    }

    return std::nullopt;
  }();

  if (!result_attr.has_value()) {
    return;
  }

  result.Join(JsirConstantPropagationValue(*result_attr));
  std::move(fallback_to_folder).Cancel();
}

}  // namespace maldoca
