// Copyright 2025 Google LLC
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

#ifndef MALDOCA_ASTGEN_IR_TO_AST_UTIL_H_
#define MALDOCA_ASTGEN_IR_TO_AST_UTIL_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "maldoca/base/status_macros.h"

namespace maldoca {

template <typename OpT, typename T>
class ToOpConverter {
 public:
  // Specifically ignoring `explicit` to allow for implicit construction in
  // `ToOpConverter`.
  // NOLINTNEXTLINE(google-explicit-constructor)
  ToOpConverter(std::function<absl::StatusOr<T>(OpT)> typed_op_converter)
      : typed_op_converter_(std::move(typed_op_converter)),
        match_op_([](mlir::Operation* op) {
          if constexpr (std::is_same_v<OpT, mlir::Operation*>) {
            return true;
          } else {
            return llvm::isa<OpT>(op);
          }
        }) {}

  explicit ToOpConverter(
      std::function<absl::StatusOr<T>(OpT)> typed_op_converter,
      std::function<absl::StatusOr<T>(mlir::Operation*)> op_converter)
      : typed_op_converter_(std::move(typed_op_converter)),
        op_converter_(std::move(op_converter)),
        match_op_([](mlir::Operation* op) {
          if constexpr (std::is_same_v<OpT, mlir::Operation*>) {
            return true;
          } else {
            return llvm::isa<OpT>(op);
          }
        }) {}

  explicit ToOpConverter(
      std::function<absl::StatusOr<T>(OpT)> typed_op_converter,
      std::function<bool(mlir::Operation*)> match_op)
      : typed_op_converter_(std::move(typed_op_converter)),
        match_op_(std::move(match_op)) {}

  explicit ToOpConverter(
      std::function<absl::StatusOr<T>(OpT)> typed_op_converter,
      std::function<absl::StatusOr<T>(mlir::Operation*)> op_converter,
      std::function<bool(mlir::Operation*)> match_op)
      : typed_op_converter_(std::move(typed_op_converter)),
        op_converter_(std::move(op_converter)),
        match_op_(std::move(match_op)) {}

  absl::StatusOr<T> operator()(OpT op) const { return typed_op_converter_(op); }

  // Only enable this overload if OpT is NOT mlir::Operation*
  template <typename U = OpT,
            std::enable_if_t<!std::is_same_v<U, mlir::Operation*>, bool> = true>
  absl::StatusOr<T> operator()(mlir::Operation* op) const {
    if (op_converter_ != nullptr) {
      return op_converter_(op);
    }

    if (op == nullptr) {
      return absl::InvalidArgumentError("Input is null.");
    }

    auto typed_op = llvm::dyn_cast<OpT>(op);
    if (typed_op == nullptr) {
      return absl::InvalidArgumentError("Input is not of type OpT.");
    }

    return typed_op_converter_(typed_op);
  }

  absl::StatusOr<T> operator()(mlir::Value value) const {
    if (value == nullptr) {
      return (*this)(static_cast<mlir::Operation*>(nullptr));
    }
    return (*this)(value.getDefiningOp());
  }

  bool MatchOp(mlir::Operation* op) const { return match_op_(op); }

 private:
  std::function<absl::StatusOr<T>(OpT)> typed_op_converter_;
  std::function<absl::StatusOr<T>(mlir::Operation*)> op_converter_ = nullptr;
  std::function<bool(mlir::Operation*)> match_op_;
};

// Deduction guide for ToOpConverter to allow implicit deduction from function
// pointers.
template <typename OpT, typename T>
ToOpConverter(absl::StatusOr<T> (*)(OpT)) -> ToOpConverter<OpT, T>;

template <typename AttrT, typename T>
class ToAttrConverter {
 public:
  // Specifically ignoring `explicit` to allow for implicit construction in
  // `ToAttrConverter`.
  // NOLINTNEXTLINE(google-explicit-constructor)
  ToAttrConverter(std::function<absl::StatusOr<T>(AttrT)> typed_attr_converter)
      : typed_attr_converter_(std::move(typed_attr_converter)) {}

  explicit ToAttrConverter(
      std::function<absl::StatusOr<T>(AttrT)> typed_attr_converter,
      std::function<absl::StatusOr<T>(mlir::Attribute)> attr_converter)
      : typed_attr_converter_(std::move(typed_attr_converter)),
        attr_converter_(std::move(attr_converter)) {}

  absl::StatusOr<T> operator()(AttrT attr) const {
    return typed_attr_converter_(attr);
  }

  // Only enable this overload if AttrT is NOT mlir::Attribute
  template <typename U = AttrT,
            std::enable_if_t<!std::is_same_v<U, mlir::Attribute>, bool> = true>
  absl::StatusOr<T> operator()(mlir::Attribute attr) const {
    if (attr_converter_ != nullptr) {
      return attr_converter_(attr);
    }

    if (attr == nullptr) {
      return absl::InvalidArgumentError("Input is null.");
    }

    auto typed_attr = llvm::dyn_cast<AttrT>(attr);
    if (typed_attr == nullptr) {
      return absl::InvalidArgumentError("Input is not of type AttrT.");
    }

    return typed_attr_converter_(typed_attr);
  }

 private:
  std::function<absl::StatusOr<T>(AttrT)> typed_attr_converter_;
  std::function<absl::StatusOr<T>(mlir::Attribute)> attr_converter_ = nullptr;
};

// Deduction guide for ToAttrConverter to allow implicit deduction from function
// pointers.
template <typename AttrT, typename T>
ToAttrConverter(absl::StatusOr<T> (*)(AttrT)) -> ToAttrConverter<AttrT, T>;

template <typename T>
class ToOpsConverter {
 public:
  ToOpsConverter(std::function<absl::StatusOr<std::vector<T>>(mlir::ValueRange)>
                     values_converter,
                 std::function<absl::StatusOr<std::vector<T>>(mlir::Block*)>
                     block_converter)
      : values_converter_(std::move(values_converter)),
        block_converter_(std::move(block_converter)) {}

  absl::StatusOr<std::vector<T>> operator()(mlir::ValueRange values) const {
    if (values_converter_ == nullptr) {
      return absl::UnimplementedError("values_converter_ is null.");
    }
    return values_converter_(values);
  }

  absl::StatusOr<std::vector<T>> operator()(mlir::Block* block) const {
    if (block_converter_ == nullptr) {
      return absl::UnimplementedError("block_converter_ is null.");
    }
    return block_converter_(block);
  }

 private:
  std::function<absl::StatusOr<std::vector<T>>(mlir::ValueRange)>
      values_converter_ = nullptr;
  std::function<absl::StatusOr<std::vector<T>>(mlir::Block*)> block_converter_ =
      nullptr;
};

template <typename T>
class ToRegionConverter {
 public:
  // Specifically ignoring `explicit` to allow for implicit construction in
  // `ToRegionConverter`.
  // NOLINTNEXTLINE(google-explicit-constructor)
  ToRegionConverter(std::function<absl::StatusOr<T>(mlir::Region&)> converter)
      : converter_(std::move(converter)) {}

  absl::StatusOr<T> operator()(mlir::Region& region) const {
    if (converter_ == nullptr) {
      return absl::UnimplementedError("converter_ is null.");
    }
    return converter_(region);
  }

 private:
  std::function<absl::StatusOr<T>(mlir::Region&)> converter_ = nullptr;
};

template <typename... OpT, typename... T>
ToOpConverter<mlir::Operation*, std::variant<T...>> OpVariant(
    ToOpConverter<OpT, T>... converters) {
  return ToOpConverter<mlir::Operation*, std::variant<T...>>(
      [=](mlir::Operation* op) -> absl::StatusOr<std::variant<T...>> {
        std::optional<std::variant<T...>> result;
        bool matched = false;
        absl::Status last_error =
            absl::InvalidArgumentError("No variant matched.");
        (
            [&] {
              if (!matched) {
                auto val_or = converters(op);
                if (val_or.ok()) {
                  matched = true;
                  result = std::variant<T...>(*std::move(val_or));
                } else if (!absl::IsInvalidArgument(val_or.status())) {
                  last_error = val_or.status();
                }
              }
            }(),
            ...);
        if (!result.has_value()) {
          return last_error;
        }
        return *std::move(result);
      },
      /*op_converter=*/nullptr,
      /*match_op=*/
      [=](mlir::Operation* op) { return (converters.MatchOp(op) || ...); });
}

template <typename AttrT, typename T>
ToAttrConverter<AttrT, std::optional<T>> Nullable(
    ToAttrConverter<AttrT, T> converter) {
  return ToAttrConverter<AttrT, std::optional<T>>{
      /*typed_attr_converter=*/
      [=](AttrT attr) -> absl::StatusOr<std::optional<T>> {
        if (attr == nullptr) {
          return std::nullopt;
        }
        return converter(attr);
      },
      /*attr_converter=*/
      [=](mlir::Attribute attr) -> absl::StatusOr<std::optional<T>> {
        if (attr == nullptr) {
          return std::nullopt;
        }
        return converter(attr);
      }};
}

template <typename OpT, typename T>
ToOpConverter<OpT, std::optional<T>> Nullable(ToOpConverter<OpT, T> converter) {
  return ToOpConverter<OpT, std::optional<T>>{
      /*typed_op_converter=*/
      [=](OpT op) -> absl::StatusOr<std::optional<T>> {
        if (op == nullptr) {
          return std::nullopt;
        }
        return converter(op);
      },
      /*op_converter=*/
      [=](mlir::Operation* op) -> absl::StatusOr<std::optional<T>> {
        if (op == nullptr) {
          return std::nullopt;
        }
        return converter(op);
      },
      /*match_op=*/
      [=](mlir::Operation* op) {
        return op == nullptr || converter.MatchOp(op);
      }};
}

template <typename NoneOp, typename OpT, typename T>
ToOpConverter<OpT, std::optional<T>> Nullable(ToOpConverter<OpT, T> converter) {
  return ToOpConverter<OpT, std::optional<T>>{
      /*typed_op_converter=*/
      [=](OpT op) -> absl::StatusOr<std::optional<T>> {
        if (op == nullptr) {
          return std::nullopt;
        }
        if (llvm::isa<NoneOp>(op)) {
          return std::nullopt;
        }
        return converter(op);
      },
      /*op_converter=*/
      [=](mlir::Operation* op) -> absl::StatusOr<std::optional<T>> {
        if (op == nullptr) {
          return std::nullopt;
        }
        if (llvm::isa<NoneOp>(op)) {
          return std::nullopt;
        }
        return converter(op);
      },
      /*match_op=*/
      [=](mlir::Operation* op) {
        return op == nullptr || llvm::isa<NoneOp>(op) || converter.MatchOp(op);
      }};
}

template <typename T>
ToRegionConverter<std::optional<T>> Nullable(ToRegionConverter<T> converter) {
  return ToRegionConverter<std::optional<T>>(
      [=](mlir::Region& region) -> absl::StatusOr<std::optional<T>> {
        if (region.empty()) {
          return std::nullopt;
        }
        return converter(region);
      });
}

template <typename EndOpT>
absl::StatusOr<mlir::Value> GetExprRegionValue(mlir::Region& region) {
  if (!region.hasOneBlock()) {
    return absl::InvalidArgumentError("Region should have exactly one block.");
  }
  mlir::Block& block = region.front();
  if (block.empty()) {
    return absl::InvalidArgumentError("Block cannot be empty.");
  }
  auto expr_region_end = llvm::dyn_cast<EndOpT>(block.back());
  if (expr_region_end == nullptr) {
    return absl::InvalidArgumentError(
        "Block should end with expected terminator.");
  }
  return expr_region_end.getArgument();
}

template <typename EndOpT>
absl::StatusOr<mlir::ValueRange> GetExprsRegionValues(mlir::Region& region) {
  if (!region.hasOneBlock()) {
    return absl::InvalidArgumentError("Region should have exactly one block.");
  }
  mlir::Block& block = region.front();
  if (block.empty()) {
    return absl::InvalidArgumentError("Block cannot be empty.");
  }
  auto exprs_region_end = llvm::dyn_cast<EndOpT>(block.back());
  if (exprs_region_end == nullptr) {
    return absl::InvalidArgumentError(
        "Block should end with expected terminator.");
  }
  return exprs_region_end.getArguments();
}

inline absl::StatusOr<mlir::Operation*> GetStmtRegionOperation(
    mlir::Region& region) {
  if (!region.hasOneBlock()) {
    return absl::InvalidArgumentError("Region should have exactly one block.");
  }
  mlir::Block& block = region.front();
  if (block.empty()) {
    return absl::InvalidArgumentError("Block cannot be empty.");
  }
  return &block.back();
}

inline absl::StatusOr<mlir::Block*> GetStmtsRegionBlock(mlir::Region& region) {
  if (!region.hasOneBlock()) {
    return absl::InvalidArgumentError("Region should have exactly one block.");
  }
  mlir::Block& block = region.front();
  return &block;
}

template <typename EndOpT, typename OpT, typename T>
ToRegionConverter<T> ExprRegion(ToOpConverter<OpT, T> converter) {
  return ToRegionConverter<T>([=](mlir::Region& region) -> absl::StatusOr<T> {
    MALDOCA_ASSIGN_OR_RETURN(auto val, GetExprRegionValue<EndOpT>(region));
    return Convert(val, converter);
  });
}

template <typename EndOpT, typename T>
ToRegionConverter<std::vector<T>> ExprsRegion(ToOpsConverter<T> converter) {
  return ToRegionConverter<std::vector<T>>(
      [=](mlir::Region& region) -> absl::StatusOr<std::vector<T>> {
        MALDOCA_ASSIGN_OR_RETURN(auto vals,
                                 GetExprsRegionValues<EndOpT>(region));
        return Convert(vals, converter);
      });
}

template <typename OpT, typename T>
ToRegionConverter<T> StmtRegion(ToOpConverter<OpT, T> converter) {
  return ToRegionConverter<T>([=](mlir::Region& region) -> absl::StatusOr<T> {
    MALDOCA_ASSIGN_OR_RETURN(auto op, GetStmtRegionOperation(region));
    return Convert(op, converter);
  });
}

template <typename T>
ToRegionConverter<std::vector<T>> StmtsRegion(ToOpsConverter<T> converter) {
  return ToRegionConverter<std::vector<T>>(
      [=](mlir::Region& region) -> absl::StatusOr<std::vector<T>> {
        MALDOCA_ASSIGN_OR_RETURN(auto block, GetStmtsRegionBlock(region));
        return Convert(block, converter);
      });
}

template <typename T>
inline ToAttrConverter<mlir::StringAttr, T> Enum(
    std::function<absl::StatusOr<T>(absl::string_view)> string_to_enum) {
  return ToAttrConverter<mlir::StringAttr, T>(
      [=](mlir::StringAttr attr) -> absl::StatusOr<T> {
        return string_to_enum(attr.str());
      });
}

template <typename... AttrT, typename... T>
ToAttrConverter<mlir::Attribute, std::variant<T...>> AttrVariant(
    ToAttrConverter<AttrT, T>... converters) {
  return ToAttrConverter<mlir::Attribute, std::variant<T...>>(
      [=](mlir::Attribute attr) -> absl::StatusOr<std::variant<T...>> {
        std::optional<std::variant<T...>> result;
        bool matched = false;
        absl::Status last_error =
            absl::InvalidArgumentError("No variant matched.");
        (
            [&] {
              if (!matched) {
                auto val_or = converters(attr);
                if (val_or.ok()) {
                  matched = true;
                  result = std::variant<T...>(*std::move(val_or));
                } else if (!absl::IsInvalidArgument(val_or.status())) {
                  last_error = val_or.status();
                }
              }
            }(),
            ...);
        if (!result.has_value()) {
          return last_error;
        }
        return *std::move(result);
      });
}

template <typename AttrT, typename T>
ToAttrConverter<mlir::ArrayAttr, std::vector<T>> List(
    ToAttrConverter<AttrT, T> elem_converter) {
  return ToAttrConverter<mlir::ArrayAttr, std::vector<T>>(
      [=](mlir::ArrayAttr attr) -> absl::StatusOr<std::vector<T>> {
        std::vector<T> result;
        result.reserve(attr.size());
        for (mlir::Attribute element : attr.getValue()) {
          MALDOCA_ASSIGN_OR_RETURN(T value, elem_converter(element));
          result.push_back(std::move(value));
        }
        return result;
      });
}

template <typename OpT, typename T>
ToOpsConverter<T> List(ToOpConverter<OpT, T> elem_converter) {
  return ToOpsConverter<T>(
      /*values_converter=*/
      [=](mlir::ValueRange values) -> absl::StatusOr<std::vector<T>> {
        std::vector<T> result;
        result.reserve(values.size());
        for (mlir::Value element : values) {
          MALDOCA_ASSIGN_OR_RETURN(T value, elem_converter(element));
          result.push_back(std::move(value));
        }
        return result;
      },
      /*block_converter=*/
      [=](mlir::Block* block) -> absl::StatusOr<std::vector<T>> {
        std::vector<T> result;
        if (block == nullptr) {
          return result;
        }
        for (mlir::Operation& element : *block) {
          if (element.hasTrait<mlir::OpTrait::IsTerminator>()) {
            continue;
          }
          if (!elem_converter.MatchOp(&element)) {
            continue;
          }
          auto casted = llvm::dyn_cast<OpT>(&element);
          if (!casted) {
            continue;
          }
          MALDOCA_ASSIGN_OR_RETURN(T value, elem_converter(casted));
          result.push_back(std::move(value));
        }
        return result;
      });
}

template <typename T>
absl::StatusOr<std::vector<T>> Convert(mlir::ValueRange values,
                                       const ToOpsConverter<T>& converter) {
  return converter(values);
}

template <typename OpT, typename T>
absl::StatusOr<T> Convert(mlir::Value value,
                          const ToOpConverter<OpT, T>& converter) {
  return converter(value);
}

template <typename T>
absl::StatusOr<std::vector<T>> Convert(mlir::Block* block,
                                       const ToOpsConverter<T>& converter) {
  return converter(block);
}

template <typename T>
absl::StatusOr<T> Convert(mlir::Region& region,
                          const ToRegionConverter<T>& converter) {
  return converter(region);
}

template <typename OpT, typename T>
absl::StatusOr<T> Convert(mlir::Operation* op,
                          ToOpConverter<OpT, T> converter) {
  return converter(op);
}

template <typename AttrT, typename T>
absl::StatusOr<T> Convert(mlir::Attribute attr,
                          ToAttrConverter<AttrT, T> converter) {
  return converter(attr);
}

// Attr converters
inline bool IsBool(mlir::Attribute attr) {
  return llvm::isa<mlir::BoolAttr>(attr);
}
inline ToAttrConverter<mlir::BoolAttr, bool> ToBool() {
  return ToAttrConverter<mlir::BoolAttr, bool>(
      [](mlir::BoolAttr attr) -> absl::StatusOr<bool> {
        return attr.getValue();
      });
}
inline bool IsString(mlir::Attribute attr) {
  return llvm::isa<mlir::StringAttr>(attr);
}
inline ToAttrConverter<mlir::StringAttr, std::string> ToString() {
  return ToAttrConverter<mlir::StringAttr, std::string>(
      [](mlir::StringAttr attr) -> absl::StatusOr<std::string> {
        return attr.str();
      });
}
inline bool IsDouble(mlir::Attribute attr) {
  return llvm::isa<mlir::FloatAttr>(attr);
}
inline ToAttrConverter<mlir::FloatAttr, double> ToDouble() {
  return ToAttrConverter<mlir::FloatAttr, double>(
      [](mlir::FloatAttr attr) -> absl::StatusOr<double> {
        return attr.getValueAsDouble();
      });
}
inline bool IsInt64(mlir::Attribute attr) {
  return llvm::isa<mlir::IntegerAttr>(attr);
}
inline ToAttrConverter<mlir::IntegerAttr, int64_t> ToInt64() {
  return ToAttrConverter<mlir::IntegerAttr, int64_t>(
      [](mlir::IntegerAttr attr) -> absl::StatusOr<int64_t> {
        return attr.getValue().getSExtValue();
      });
}

}  // namespace maldoca

#endif  // MALDOCA_ASTGEN_IR_TO_AST_UTIL_H_
