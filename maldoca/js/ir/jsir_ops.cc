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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// IWYU pragma: begin_keep

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "maldoca/js/ir/ir.h"
#include "maldoca/js/quickjs/quickjs.h"
#include "quickjs/quickjs.h"

// IWYU pragma: end_keep

#define GET_OP_CLASSES
#include "maldoca/js/ir/jsir_ops.cc.inc"

namespace maldoca {

std::optional<QjsValue> MlirAttributeToQuickJsValue(JSContext *qjs_context,
                                                    mlir::Attribute val) {
  if (val == nullptr) {
    return std::nullopt;
  }
  return llvm::TypeSwitch<mlir::Attribute, std::optional<QjsValue>>(val)
      .Case([&](JsirNullLiteralAttr val) {
        return QjsValue{qjs_context, JS_NULL};
      })
      .Case([&](mlir::FloatAttr val) {
        return QjsValue{
            qjs_context,
            JS_NewFloat64(qjs_context, val.getValue().convertToDouble()),
        };
      })
      .Case([&](JsirNumericLiteralAttr val) {
        return MlirAttributeToQuickJsValue(qjs_context, val.getValue());
      })
      .Case([&](mlir::StringAttr val) {
        return QjsValue{
            qjs_context,
            JS_NewStringLen(qjs_context, val.data(), val.size()),
        };
      })
      .Case([&](JsirStringLiteralAttr val) {
        return MlirAttributeToQuickJsValue(qjs_context, val.getValue());
      })
      .Case([&](mlir::BoolAttr val) {
        return QjsValue{qjs_context, JS_NewBool(qjs_context, val.getValue())};
      })
      .Case([&](JsirBigIntLiteralAttr val) -> std::optional<QjsValue> {
        // Run the JavaScript code:
        //
        // ```
        // BigInt(val_str)
        // ```

        QjsValue global{qjs_context, JS_GetGlobalObject(qjs_context)};

        QjsValue qjs_big_int_ctor{
            qjs_context,
            JS_GetPropertyStr(qjs_context, global.get(), "BigInt"),
        };
        if (!JS_IsFunction(qjs_context, qjs_big_int_ctor.get())) {
          return std::nullopt;
        }

        std::string val_str = val.getValue().str();
        QjsValue qjs_val_str{
            qjs_context,
            JS_NewStringLen(qjs_context, val_str.data(), val_str.size()),
        };

        std::vector<JSValue> argv = {qjs_val_str.get()};
        return QjsValue{
            qjs_context,
            JS_Call(qjs_context, /*func_obj=*/qjs_big_int_ctor.get(),
                    /*this_obj=*/global.get(), argv.size(), argv.data()),
        };
      })
      .Default([&](mlir::Attribute val) { return std::nullopt; });
}

std::optional<mlir::Attribute> QuickJsValueToMlirAttribute(
    JSContext *qjs_context, mlir::MLIRContext *context, JSValue val) {
  mlir::OpBuilder builder(context);
  if (JS_IsNull(val)) {
    return JsirNullLiteralAttr::get(context);
  }

  if (JS_IsBool(val)) {
    return builder.getBoolAttr(JS_ToBool(qjs_context, val));
  }

  if (JS_IsNumber(val)) {
    double double_value;

    // JS_TAG_INT or JS_TAG_FLOAT64
    int tag = JS_VALUE_GET_TAG(val);
    if (tag == JS_TAG_INT) {
      int32_t int32_val;
      if (JS_ToInt32(qjs_context, &int32_val, val) != 0) {
        LOG(ERROR) << "JS_ToInt32 returned an exception?";
        return std::nullopt;
      }
      double_value = int32_val;
    } else if (JS_TAG_IS_FLOAT64(tag)) {
      if (JS_ToFloat64(qjs_context, &double_value, val) != 0) {
        return std::nullopt;
      }
    } else {
      return std::nullopt;
    }
    return builder.getF64FloatAttr(double_value);
  }

  if (JS_IsString(val)) {
    size_t len;
    const char *cstr = JS_ToCStringLen(qjs_context, &len, val);
    absl::Cleanup cleanup_cstr = [&] { JS_FreeCString(qjs_context, cstr); };

    std::string s{cstr, len};
    return builder.getStringAttr(s);
  }

  if (JS_IsBigInt(qjs_context, val)) {
    size_t len;
    const char *cstr = JS_ToCStringLen(qjs_context, &len, val);
    absl::Cleanup cleanup_cstr = [&] { JS_FreeCString(qjs_context, cstr); };

    std::string s{cstr, len};
    return builder.getAttr<JsirBigIntLiteralAttr>(builder.getStringAttr(s),
                                                  /*extra=*/nullptr);
  }

  return std::nullopt;
}

QjsValue EmulateBinOp(JSContext *qjs_context, std::string op, QjsValue left,
                      QjsValue right) {
  constexpr char kSourceTemplate[] = R"js(
        (function(a, b) {
          return (a) %s (b);
        })
      )js";

  QjsValue qjs_global{qjs_context, JS_GetGlobalObject(qjs_context)};
  std::string source = absl::StrFormat(kSourceTemplate, op);

  QjsValue evaluate_bin_op{
      qjs_context,
      JS_Eval(qjs_context, source.data(), source.length(), "evaluate_bin_op.js",
              JS_EVAL_TYPE_GLOBAL),
  };

  std::vector<JSValue> args = {left.get(), right.get()};
  QjsValue bin_op_result{
      qjs_context,
      JS_Call(qjs_context, evaluate_bin_op.get(), qjs_global.get(), args.size(),
              args.data()),
  };
  return bin_op_result;
}

std::optional<mlir::Attribute> EmulateBinOp(std::string op,
                                            mlir::MLIRContext *context,
                                            mlir::Attribute mlir_left,
                                            mlir::Attribute mlir_right) {
  std::unique_ptr<JSRuntime, QjsRuntimeDeleter> qjs_runtime{JS_NewRuntime()};
  std::unique_ptr<JSContext, QjsContextDeleter> qjs_context{
      JS_NewContext(qjs_runtime.get())};
  auto left = MlirAttributeToQuickJsValue(qjs_context.get(), mlir_left);
  auto right = MlirAttributeToQuickJsValue(qjs_context.get(), mlir_right);
  if (!left.has_value() || !right.has_value()) {
    return std::nullopt;
  }

  QjsValue bin_op_result = EmulateBinOp(qjs_context.get(), op, *left, *right);

  if (JS_IsException(bin_op_result.get())) {
    LOG(ERROR) << "evaluate_bin_op returned an exception?";
    return std::nullopt;
  }

  return QuickJsValueToMlirAttribute(qjs_context.get(), context,
                                     bin_op_result.get());
}

QjsValue EmulateUnaryOp(JSContext *qjs_context, std::string op,
                        QjsValue operand) {
  QjsValue qjs_global{qjs_context, JS_GetGlobalObject(qjs_context)};

  static constexpr char kSourceTemplate[] = R"js(
    function evaluate_unary_op(a) {
      return %s (a);
    }
    evaluate_unary_op
  )js";
  std::string source = absl::StrFormat(kSourceTemplate, op);
  QjsValue evaluate_unary_op{
      qjs_context,
      JS_Eval(qjs_context, source.data(), source.length(),
              "evaluate_unary_op.js", JS_EVAL_TYPE_GLOBAL),
  };

  std::vector<JSValue> args = {operand.get()};
  QjsValue unary_op_result{
      qjs_context,
      JS_Call(qjs_context, evaluate_unary_op.get(), qjs_global.get(),
              args.size(), args.data()),
  };

  return unary_op_result;
}

std::optional<mlir::Attribute> EmulateUnaryOp(std::string op,
                                              mlir::MLIRContext *context,
                                              mlir::Attribute mlir_operand) {
  std::unique_ptr<JSRuntime, QjsRuntimeDeleter> qjs_runtime{JS_NewRuntime()};
  std::unique_ptr<JSContext, QjsContextDeleter> qjs_context{
      JS_NewContext(qjs_runtime.get())};

  auto expr = MlirAttributeToQuickJsValue(qjs_context.get(), mlir_operand);
  if (!expr.has_value()) {
    return std::nullopt;
  }

  QjsValue unary_op_result = EmulateUnaryOp(qjs_context.get(), op, *expr);

  if (JS_IsException(unary_op_result.get())) {
    LOG(ERROR) << "evaluate_unary_op returned an exception?";
    return std::nullopt;
  }

  return QuickJsValueToMlirAttribute(qjs_context.get(), context,
                                     unary_op_result.get());
}

std::optional<mlir::Attribute> EmulateMemberAccessOp(mlir::MLIRContext *context,
                                                     mlir::Attribute mlir_obj,
                                                     mlir::Attribute mlir_key) {
  std::unique_ptr<JSRuntime, QjsRuntimeDeleter> qjs_runtime{JS_NewRuntime()};
  std::unique_ptr<JSContext, QjsContextDeleter> qjs_context{
      JS_NewContext(qjs_runtime.get())};
  QjsValue global{qjs_context.get(), JS_GetGlobalObject(qjs_context.get())};

  static constexpr absl::string_view kSource = R"js(
    function evaluate_member_access(a, b) {
      return a[b];
    }
    evaluate_member_access
  )js";

  QjsValue evaluate_member_access{
      qjs_context.get(),
      JS_Eval(qjs_context.get(), kSource.data(), kSource.length(),
              "evaluate_member_access.js", JS_EVAL_TYPE_GLOBAL),
  };

  if (!JS_IsFunction(qjs_context.get(), evaluate_member_access.get())) {
    LOG(FATAL) << "evaluate_member_access is not a function?";
    return std::nullopt;
  }

  auto obj = MlirAttributeToQuickJsValue(qjs_context.get(), mlir_obj);
  auto key = MlirAttributeToQuickJsValue(qjs_context.get(), mlir_key);
  if (!obj.has_value() || !key.has_value()) {
    return std::nullopt;
  }
  std::vector<JSValue> args = {obj->get(), key->get()};
  QjsValue member_access_result{
      qjs_context.get(),
      JS_Call(qjs_context.get(), evaluate_member_access.get(), global.get(),
              args.size(), args.data()),
  };

  if (JS_IsException(member_access_result.get())) {
    LOG(ERROR) << "evaluate_member_access returned an exception?";
    return std::nullopt;
  }

  return QuickJsValueToMlirAttribute(qjs_context.get(), context,
                                     member_access_result.get());
}

mlir::OpFoldResult JsirUnaryExpressionOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  auto ans = EmulateUnaryOp(getOperator_().str(), getContext(), operands[0]);
  return ans.value_or(nullptr);
}

mlir::OpFoldResult JsirBinaryExpressionOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  auto ans = EmulateBinOp(getOperator_().str(), getContext(), operands[0],
                          operands[1]);
  return ans.value_or(nullptr);
}

mlir::OpFoldResult JsirNullLiteralOp::fold(FoldAdaptor adaptor) {
  return JsirNullLiteralAttr::get(getContext());
}

mlir::OpFoldResult JsirRegExpLiteralOp::fold(FoldAdaptor adaptor) {
  return JsirRegExpLiteralAttr::get(getContext(), getPatternAttr(),
                                    getFlagsAttr());
}

mlir::OpFoldResult JsirStringLiteralOp::fold(FoldAdaptor adaptor) {
  return getValueAttr();
}

mlir::OpFoldResult JsirBooleanLiteralOp::fold(FoldAdaptor adaptor) {
  return getValueAttr();
}

mlir::OpFoldResult JsirNumericLiteralOp::fold(FoldAdaptor adaptor) {
  return getValueAttr();
}

mlir::OpFoldResult JsirBigIntLiteralOp::fold(FoldAdaptor adaptor) {
  mlir::Builder builder(getContext());
  return builder.getAttr<JsirBigIntLiteralAttr>(getValueAttr(), getExtraAttr());
}

mlir::OpFoldResult JsirParenthesizedExpressionOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  return operands[0];
}

mlir::OpFoldResult JsirParenthesizedExpressionRefOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  return operands[0];
}

mlir::OpFoldResult JsirMemberExpressionOp::fold(FoldAdaptor adaptor) {
  mlir::Attribute object = adaptor.getObject();

  mlir::Attribute computed_property;
  if (getLiteralProperty().has_value()) {
    mlir::Attribute literal_property = getLiteralPropertyAttr();
    if (auto identifier =
            mlir::dyn_cast<JsirIdentifierAttr>(literal_property)) {
      computed_property = identifier.getName();
    } else if (auto private_name =
                   mlir::dyn_cast<JsirPrivateNameAttr>(literal_property)) {
      emitWarning("a.#b not supported yet");
      return nullptr;
    } else {
      emitOpError(
          "invalid JsirMemberExpressionOp: literal_property must be "
          "JsirIdentifierAttr or JsirPrivateNameAttr");
      return nullptr;
    }

  } else if (getComputedProperty() != nullptr) {
    computed_property = adaptor.getComputedProperty();

  } else {
    emitOpError(
        "invalid JsirMemberExpressionOp: neither literal_property nor "
        "computed_property available");
    return nullptr;
  }

  return EmulateMemberAccessOp(getContext(), object, computed_property)
      .value_or(nullptr);
}

}  // namespace maldoca
