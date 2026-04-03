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
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "maldoca/astgen/ast_def.h"
#include "maldoca/base/path.h"
#include "google/protobuf/io/printer.h"

namespace maldoca {

inline constexpr absl::string_view kJsonValueVariableName = "json";
inline constexpr absl::string_view kOsValueVariableName = "os";

inline std::string GetAstHeaderPath(absl::string_view ast_path) {
  return JoinPath(ast_path, "ast.generated.h");
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
inline bool FieldIsArgument(const FieldDef* field) {
  return !field->ignore_in_ir() && !field->enclose_in_region();
}

inline bool FieldIsRegion(const FieldDef* field) {
  return !field->ignore_in_ir() && field->enclose_in_region();
}

}  // namespace maldoca

#endif  // MALDOCA_ASTGEN_AST_GEN_UTILS_H_
