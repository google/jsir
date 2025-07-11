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

// Converts a JavaScript file into JSIR. Outputs the IR to stdout.

// Usage:
//
// bazel run //maldoca/js/ir:jsir_gen -- \
//   --input_file=test.js \
//   --output_dialect=jshir

#include <algorithm>
#include <iostream>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "maldoca/base/filesystem.h"
#include "maldoca/js/driver/driver.pb.h"
#include "maldoca/js/ir/jsir_gen_lib.h"
#include "maldoca/js/quickjs_babel/quickjs_babel.h"

// Maps pass string to enum.
static auto *kStringToPassKind =
    new absl::flat_hash_map<std::string, maldoca::JsirPassKind>{
        {"source2ast", maldoca::JsirPassKind::kSourceToAst},
        {"ast2hir", maldoca::JsirPassKind::kAstToJshir},
        {"hir2lir", maldoca::JsirPassKind::kJshirToJslir},
        {"lir2hir", maldoca::JsirPassKind::kJslirToJshir},
        {"hir2ast", maldoca::JsirPassKind::kJshirToAst},
        {"ast2source", maldoca::JsirPassKind::kAstToSource},

        {"erase_comments", maldoca::JsirPassKind::kEraseComments},
        {"constprop", maldoca::JsirPassKind::kConstantPropagation},
        {"movenamedfuncs", maldoca::JsirPassKind::kMoveNamedFunctions},
        {"normalizeobjprops",
         maldoca::JsirPassKind::kNormalizeObjectProperties},
        {"peelparens", maldoca::JsirPassKind::kPeelParentheses},
        {"split_sequence_expressions",
         maldoca::JsirPassKind::kSplitSequenceExpressions},
        {"remove_directives", maldoca::JsirPassKind::kRemoveDirectives},
        {"split_declaration_statements",
         maldoca::JsirPassKind::kSplitDeclarationStatements},
    };

ABSL_FLAG(std::string, input_file, "", "The JavaScript file.");
ABSL_FLAG(std::string, output_file, "", "The output file.");
ABSL_FLAG(std::string, jsir_analysis, {}, "The JSIR analysis to run.");
ABSL_FLAG(
    std::vector<std::string>, passes, {},
    absl::StrCat("The passes to run. Available passes: ", []() -> std::string {
      std::vector<std::string> available_passes;
      for (const auto &[pass, pass_kind] : *kStringToPassKind) {
        available_passes.push_back(pass);
      }
      return absl::StrJoin(available_passes, ", ");
    }()));

namespace maldoca {

static std::optional<JsirAnalysisConfig::KindCase> StringToJsirAnalysisKind(
    absl::string_view kind) {
  static const auto *kMap =
      new absl::flat_hash_map<std::string, JsirAnalysisConfig::KindCase>{
          {"constant_propagation", JsirAnalysisConfig::kConstantPropagation},
      };

  auto it = kMap->find(kind);
  if (it != kMap->end()) {
    return it->second;
  }

  return std::nullopt;
}

static JsirAnalysisConfig GetJsirAnalysisConfig() {
  auto analysis = absl::GetFlag(FLAGS_jsir_analysis);

  if (analysis.empty()) {
    return JsirAnalysisConfig();
  }

  std::optional<JsirAnalysisConfig::KindCase> kind =
      StringToJsirAnalysisKind(analysis);

  CHECK(kind.has_value()) << "Unknown JsirAnalysisConfig: " << analysis;

  switch (*kind) {
    case JsirAnalysisConfig::kConstantPropagation: {
      JsirAnalysisConfig config;
      config.mutable_constant_propagation();
      return config;
    }

    case JsirAnalysisConfig::KIND_NOT_SET:
      LOG(FATAL) << "JsirAnalysisConfig KIND_NOT_SET";
  }

  LOG(FATAL) << "Unknown JsirAnalysisConfig: " << analysis;
}

}  // namespace maldoca

int main(int argc, char *argv[]) {
  absl::ParseCommandLine(argc, argv);

  auto input_file = absl::GetFlag(FLAGS_input_file);
  auto input = maldoca::GetFileContents(input_file);
  if (!input.ok()) {
    std::cerr << input.status().ToString() << std::endl;
    return 1;
  }

  auto passes = absl::GetFlag(FLAGS_passes);
  std::vector<maldoca::JsirPassKind> pass_kinds;
  std::for_each(passes.begin(), passes.end(),
                [&pass_kinds](absl::string_view pass) {
                  auto it = kStringToPassKind->find(pass);
                  CHECK(it != kStringToPassKind->end());
                  pass_kinds.emplace_back(it->second);
                });

  maldoca::QuickJsBabel babel;

  maldoca::JsirAnalysisConfig analysis_config =
      maldoca::GetJsirAnalysisConfig();

  std::vector<maldoca::JsirTransformConfig> transform_configs;

  auto output = maldoca::JsirGen(babel, *input, pass_kinds, analysis_config,
                                 transform_configs);
  if (!output.ok()) {
    std::cerr << output.status().ToString() << std::endl;
    return 1;
  }

  std::string output_file = absl::GetFlag(FLAGS_output_file);
  if (!output_file.empty()) {
    CHECK_OK(maldoca::SetFileContents(output_file, output->repr));
  } else {
    std::cout << output->repr << std::endl;
  }

  for (const auto &analysis_output : output->analysis_outputs.outputs()) {
    std::cout << DumpJsAnalysisOutput(*input, analysis_output) << std::endl;
  }

  return 0;
}
