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

#include "maldoca/base/get_runfiles_dir.h"

#include <string>

#include "rules_cc/cc/runfiles/runfiles.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "maldoca/base/path.h"

namespace maldoca {

std::string GetDataDependencyFilepath(absl::string_view path) {
  using ::rules_cc::cc::runfiles::Runfiles;
  std::string error;
  std::unique_ptr<Runfiles> runfiles(
      Runfiles::CreateForTest(BAZEL_CURRENT_REPOSITORY, &error));
  CHECK(runfiles != nullptr) << error;
  return runfiles->Rlocation(JoinPath("com_google_maldoca", path));
}

}  // namespace maldoca
