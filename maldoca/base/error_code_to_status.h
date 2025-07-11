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

#ifndef MALDOCA_BASE_ERROR_CODE_TO_STATUS_H_
#define MALDOCA_BASE_ERROR_CODE_TO_STATUS_H_

#include <system_error>  // NOLINT(build/c++11): open source

#include "absl/status/status.h"
#include "maldoca/base/status_builder.h"

namespace maldoca {

absl::StatusCode ErrorCodeToStatusCode(const std::error_code& ec);

StatusBuilder ErrorCodeToStatus(const std::error_code& ec);

// Converts an `errno` value into an absl::Status.
StatusBuilder ErrnoToStatus(int errno_value);

}  // namespace maldoca

#endif  // MALDOCA_BASE_ERROR_CODE_TO_STATUS_H_
