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

#ifndef MALDOCA_BASE_FILESYSTEM_H_
#define MALDOCA_BASE_FILESYSTEM_H_

#include <filesystem>  // NOLINT: open source
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/message.h"

namespace maldoca {

// Reads and returns the contents of the file `file_name`.
//
// Typical return codes (not guaranteed exhaustive):
//  * StatusCode::kOk
//  * StatusCode::kPermissionDenied (file not readable)
//  * StatusCode::kNotFound (no such file)
//  * StatusCode::kUnknown (a Read or Open error occurred)
absl::StatusOr<std::string> GetFileContents(
    const std::filesystem::path& file_name);

// Parses a single text formatted protobuf from the given string which is
// assumed to have come from the given file. The file name is only used for
// error reporting.
//
// REQUIRES: `contents` is a single text formatted protobuf.
// REQUIRES: The proto must point to a valid object compatible with the text.
// Typical return codes (not guaranteed to be exhaustive):
//  * StatusCode::kOk
//  * StatusCode::kFailedPrecondition (the file contents couldn't be parsed as
//    text proto)
//
// *proto will hold the result of parsing the given string of the file as
// protobuf only if OK is returned.  Regardless of success, the
// contents of that protobuf may be modified.
absl::Status ParseTextProto(absl::string_view contents,
                            const std::filesystem::path& file_name,
                            google::protobuf::Message* proto);

// Reads a single text formatted protobuf from a file.
//
// REQUIRES: `file_name` can be opened for reading.
// REQUIRES: The contents of `file_name` are a single text formatted protobuf.
// REQUIRES: The proto must point to a valid object compatible with the text.
// Typical return codes (not guaranteed to be exhaustive):
//  * StatusCode::kOk
//  * StatusCode::kPermissionDenied (file not readable)
//  * StatusCode::kNotFound (no such file)
//  * StatusCode::kFailedPrecondition (the file contents couldn't be parsed as
//    text proto)
//
// *proto will hold the result of parsing the text of the file as
// protobuf only if OK is returned.  Regardless of success, the
// contents of that protobuf may be modified.
absl::Status ParseTextProtoFile(const std::filesystem::path& file_name,
                                google::protobuf::Message* proto);

// Writes the data provided in `content` to the file `file_name`, overwriting
// any existing content. Fails if directory does not exist.
//
// NOTE: Will return OK iff all of the data in `content` was written.
// May write some of the data and return an error.
//
// WARNING: The file update is NOT guaranteed to be atomic.
//
// Typical return codes (not guaranteed exhaustive):
//  * StatusCode::kOk
//  * StatusCode::kPermissionDenied (file not writable)
//  * StatusCode::kUnknown (a Write or Open error occurred)
absl::Status SetFileContents(const std::filesystem::path& file_name,
                             std::string_view content);

}  // namespace maldoca

#endif  // MALDOCA_BASE_FILESYSTEM_H_
