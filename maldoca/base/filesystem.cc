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

#include "maldoca/base/filesystem.h"

#include <fcntl.h>

#include <cerrno>
#include <filesystem>  // NOLINT: open source
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "maldoca/base/error_code_to_status.h"
#include "maldoca/base/status_builder.h"
#include "maldoca/base/status_macros.h"
#include "google/protobuf/io/tokenizer.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"

namespace maldoca {
namespace {

// Returns a Status error based on the errno value. The error message includes
// the filename.
absl::Status ErrNoToStatusWithFilename(int errno_value,
                                       const std::filesystem::path& file_name) {
  StatusBuilder builder = ErrnoToStatus(errno);
  builder << file_name.string();
  return std::move(builder);
}

// For use in reading serialized protos from files.
class ParseTextProtoFileErrorCollector : public google::protobuf::io::ErrorCollector {
 public:
  explicit ParseTextProtoFileErrorCollector(
      const std::filesystem::path& file_name, const google::protobuf::Message& proto)
      : file_name_(file_name), proto_(proto) {}

  void RecordError(int line, google::protobuf::io::ColumnNumber column,
                   std::string_view message) final {
    status_.Update(absl::Status(
        absl::StatusCode::kFailedPrecondition,
        absl::StrCat("Failed to parse ", proto_.GetDescriptor()->name(),
                     " proto from text.  First failure is at line ", line,
                     " column ", column, " in file '", file_name_.string(),
                     "'.  Proto parser error:\n", message)));
  }

  absl::Status status() const { return status_; }

 private:
  absl::Status status_;
  const std::filesystem::path file_name_;
  const google::protobuf::Message& proto_;
};

enum class SetOrAppend { kSet, kAppend };

absl::Status SetFileContentsOrAppend(const std::filesystem::path& file_name,
                                     std::string_view content,
                                     SetOrAppend set_or_append) {
  // Use POSIX C APIs instead of C++ iostreams to avoid exceptions.
  int fd = open(file_name.c_str(),
                O_WRONLY | O_CREAT | O_CLOEXEC |
                    (set_or_append == SetOrAppend::kAppend ? O_APPEND : 0),
                0664);
  if (fd == -1) {
    return ErrNoToStatusWithFilename(errno, file_name);
  }

  // Clear existing contents if not appending.
  if (set_or_append == SetOrAppend::kSet) {
    if (ftruncate(fd, 0) == -1) {
      return ErrNoToStatusWithFilename(errno, file_name);
    }
  }

  ssize_t written = 0;
  while (written < content.size()) {
    ssize_t n = write(fd, content.data() + written, content.size() - written);
    if (n < 0) {
      if (errno == EAGAIN) {
        continue;
      }
      close(fd);
      return ErrNoToStatusWithFilename(errno, file_name);
    }
    written += n;
  }

  if (close(fd) != 0) {
    return ErrNoToStatusWithFilename(errno, file_name);
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::string> GetFileContents(
    const std::filesystem::path& file_name) {
  // Use POSIX C APIs instead of C++ iostreams to avoid exceptions.
  std::string result;

  int fd;
  if (file_name == "/dev/stdin" || file_name == "-") {
    fd = dup(STDIN_FILENO);  // dup standard input fd for portability:
                             // - /dev/stdin is not posix
                             // - avoid closing stdin's file descriptor
  } else {
    fd = open(file_name.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd == -1) {
      return ErrNoToStatusWithFilename(errno, file_name);
    }
  }

  char buf[4096];
  while (ssize_t n = read(fd, buf, sizeof(buf))) {
    if (n < 0) {
      if (errno == EAGAIN) {
        continue;
      }
      close(fd);
      return ErrNoToStatusWithFilename(errno, file_name);
    }
    result.append(buf, n);
  }

  if (close(fd) != 0) {
    return ErrNoToStatusWithFilename(errno, file_name);
  }
  return std::move(result);
}

absl::Status ParseTextProto(absl::string_view contents,
                            const std::filesystem::path& file_name,
                            google::protobuf::Message* proto) {
  if (proto == nullptr) {
    return absl::FailedPreconditionError("Invalid pointer value.");
  }
  ParseTextProtoFileErrorCollector collector(file_name, *proto);
  google::protobuf::TextFormat::Parser parser;
  parser.RecordErrorsTo(&collector);

  // Needed for this to compile in OSS, for some reason
  std::string contents_owned(contents.begin(), contents.end());
  const bool success = parser.ParseFromString(contents_owned, proto);
  DCHECK_EQ(success, collector.status().ok());
  return collector.status();
}

absl::Status ParseTextProtoFile(const std::filesystem::path& file_name,
                                google::protobuf::Message* proto) {
  MALDOCA_ASSIGN_OR_RETURN(std::string text_proto, GetFileContents(file_name));
  return ParseTextProto(text_proto, file_name, proto);
}

absl::Status SetFileContents(const std::filesystem::path& file_name,
                             std::string_view content) {
  return SetFileContentsOrAppend(file_name, content, SetOrAppend::kSet);
}

}  // namespace maldoca
