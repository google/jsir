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

#ifndef MALDOCA_BASE_INDENT_LOGGER_H_
#define MALDOCA_BASE_INDENT_LOGGER_H_

#include <iostream>
#include <string>

#include "absl/base/log_severity.h"
#include "absl/strings/string_view.h"
#include "absl/types/source_location.h"

namespace maldoca {

class IndentLogger {
 public:
  static IndentLogger& Get() {
    static IndentLogger* instance = new IndentLogger();
    return *instance;
  }

  virtual ~IndentLogger() = default;

  void Log(absl::string_view message,
           absl::LogSeverity severity = absl::LogSeverity::kInfo,
           absl::SourceLocation loc = absl::SourceLocation::current()) {
    return PrintIndent(message, loc, severity);
  }

  // RAII helper class to automatically manage indentation levels
  class IndentScope {
   public:
    explicit IndentScope(IndentLogger& logger) : logger_(logger) {
      logger_.IncrementIndent();
    }
    ~IndentScope() { logger_.DecrementIndent(); }

    // Non-copyable and non-movable
    IndentScope(const IndentScope&) = delete;
    IndentScope& operator=(const IndentScope&) = delete;

   private:
    IndentLogger& logger_;
  };

 private:
  friend class IndentScope;

  void IncrementIndent() { indent_ += 2; }
  void DecrementIndent() {
    if (indent_ >= 2) {
      indent_ -= 2;
    }
  }

 private:
  void PrintIndent(absl::string_view message, absl::SourceLocation loc,
                   absl::LogSeverity severity) {
    std::cout << "[" << absl::LogSeverityName(severity) << "] "
              << std::string(indent_, ' ') << message << " (" << loc.file_name()
              << ":" << loc.line() << ")" << std::endl;
    std::cout.flush();
  }

  int indent_ = 0;
};

#define MALDOCA_LOG_INDENT(message) ::maldoca::IndentLogger::Get().Log(message)

#define MALDOCA_LOG_INDENT_SEVERITY(message, severity) \
  ::maldoca::IndentLogger::Get().Log(message, severity)

#define MALDOCA_INDENT_SCOPE()                        \
  ::maldoca::IndentLogger::IndentScope _indent_scope( \
      ::maldoca::IndentLogger::Get())

}  // namespace maldoca
#endif  // MALDOCA_BASE_INDENT_LOGGER_H_
