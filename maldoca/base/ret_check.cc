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

#include "maldoca/base/ret_check.h"

#include <cstddef>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>

#include "absl/base/log_severity.h"
#include "absl/status/status.h"
#include "maldoca/base/source_location.h"
#include "maldoca/base/status_builder.h"

namespace maldoca {
namespace internal_status_macros_ret_check {

StatusBuilder RetCheckFailSlowPath(SourceLocation location) {
  return InternalErrorBuilder(location)
             .Log(absl::LogSeverity::kError)
             .EmitStackTrace()
         << "MALDOCA_RET_CHECK failure (" << location.file_name() << ":"
         << location.line() << ") ";
}

StatusBuilder RetCheckFailSlowPath(SourceLocation location,
                                   std::string* condition) {
  std::unique_ptr<std::string> cleanup(condition);
  return RetCheckFailSlowPath(location) << *condition << " ";
}

StatusBuilder RetCheckFailSlowPath(SourceLocation location,
                                   const char* condition) {
  return RetCheckFailSlowPath(location) << condition << " ";
}

StatusBuilder RetCheckFailSlowPath(SourceLocation location,
                                   const char* condition,
                                   const absl::Status& status) {
  return RetCheckFailSlowPath(location)
         << condition << " returned " << status.ToString() << " ";
}

CheckOpMessageBuilder::CheckOpMessageBuilder(const char* exprtext)
    : stream_(new std::ostringstream) {
  *stream_ << exprtext << " (";
}

CheckOpMessageBuilder::~CheckOpMessageBuilder() { delete stream_; }

std::ostream* CheckOpMessageBuilder::ForVar2() {
  *stream_ << " vs. ";
  return stream_;
}

std::string* CheckOpMessageBuilder::NewString() {
  *stream_ << ")";
  return new std::string(stream_->str());
}

void MakeCheckOpValueString(std::ostream* os, char v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "char value " << int{v};
  }
}

void MakeCheckOpValueString(std::ostream* os, signed char v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "signed char value " << int{v};
  }
}

void MakeCheckOpValueString(std::ostream* os, unsigned char v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "unsigned char value " << int{v};
  }
}

void MakeCheckOpValueString(std::ostream* os, std::nullptr_t) {
  (*os) << "nullptr";
}

void MakeCheckOpValueString(std::ostream* os, const char* v) {
  if (v == nullptr) {
    (*os) << "nullptr";
  } else {
    (*os) << v;
  }
}

void MakeCheckOpValueString(std::ostream* os, const signed char* v) {
  if (v == nullptr) {
    (*os) << "nullptr";
  } else {
    (*os) << v;
  }
}

void MakeCheckOpValueString(std::ostream* os, const unsigned char* v) {
  if (v == nullptr) {
    (*os) << "nullptr";
  } else {
    (*os) << v;
  }
}

void MakeCheckOpValueString(std::ostream* os, char* v) {
  if (v == nullptr) {
    (*os) << "nullptr";
  } else {
    (*os) << v;
  }
}

void MakeCheckOpValueString(std::ostream* os, signed char* v) {
  if (v == nullptr) {
    (*os) << "nullptr";
  } else {
    (*os) << v;
  }
}

void MakeCheckOpValueString(std::ostream* os, unsigned char* v) {
  if (v == nullptr) {
    (*os) << "nullptr";
  } else {
    (*os) << v;
  }
}

}  // namespace internal_status_macros_ret_check
}  // namespace maldoca
