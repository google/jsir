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

#include "maldoca/base/indent_logger.h"

#include <iostream>
#include <sstream>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/log_severity.h"

namespace maldoca {
namespace {

class IndentLoggerTest : public ::testing::Test {
 protected:
  void SetUp() override { old_cout_buf_ = std::cout.rdbuf(ss_.rdbuf()); }

  void TearDown() override { std::cout.rdbuf(old_cout_buf_); }

  std::stringstream ss_;
  std::streambuf* old_cout_buf_;
};

TEST_F(IndentLoggerTest, SimpleLog) {
  MALDOCA_LOG_INDENT("Test message");
  EXPECT_THAT(ss_.str(), testing::HasSubstr("[INFO]"));
  EXPECT_THAT(ss_.str(), testing::HasSubstr("Test message"));
}

TEST_F(IndentLoggerTest, ScopedIndent) {
  MALDOCA_LOG_INDENT("Level 0");
  {
    MALDOCA_INDENT_SCOPE();
    MALDOCA_LOG_INDENT("Level 1");
  }
  MALDOCA_LOG_INDENT("Level 0 again");

  // Verify that Level 1 is indented by exactly two spaces,
  // and that Level 0 has no indentation spaces after the severity prefix.
  EXPECT_THAT(ss_.str(), testing::HasSubstr("[INFO] Level 0"));
  EXPECT_THAT(ss_.str(), testing::HasSubstr("[INFO]   Level 1"));
  EXPECT_THAT(ss_.str(), testing::HasSubstr("[INFO] Level 0 again"));
}

TEST_F(IndentLoggerTest, SeverityLogs) {
  MALDOCA_LOG_INDENT_SEVERITY("A warning!", absl::LogSeverity::kWarning);
  MALDOCA_LOG_INDENT_SEVERITY("An error!", absl::LogSeverity::kError);

  EXPECT_THAT(ss_.str(), testing::HasSubstr("[WARNING] A warning!"));
  EXPECT_THAT(ss_.str(), testing::HasSubstr("[ERROR] An error!"));
}

}  // namespace
}  // namespace maldoca
