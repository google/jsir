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

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "maldoca/base/testing/protocol-buffer-matchers.h"
#include "maldoca/base/testing/status_matchers.h"
#include "maldoca/js/babel/babel.pb.h"
#include "google/protobuf/json/json.h"

namespace maldoca {
namespace {

using ::maldoca::testing::EqualsProto;

TEST(BabelJsonTest, BabelParseRequest) {
  BabelParseRequest request;
  request.set_error_recovery(true);
  request.set_source_type(BabelParseRequest::SOURCE_TYPE_SCRIPT);

  std::string json_string;
  MALDOCA_ASSERT_OK(
      google::protobuf::json::MessageToJsonString(request, &json_string, {}));

  EXPECT_EQ(json_string,
            R"({"errorRecovery":true,"sourceType":"SOURCE_TYPE_SCRIPT"})");

  BabelParseRequest parsed_request;
  MALDOCA_ASSERT_OK(
      google::protobuf::json::JsonStringToMessage(json_string, &parsed_request));

  EXPECT_THAT(parsed_request, EqualsProto(request));
}

TEST(BabelJsonTest, BabelScopes) {
  BabelScopes scopes;
  {
    BabelScope scope;
    scope.set_uid(0);
    {
      BabelBinding binding;
      binding.set_kind(BabelBinding::KIND_VAR);
      binding.set_name("global");
      binding.set_uid(1);

      scopes.mutable_bindings()->insert({binding.uid(), binding});
      scope.mutable_binding_uids()->insert({binding.name(), binding.uid()});
    }

    scopes.mutable_scopes()->insert({scope.uid(), scope});
  }

  google::protobuf::json::PrintOptions options = {
      .add_whitespace = true,
  };

  std::string json_string;
  MALDOCA_ASSERT_OK(
      google::protobuf::json::MessageToJsonString(scopes, &json_string, options));

  static const char kExpectedJsonString[] = R"json({
 "scopes": {
  "0": {
   "uid": 0,
   "bindingUids": {
    "global": 1
   }
  }
 },
 "bindings": {
  "1": {
   "kind": "KIND_VAR",
   "name": "global",
   "uid": 1
  }
 }
}
)json";

  EXPECT_EQ(json_string, kExpectedJsonString);

  BabelScopes parsed_scopes;
  MALDOCA_ASSERT_OK(
      google::protobuf::json::JsonStringToMessage(json_string, &parsed_scopes));

  EXPECT_THAT(parsed_scopes, EqualsProto(scopes));
}

}  // namespace
}  // namespace maldoca
