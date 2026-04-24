// Copyright 2026 Google LLC
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

#ifndef MALDOCA_JS_IR_UTF16_H_
#define MALDOCA_JS_IR_UTF16_H_

#include <string>
#include <string_view>

#include "absl/strings/string_view.h"

namespace maldoca {

std::u16string Utf8ToUtf16(absl::string_view src);
std::string Utf16ToUtf8(std::u16string_view src);

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_UTF16_H_
