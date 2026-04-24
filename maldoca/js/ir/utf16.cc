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

#include "maldoca/js/ir/utf16.h"

#include <string>
#include <string_view>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ConvertUTF.h"
#include "absl/strings/string_view.h"

namespace maldoca {

std::u16string Utf8ToUtf16(absl::string_view src) {
  llvm::StringRef src_ref(src.data(), src.size());
  llvm::SmallVector<llvm::UTF16, 1024> dst;
  if (!llvm::convertUTF8ToUTF16String(src_ref, dst)) {
    return u"";
  }
  return std::u16string(reinterpret_cast<const char16_t*>(dst.data()),
                        dst.size());
}

std::string Utf16ToUtf8(std::u16string_view src) {
  llvm::ArrayRef<llvm::UTF16> src_ref(
      reinterpret_cast<const llvm::UTF16*>(src.data()), src.size());
  std::string dst;
  if (!llvm::convertUTF16ToUTF8String(src_ref, dst)) {
    return "";
  }
  return dst;
}

}  // namespace maldoca
