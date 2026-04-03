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

#ifndef MALDOCA_ASTGEN_PRINTER_BASE_H_
#define MALDOCA_ASTGEN_PRINTER_BASE_H_

#include <utility>

#include "google/protobuf/io/printer.h"
#include "google/protobuf/io/zero_copy_stream.h"

namespace maldoca {

class AstGenPrinterBase : public google::protobuf::io::Printer {
 public:
  explicit AstGenPrinterBase(google::protobuf::io::ZeroCopyOutputStream* os)
      : google::protobuf::io::Printer(os, /*variable_delimiter=*/'$') {}

  template <typename... Args>
  void Println(Args&&... args) {
    Print(std::forward<Args>(args)...);
    Print("\n");
  }

  void Println() { Println(""); }
};

}  // namespace maldoca

#endif  // MALDOCA_ASTGEN_PRINTER_BASE_H_
