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

#ifndef MALDOCA_ASTGEN_TEST_ASSIGN_AIR_TYPES_TD_
#define MALDOCA_ASTGEN_TEST_ASSIGN_AIR_TYPES_TD_

include "maldoca/astgen/test/assign/air_dialect.td"

def AirAnyType : Air_Type<"AirAny"> {
  let summary = "A placeholder singleton type.";
  let mnemonic = "any";
  let assemblyFormat = "";
}

#endif  // MALDOCA_ASTGEN_TEST_ASSIGN_AIR_TYPES_TD_
