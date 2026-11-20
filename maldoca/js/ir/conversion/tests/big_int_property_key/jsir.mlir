// JSIR:      "jsir.file"() <{comments = []}> ({
// JSIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSIR-NEXT:     %0 = "jsir.object_expression"() ({
// JSIR-NEXT:       %2 = "jsir.numeric_literal"() <{value = 0.000000e+00 : f64}> : () -> !jsir.any
// JSIR-NEXT:       %3 = "jsir.object_property"(%2) <{literal_key = #jsir<big_int_literal <L 2 C 2>, <L 2 C 4>, 5, 7, 0, {raw = "0n", rawValue = "0"}, "0">, shorthand = false}> : (!jsir.any) -> !jsir.any
// JSIR-NEXT:       %4 = "jsir.object_method"() <{async = false, generator = false, kind = "method", literal_key = #jsir<big_int_literal <L 3 C 2>, <L 3 C 4>, 14, 16, 0, {raw = "1n", rawValue = "1"}, "1">}> ({
// JSIR-NEXT:         "jsir.exprs_region_end"() : () -> ()
// JSIR-NEXT:       }, {
// JSIR-NEXT:         "jshir.block_statement"() ({
// JSIR-NEXT:         ^bb0:
// JSIR-NEXT:         }, {
// JSIR-NEXT:         ^bb0:
// JSIR-NEXT:         }) : () -> ()
// JSIR-NEXT:       }) : () -> !jsir.any
// JSIR-NEXT:       %5 = "jsir.object_method"() <{async = false, generator = false, kind = "get", literal_key = #jsir<big_int_literal <L 4 C 6>, <L 4 C 8>, 29, 31, 0, {raw = "2n", rawValue = "2"}, "2">}> ({
// JSIR-NEXT:         "jsir.exprs_region_end"() : () -> ()
// JSIR-NEXT:       }, {
// JSIR-NEXT:         "jshir.block_statement"() ({
// JSIR-NEXT:         ^bb0:
// JSIR-NEXT:         }, {
// JSIR-NEXT:         ^bb0:
// JSIR-NEXT:         }) : () -> ()
// JSIR-NEXT:       }) : () -> !jsir.any
// JSIR-NEXT:       %6 = "jsir.numeric_literal"() <{value = 1.000000e+01 : f64}> : () -> !jsir.any
// JSIR-NEXT:       %7 = "jsir.object_property"(%6) <{literal_key = #jsir<big_int_literal <L 5 C 2>, <L 5 C 6>, 40, 44, 0, {raw = "0xan", rawValue = "0xa"}, "0xa">, shorthand = false}> : (!jsir.any) -> !jsir.any
// JSIR-NEXT:       "jsir.exprs_region_end"(%3, %4, %5, %7) : (!jsir.any, !jsir.any, !jsir.any, !jsir.any) -> ()
// JSIR-NEXT:     }) : () -> !jsir.any
// JSIR-NEXT:     %1 = "jsir.parenthesized_expression"(%0) : (!jsir.any) -> !jsir.any
// JSIR-NEXT:     "jsir.expression_statement"(%1) : (!jsir.any) -> ()
// JSIR-NEXT:   }, {
// JSIR-NEXT:   ^bb0:
// JSIR-NEXT:   }) : () -> ()
// JSIR-NEXT: }) : () -> ()
