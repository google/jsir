// JSIR:      "jsir.file"() <{comments = []}> ({
// JSIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSIR-NEXT:     %0 = "jsir.object_expression"() ({
// JSIR-NEXT:       %2 = "jsir.identifier"() <{name = "short_hand"}> : () -> !jsir.any
// JSIR-NEXT:       %3 = "jsir.object_property"(%2) <{literal_key = #jsir<identifier <L 2 C 2>, <L 2 C 12>, "short_hand", 5, 15, 0, "short_hand">, shorthand = true}> : (!jsir.any) -> !jsir.any
// JSIR-NEXT:       %4 = "jsir.numeric_literal"() <{value = 1.000000e+00 : f64}> : () -> !jsir.any
// JSIR-NEXT:       %5 = "jsir.object_property"(%4) <{literal_key = #jsir<identifier <L 3 C 2>, <L 3 C 21>, "property_identifier", 19, 38, 0, "property_identifier">, shorthand = false}> : (!jsir.any) -> !jsir.any
// JSIR-NEXT:       %6 = "jsir.numeric_literal"() <{value = 2.000000e+00 : f64}> : () -> !jsir.any
// JSIR-NEXT:       %7 = "jsir.object_property"(%6) <{literal_key = #jsir<string_literal <L 4 C 2>, <L 4 C 27>, 45, 70, 0, {raw = "\22property_string_literal\22", rawValue = "property_string_literal"}, "property_string_literal">, shorthand = false}> : (!jsir.any) -> !jsir.any
// JSIR-NEXT:       %8 = "jsir.numeric_literal"() <{value = 3.000000e+00 : f64}> : () -> !jsir.any
// JSIR-NEXT:       %9 = "jsir.object_property"(%8) <{literal_key = #jsir<numeric_literal <L 5 C 2>, <L 5 C 5>, 77, 80, 0, {raw = "1.0", rawValue = 1 : i64}, 1.000000e+00 : f64>, shorthand = false}> : (!jsir.any) -> !jsir.any
// JSIR-NEXT:       %10 = "jsir.string_literal"() <{value = "property_computed"}> : () -> !jsir.any
// JSIR-NEXT:       %11 = "jsir.numeric_literal"() <{value = 4.000000e+00 : f64}> : () -> !jsir.any
// JSIR-NEXT:       %12 = "jsir.object_property"(%10, %11) <{shorthand = false}> : (!jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:       %13 = "jsir.object_method"() <{async = false, generator = false, kind = "method", literal_key = #jsir<identifier <L 7 C 2>, <L 7 C 19>, "method_identifier", 115, 132, 0, "method_identifier">}> ({
// JSIR-NEXT:         "jsir.exprs_region_end"() : () -> ()
// JSIR-NEXT:       }, {
// JSIR-NEXT:         "jshir.block_statement"() ({
// JSIR-NEXT:         ^bb0:
// JSIR-NEXT:         }, {
// JSIR-NEXT:         ^bb0:
// JSIR-NEXT:         }) : () -> ()
// JSIR-NEXT:       }) : () -> !jsir.any
// JSIR-NEXT:       %14 = "jsir.object_method"() <{async = false, generator = false, kind = "method", literal_key = #jsir<string_literal <L 8 C 2>, <L 8 C 27>, 141, 166, 0, {raw = "\22property_string_literal\22", rawValue = "property_string_literal"}, "property_string_literal">}> ({
// JSIR-NEXT:         "jsir.exprs_region_end"() : () -> ()
// JSIR-NEXT:       }, {
// JSIR-NEXT:         "jshir.block_statement"() ({
// JSIR-NEXT:         ^bb0:
// JSIR-NEXT:         }, {
// JSIR-NEXT:         ^bb0:
// JSIR-NEXT:         }) : () -> ()
// JSIR-NEXT:       }) : () -> !jsir.any
// JSIR-NEXT:       %15 = "jsir.object_method"() <{async = false, generator = false, kind = "method", literal_key = #jsir<numeric_literal <L 9 C 2>, <L 9 C 5>, 175, 178, 0, {raw = "1.0", rawValue = 1 : i64}, 1.000000e+00 : f64>}> ({
// JSIR-NEXT:         "jsir.exprs_region_end"() : () -> ()
// JSIR-NEXT:       }, {
// JSIR-NEXT:         "jshir.block_statement"() ({
// JSIR-NEXT:         ^bb0:
// JSIR-NEXT:         }, {
// JSIR-NEXT:         ^bb0:
// JSIR-NEXT:         }) : () -> ()
// JSIR-NEXT:       }) : () -> !jsir.any
// JSIR-NEXT:       %16 = "jsir.string_literal"() <{value = "property_computed"}> : () -> !jsir.any
// JSIR-NEXT:       %17 = "jsir.object_method"(%16) <{async = false, generator = false, kind = "method"}> ({
// JSIR-NEXT:         "jsir.exprs_region_end"() : () -> ()
// JSIR-NEXT:       }, {
// JSIR-NEXT:         "jshir.block_statement"() ({
// JSIR-NEXT:         ^bb0:
// JSIR-NEXT:         }, {
// JSIR-NEXT:         ^bb0:
// JSIR-NEXT:         }) : () -> ()
// JSIR-NEXT:       }) : (!jsir.any) -> !jsir.any
// JSIR-NEXT:       %18 = "jsir.identifier"() <{name = "spread_element"}> : () -> !jsir.any
// JSIR-NEXT:       %19 = "jsir.spread_element"(%18) : (!jsir.any) -> !jsir.any
// JSIR-NEXT:       "jsir.exprs_region_end"(%3, %5, %7, %9, %12, %13, %14, %15, %17, %19) : (!jsir.any, !jsir.any, !jsir.any, !jsir.any, !jsir.any, !jsir.any, !jsir.any, !jsir.any, !jsir.any, !jsir.any) -> ()
// JSIR-NEXT:     }) : () -> !jsir.any
// JSIR-NEXT:     %1 = "jsir.parenthesized_expression"(%0) : (!jsir.any) -> !jsir.any
// JSIR-NEXT:     "jsir.expression_statement"(%1) : (!jsir.any) -> ()
// JSIR-NEXT:   }, {
// JSIR-NEXT:   ^bb0:
// JSIR-NEXT:   }) : () -> ()
// JSIR-NEXT: }) : () -> ()
