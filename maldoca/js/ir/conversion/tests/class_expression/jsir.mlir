// JSIR:      "jsir.file"() <{comments = []}> ({
// JSIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSIR-NEXT:     %0 = "jsir.class_expression"() ({
// JSIR-NEXT:       "jsir.class_body"() ({
// JSIR-NEXT:         "jsir.class_property"() <{literal_key = #jsir<identifier <L 2 C 2>, <L 2 C 21>, "property_identifier", 11, 30, 1, "property_identifier">, static_ = false}> ({
// JSIR-NEXT:           %4 = "jsir.numeric_literal"() <{value = 1.000000e+00 : f64}> : () -> !jsir.any
// JSIR-NEXT:           "jsir.expr_region_end"(%4) : (!jsir.any) -> ()
// JSIR-NEXT:         }) : () -> ()
// JSIR-NEXT:         "jsir.class_private_property"() <{key = #jsir<private_name <L 3 C 2>, <L 3 C 24>, 38, 60, 1, <L 3 C 3>, <L 3 C 24>, "property_private_name", 39, 60, 1, "property_private_name">, static_ = false}> ({
// JSIR-NEXT:           %4 = "jsir.numeric_literal"() <{value = 2.000000e+00 : f64}> : () -> !jsir.any
// JSIR-NEXT:           "jsir.expr_region_end"(%4) : (!jsir.any) -> ()
// JSIR-NEXT:         }) : () -> ()
// JSIR-NEXT:         "jsir.class_property"() <{literal_key = #jsir<string_literal <L 4 C 2>, <L 4 C 27>, 68, 93, 1, {raw = "\22property_literal_string\22", rawValue = "property_literal_string"}, "property_literal_string">, static_ = false}> ({
// JSIR-NEXT:           %4 = "jsir.numeric_literal"() <{value = 3.000000e+00 : f64}> : () -> !jsir.any
// JSIR-NEXT:           "jsir.expr_region_end"(%4) : (!jsir.any) -> ()
// JSIR-NEXT:         }) : () -> ()
// JSIR-NEXT:         "jsir.class_property"() <{literal_key = #jsir<numeric_literal <L 5 C 2>, <L 5 C 5>, 101, 104, 1, {raw = "1.0", rawValue = 1 : i64}, 1.000000e+00 : f64>, static_ = false}> ({
// JSIR-NEXT:           %4 = "jsir.numeric_literal"() <{value = 4.000000e+00 : f64}> : () -> !jsir.any
// JSIR-NEXT:           "jsir.expr_region_end"(%4) : (!jsir.any) -> ()
// JSIR-NEXT:         }) : () -> ()
// JSIR-NEXT:         %2 = "jsir.string_literal"() <{value = "property_computed"}> : () -> !jsir.any
// JSIR-NEXT:         "jsir.class_property"(%2) <{static_ = false}> ({
// JSIR-NEXT:           %4 = "jsir.numeric_literal"() <{value = 5.000000e+00 : f64}> : () -> !jsir.any
// JSIR-NEXT:           "jsir.expr_region_end"(%4) : (!jsir.any) -> ()
// JSIR-NEXT:         }) : (!jsir.any) -> ()
// JSIR-NEXT:         "jsir.class_method"() <{async = false, generator = false, kind = "method", literal_key = #jsir<identifier <L 7 C 2>, <L 7 C 19>, "method_identifier", 141, 158, 1, "method_identifier">, static_ = false}> ({
// JSIR-NEXT:           "jsir.exprs_region_end"() : () -> ()
// JSIR-NEXT:         }, {
// JSIR-NEXT:           "jshir.block_statement"() ({
// JSIR-NEXT:           ^bb0:
// JSIR-NEXT:           }, {
// JSIR-NEXT:           ^bb0:
// JSIR-NEXT:           }) : () -> ()
// JSIR-NEXT:         }) : () -> ()
// JSIR-NEXT:         "jsir.class_private_method"() <{async = false, generator = false, key = #jsir<private_name <L 8 C 2>, <L 8 C 22>, 166, 186, 1, <L 8 C 3>, <L 8 C 22>, "method_private_name", 167, 186, 1, "method_private_name">, kind = "method", static_ = false}> ({
// JSIR-NEXT:           "jsir.exprs_region_end"() : () -> ()
// JSIR-NEXT:         }, {
// JSIR-NEXT:           "jshir.block_statement"() ({
// JSIR-NEXT:           ^bb0:
// JSIR-NEXT:           }, {
// JSIR-NEXT:           ^bb0:
// JSIR-NEXT:           }) : () -> ()
// JSIR-NEXT:         }) : () -> ()
// JSIR-NEXT:         "jsir.class_method"() <{async = false, generator = false, kind = "method", literal_key = #jsir<string_literal <L 9 C 2>, <L 9 C 25>, 194, 217, 1, {raw = "\22method_literal_string\22", rawValue = "method_literal_string"}, "method_literal_string">, static_ = false}> ({
// JSIR-NEXT:           "jsir.exprs_region_end"() : () -> ()
// JSIR-NEXT:         }, {
// JSIR-NEXT:           "jshir.block_statement"() ({
// JSIR-NEXT:           ^bb0:
// JSIR-NEXT:           }, {
// JSIR-NEXT:           ^bb0:
// JSIR-NEXT:           }) : () -> ()
// JSIR-NEXT:         }) : () -> ()
// JSIR-NEXT:         "jsir.class_method"() <{async = false, generator = false, kind = "method", literal_key = #jsir<numeric_literal <L 10 C 2>, <L 10 C 5>, 225, 228, 1, {raw = "1.0", rawValue = 1 : i64}, 1.000000e+00 : f64>, static_ = false}> ({
// JSIR-NEXT:           "jsir.exprs_region_end"() : () -> ()
// JSIR-NEXT:         }, {
// JSIR-NEXT:           "jshir.block_statement"() ({
// JSIR-NEXT:           ^bb0:
// JSIR-NEXT:           }, {
// JSIR-NEXT:           ^bb0:
// JSIR-NEXT:           }) : () -> ()
// JSIR-NEXT:         }) : () -> ()
// JSIR-NEXT:         %3 = "jsir.string_literal"() <{value = "method_computed"}> : () -> !jsir.any
// JSIR-NEXT:         "jsir.class_method"(%3) <{async = false, generator = false, kind = "method", static_ = false}> ({
// JSIR-NEXT:           "jsir.exprs_region_end"() : () -> ()
// JSIR-NEXT:         }, {
// JSIR-NEXT:           "jshir.block_statement"() ({
// JSIR-NEXT:           ^bb0:
// JSIR-NEXT:           }, {
// JSIR-NEXT:           ^bb0:
// JSIR-NEXT:           }) : () -> ()
// JSIR-NEXT:         }) : (!jsir.any) -> ()
// JSIR-NEXT:       }) : () -> ()
// JSIR-NEXT:     }) : () -> !jsir.any
// JSIR-NEXT:     %1 = "jsir.parenthesized_expression"(%0) : (!jsir.any) -> !jsir.any
// JSIR-NEXT:     "jsir.expression_statement"(%1) : (!jsir.any) -> ()
// JSIR-NEXT:   }, {
// JSIR-NEXT:   ^bb0:
// JSIR-NEXT:   }) : () -> ()
// JSIR-NEXT: }) : () -> ()
