// JSIR:      "jsir.file"() <{comments = []}> ({
// JSIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSIR-NEXT:     "jsir.function_declaration"() <{async = false, generator = false, id = #jsir<identifier <L 1 C 9>, <L 1 C 19>, "sideEffect", 9, 19, 1, "sideEffect">}> ({
// JSIR-NEXT:       "jsir.exprs_region_end"() : () -> ()
// JSIR-NEXT:     }, {
// JSIR-NEXT:       "jshir.block_statement"() ({
// JSIR-NEXT:         %0 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "1", 1.000000e+00 : f64>, value = 1.000000e+00 : f64}> : () -> !jsir.any
// JSIR-NEXT:         "jsir.return_statement"(%0) : (!jsir.any) -> ()
// JSIR-NEXT:       }, {
// JSIR-NEXT:       ^bb0:
// JSIR-NEXT:       }) : () -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:     "jsir.class_declaration"() <{id = #jsir<identifier <L 4 C 6>, <L 4 C 9>, "Foo", 44, 47, 2, "Foo">}> ({
// JSIR-NEXT:       "jsir.class_body"() ({
// JSIR-NEXT:         "jsir.class_method"() <{async = false, generator = false, kind = "method", literal_key = #jsir<identifier <L 5 C 2>, <L 5 C 8>, "method", 52, 58, 2, "method">, static_ = false}> ({
// JSIR-NEXT:           %2 = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:           %3 = "jsir.identifier_ref"() <{name = "b"}> : () -> !jsir.any
// JSIR-NEXT:           %4 = "jsir.identifier"() <{name = "sideEffect"}> : () -> !jsir.any
// JSIR-NEXT:           %5 = "jsir.call_expression"(%4) : (!jsir.any) -> !jsir.any
// JSIR-NEXT:           %6 = "jsir.assignment_pattern_ref"(%3, %5) : (!jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:           "jsir.exprs_region_end"(%2, %6) : (!jsir.any, !jsir.any) -> ()
// JSIR-NEXT:         }, {
// JSIR-NEXT:           "jshir.block_statement"() ({
// JSIR-NEXT:             %2 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:             %3 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSIR-NEXT:             %4 = "jsir.binary_expression"(%2, %3) <{operator_ = "+"}> : (!jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:             "jsir.return_statement"(%4) : (!jsir.any) -> ()
// JSIR-NEXT:           }, {
// JSIR-NEXT:           ^bb0:
// JSIR-NEXT:           }) : () -> ()
// JSIR-NEXT:         }) : () -> ()
// JSIR-NEXT:         "jsir.class_method"() <{async = false, generator = false, kind = "method", literal_key = #jsir<identifier <L 8 C 9>, <L 8 C 21>, "staticMethod", 113, 125, 2, "staticMethod">, static_ = true}> ({
// JSIR-NEXT:           %2 = "jsir.identifier_ref"() <{name = "c"}> : () -> !jsir.any
// JSIR-NEXT:           %3 = "jsir.identifier_ref"() <{name = "d"}> : () -> !jsir.any
// JSIR-NEXT:           %4 = "jsir.identifier"() <{name = "sideEffect"}> : () -> !jsir.any
// JSIR-NEXT:           %5 = "jsir.call_expression"(%4) : (!jsir.any) -> !jsir.any
// JSIR-NEXT:           %6 = "jsir.assignment_pattern_ref"(%3, %5) : (!jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:           "jsir.exprs_region_end"(%2, %6) : (!jsir.any, !jsir.any) -> ()
// JSIR-NEXT:         }, {
// JSIR-NEXT:           "jshir.block_statement"() ({
// JSIR-NEXT:             %2 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSIR-NEXT:             %3 = "jsir.identifier"() <{name = "d"}> : () -> !jsir.any
// JSIR-NEXT:             %4 = "jsir.binary_expression"(%2, %3) <{operator_ = "+"}> : (!jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:             "jsir.return_statement"(%4) : (!jsir.any) -> ()
// JSIR-NEXT:           }, {
// JSIR-NEXT:           ^bb0:
// JSIR-NEXT:           }) : () -> ()
// JSIR-NEXT:         }) : () -> ()
// JSIR-NEXT:         "jsir.class_method"() <{async = false, generator = false, kind = "set", literal_key = #jsir<identifier <L 11 C 6>, <L 11 C 11>, "value", 177, 182, 2, "value">, static_ = false}> ({
// JSIR-NEXT:           %2 = "jsir.identifier_ref"() <{name = "e"}> : () -> !jsir.any
// JSIR-NEXT:           %3 = "jsir.identifier"() <{name = "sideEffect"}> : () -> !jsir.any
// JSIR-NEXT:           %4 = "jsir.call_expression"(%3) : (!jsir.any) -> !jsir.any
// JSIR-NEXT:           %5 = "jsir.assignment_pattern_ref"(%2, %4) : (!jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:           "jsir.exprs_region_end"(%5) : (!jsir.any) -> ()
// JSIR-NEXT:         }, {
// JSIR-NEXT:           "jshir.block_statement"() ({
// JSIR-NEXT:             %2 = "jsir.this_expression"() : () -> !jsir.any
// JSIR-NEXT:             %3 = "jsir.member_expression_ref"(%2) <{literal_property = #jsir<identifier <L 12 C 9>, <L 12 C 15>, "_value", 212, 218, 7, "_value">}> : (!jsir.any) -> !jsir.any
// JSIR-NEXT:             %4 = "jsir.identifier"() <{name = "e"}> : () -> !jsir.any
// JSIR-NEXT:             %5 = "jsir.assignment_expression"(%3, %4) <{operator_ = "="}> : (!jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:             "jsir.expression_statement"(%5) : (!jsir.any) -> ()
// JSIR-NEXT:           }, {
// JSIR-NEXT:           ^bb0:
// JSIR-NEXT:           }) : () -> ()
// JSIR-NEXT:         }) : () -> ()
// JSIR-NEXT:         "jsir.class_private_method"() <{async = false, generator = false, key = #jsir<private_name <L 14 C 2>, <L 14 C 16>, 230, 244, 2, <L 14 C 3>, <L 14 C 16>, "privateMethod", 231, 244, 2, "privateMethod">, kind = "method", static_ = false}> ({
// JSIR-NEXT:           %2 = "jsir.identifier_ref"() <{name = "f"}> : () -> !jsir.any
// JSIR-NEXT:           %3 = "jsir.identifier_ref"() <{name = "g"}> : () -> !jsir.any
// JSIR-NEXT:           %4 = "jsir.identifier"() <{name = "sideEffect"}> : () -> !jsir.any
// JSIR-NEXT:           %5 = "jsir.call_expression"(%4) : (!jsir.any) -> !jsir.any
// JSIR-NEXT:           %6 = "jsir.assignment_pattern_ref"(%3, %5) : (!jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:           "jsir.exprs_region_end"(%2, %6) : (!jsir.any, !jsir.any) -> ()
// JSIR-NEXT:         }, {
// JSIR-NEXT:           "jshir.block_statement"() ({
// JSIR-NEXT:             %2 = "jsir.identifier"() <{name = "f"}> : () -> !jsir.any
// JSIR-NEXT:             %3 = "jsir.identifier"() <{name = "g"}> : () -> !jsir.any
// JSIR-NEXT:             %4 = "jsir.binary_expression"(%2, %3) <{operator_ = "+"}> : (!jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:             "jsir.return_statement"(%4) : (!jsir.any) -> ()
// JSIR-NEXT:           }, {
// JSIR-NEXT:           ^bb0:
// JSIR-NEXT:           }) : () -> ()
// JSIR-NEXT:         }) : () -> ()
// JSIR-NEXT:         %0 = "jsir.identifier"() <{name = "sideEffect"}> : () -> !jsir.any
// JSIR-NEXT:         %1 = "jsir.call_expression"(%0) : (!jsir.any) -> !jsir.any
// JSIR-NEXT:         "jsir.class_method"(%1) <{async = false, generator = false, kind = "method", static_ = false}> ({
// JSIR-NEXT:           %2 = "jsir.identifier_ref"() <{name = "h"}> : () -> !jsir.any
// JSIR-NEXT:           %3 = "jsir.identifier_ref"() <{name = "i"}> : () -> !jsir.any
// JSIR-NEXT:           %4 = "jsir.identifier"() <{name = "sideEffect"}> : () -> !jsir.any
// JSIR-NEXT:           %5 = "jsir.call_expression"(%4) : (!jsir.any) -> !jsir.any
// JSIR-NEXT:           %6 = "jsir.assignment_pattern_ref"(%3, %5) : (!jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:           "jsir.exprs_region_end"(%2, %6) : (!jsir.any, !jsir.any) -> ()
// JSIR-NEXT:         }, {
// JSIR-NEXT:           "jshir.block_statement"() ({
// JSIR-NEXT:             %2 = "jsir.identifier"() <{name = "h"}> : () -> !jsir.any
// JSIR-NEXT:             %3 = "jsir.identifier"() <{name = "i"}> : () -> !jsir.any
// JSIR-NEXT:             %4 = "jsir.binary_expression"(%2, %3) <{operator_ = "+"}> : (!jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:             "jsir.return_statement"(%4) : (!jsir.any) -> ()
// JSIR-NEXT:           }, {
// JSIR-NEXT:           ^bb0:
// JSIR-NEXT:           }) : () -> ()
// JSIR-NEXT:         }) : (!jsir.any) -> ()
// JSIR-NEXT:       }) : () -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:   }, {
// JSIR-NEXT:   ^bb0:
// JSIR-NEXT:   }) : () -> ()
// JSIR-NEXT: }) : () -> ()
