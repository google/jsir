// JSLIR:      "jsir.file"() <{comments = [#jsir<comment_line  <L 1 C 0>, <L 1 C 48>, 0, 48, " Must wrap with \22()\22, otherwise doesn't parse.">]}> ({
// JSLIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSLIR-NEXT:     %0 = "jsir.object_pattern_ref"() ({
// JSLIR-NEXT:       %4 = "jsir.identifier_ref"() <{name = "lvalue_shorthand"}> : () -> !jsir.any
// JSLIR-NEXT:       %5 = "jsir.object_property_ref"(%4) <{literal_key = #jsir<identifier   <L 4 C 4>, <L 4 C 20>, "lvalue_shorthand", 59, 75, 0, "lvalue_shorthand">, shorthand = true}> : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:       %6 = "jsir.identifier_ref"() <{name = "lvalue_1"}> : () -> !jsir.any
// JSLIR-NEXT:       %7 = "jsir.object_property_ref"(%6) <{literal_key = #jsir<identifier   <L 5 C 4>, <L 5 C 14>, "identifier", 81, 91, 0, "identifier">, shorthand = false}> : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:       %8 = "jsir.identifier_ref"() <{name = "lvalue_2"}> : () -> !jsir.any
// JSLIR-NEXT:       %9 = "jsir.object_property_ref"(%8) <{literal_key = #jsir<string_literal   <L 6 C 4>, <L 6 C 20>, 107, 123, 0, "string_literal",  "'string_literal'", "string_literal">, shorthand = false}> : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:       %10 = "jsir.identifier_ref"() <{name = "lvalue_3"}> : () -> !jsir.any
// JSLIR-NEXT:       %11 = "jsir.object_property_ref"(%10) <{literal_key = #jsir<numeric_literal   <L 7 C 4>, <L 7 C 7>, 139, 142, 0, 1.000000e+00 : f64,  "1.0", 1.000000e+00 : f64>, shorthand = false}> : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:       %12 = "jsir.string_literal"() <{extra = #jsir<string_literal_extra "\22computed\22", "computed">, value = "computed"}> : () -> !jsir.any
// JSLIR-NEXT:       %13 = "jsir.identifier_ref"() <{name = "lvalue_4"}> : () -> !jsir.any
// JSLIR-NEXT:       %14 = "jsir.object_property_ref"(%12, %13) <{shorthand = false}> : (!jsir.any, !jsir.any) -> !jsir.any
// JSLIR-NEXT:       %15 = "jsir.identifier_ref"() <{name = "lvalue_rest"}> : () -> !jsir.any
// JSLIR-NEXT:       %16 = "jsir.rest_element_ref"(%15) : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:       "jsir.exprs_region_end"(%5, %7, %9, %11, %14, %16) : (!jsir.any, !jsir.any, !jsir.any, !jsir.any, !jsir.any, !jsir.any) -> ()
// JSLIR-NEXT:     }) : () -> !jsir.any
// JSLIR-NEXT:     %1 = "jsir.identifier"() <{name = "obj"}> : () -> !jsir.any
// JSLIR-NEXT:     %2 = "jsir.assignment_expression"(%0, %1) <{operator_ = "="}> : (!jsir.any, !jsir.any) -> !jsir.any
// JSLIR-NEXT:     %3 = "jsir.parenthesized_expression"(%2) : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%3) : (!jsir.any) -> ()
// JSLIR-NEXT:   }, {
// JSLIR-NEXT:   ^bb0:
// JSLIR-NEXT:   }) : () -> ()
// JSLIR-NEXT: }) : () -> ()
