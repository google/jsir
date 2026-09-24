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
// JSIR-NEXT:     "jsir.variable_declaration"() <{kind = "const"}> ({
// JSIR-NEXT:       %0 = "jsir.identifier_ref"() <{name = "f"}> : () -> !jsir.any
// JSIR-NEXT:       %1 = "jsir.arrow_function_expression"() <{async = false, generator = false}> ({
// JSIR-NEXT:         %3 = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:         %4 = "jsir.identifier_ref"() <{name = "b"}> : () -> !jsir.any
// JSIR-NEXT:         %5 = "jsir.identifier"() <{name = "sideEffect"}> : () -> !jsir.any
// JSIR-NEXT:         %6 = "jsir.call_expression"(%5) : (!jsir.any) -> !jsir.any
// JSIR-NEXT:         %7 = "jsir.assignment_pattern_ref"(%4, %6) : (!jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:         %8 = "jsir.identifier_ref"() <{name = "rest"}> : () -> !jsir.any
// JSIR-NEXT:         %9 = "jsir.rest_element_ref"(%8) : (!jsir.any) -> !jsir.any
// JSIR-NEXT:         "jsir.exprs_region_end"(%3, %7, %9) : (!jsir.any, !jsir.any, !jsir.any) -> ()
// JSIR-NEXT:       }, {
// JSIR-NEXT:         %3 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:         %4 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSIR-NEXT:         %5 = "jsir.binary_expression"(%3, %4) <{operator_ = "+"}> : (!jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:         "jsir.expr_region_end"(%5) : (!jsir.any) -> ()
// JSIR-NEXT:       }) : () -> !jsir.any
// JSIR-NEXT:       %2 = "jsir.variable_declarator"(%0, %1) : (!jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:       "jsir.exprs_region_end"(%2) : (!jsir.any) -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:   }, {
// JSIR-NEXT:   ^bb0:
// JSIR-NEXT:   }) : () -> ()
// JSIR-NEXT: }) : () -> ()
