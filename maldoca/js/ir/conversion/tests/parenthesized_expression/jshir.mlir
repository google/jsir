// JSIR:      "jsir.file"() <{comments = []}> ({
// JSIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSIR-NEXT:     %0 = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:     %1 = "jsir.parenthesized_expression_ref"(%0) : (!jsir.any) -> !jsir.any
// JSIR-NEXT:     %2 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "0", 0.000000e+00 : f64>, value = 0.000000e+00 : f64}> : () -> !jsir.any
// JSIR-NEXT:     %3 = "jsir.assignment_expression"(%1, %2) <{operator_ = "="}> : (!jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:     "jsir.expression_statement"(%3) : (!jsir.any) -> ()
// JSIR-NEXT:     %4 = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:     %5 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "0", 0.000000e+00 : f64>, value = 0.000000e+00 : f64}> : () -> !jsir.any
// JSIR-NEXT:     %6 = "jsir.parenthesized_expression"(%5) : (!jsir.any) -> !jsir.any
// JSIR-NEXT:     %7 = "jsir.assignment_expression"(%4, %6) <{operator_ = "="}> : (!jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:     "jsir.expression_statement"(%7) : (!jsir.any) -> ()
// JSIR-NEXT:   }, {
// JSIR-NEXT:   ^bb0:
// JSIR-NEXT:   }) : () -> ()
// JSIR-NEXT: }) : () -> ()
