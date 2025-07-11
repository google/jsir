// JSLIR:      "jsir.file"() <{comments = []}> ({
// JSLIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSLIR-NEXT:     %0 = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     %1 = "jsir.parenthesized_expression_ref"(%0) : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:     %2 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "0", 0.000000e+00 : f64>, value = 0.000000e+00 : f64}> : () -> !jsir.any
// JSLIR-NEXT:     %3 = "jsir.assignment_expression"(%1, %2) <{operator_ = "="}> : (!jsir.any, !jsir.any) -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%3) : (!jsir.any) -> ()
// JSLIR-NEXT:     %4 = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     %5 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "0", 0.000000e+00 : f64>, value = 0.000000e+00 : f64}> : () -> !jsir.any
// JSLIR-NEXT:     %6 = "jsir.parenthesized_expression"(%5) : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:     %7 = "jsir.assignment_expression"(%4, %6) <{operator_ = "="}> : (!jsir.any, !jsir.any) -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%7) : (!jsir.any) -> ()
// JSLIR-NEXT:   }, {
// JSLIR-NEXT:   ^bb0:
// JSLIR-NEXT:   }) : () -> ()
// JSLIR-NEXT: }) : () -> ()
