// JSLIR:      "jsir.file"() <{comments = []}> ({
// JSLIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSLIR-NEXT:     %0 = "jsir.identifier"() <{name = "foo"}> : () -> !jsir.any
// JSLIR-NEXT:     %1 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "1", 1.000000e+00 : f64>, value = 1.000000e+00 : f64}> : () -> !jsir.any
// JSLIR-NEXT:     %2 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "2", 2.000000e+00 : f64>, value = 2.000000e+00 : f64}> : () -> !jsir.any
// JSLIR-NEXT:     %3 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     %4 = "jsir.spread_element"(%3) : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:     %5 = "jsir.new_expression"(%0, %1, %2, %4) : (!jsir.any, !jsir.any, !jsir.any, !jsir.any) -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%5) : (!jsir.any) -> ()
// JSLIR-NEXT:   }, {
// JSLIR-NEXT:   ^bb0:
// JSLIR-NEXT:   }) : () -> ()
// JSLIR-NEXT: }) : () -> ()
