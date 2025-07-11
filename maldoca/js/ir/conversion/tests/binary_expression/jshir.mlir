// JSHIR:      "jsir.file"() <{comments = []}> ({
// JSHIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSHIR-NEXT:     %0 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "1", 1.000000e+00 : f64>, value = 1.000000e+00 : f64}> : () -> !jsir.any
// JSHIR-NEXT:     %1 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "2", 2.000000e+00 : f64>, value = 2.000000e+00 : f64}> : () -> !jsir.any
// JSHIR-NEXT:     %2 = "jsir.binary_expression"(%0, %1) <{operator_ = "+"}> : (!jsir.any, !jsir.any) -> !jsir.any
// JSHIR-NEXT:     %3 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "3", 3.000000e+00 : f64>, value = 3.000000e+00 : f64}> : () -> !jsir.any
// JSHIR-NEXT:     %4 = "jsir.binary_expression"(%2, %3) <{operator_ = "+"}> : (!jsir.any, !jsir.any) -> !jsir.any
// JSHIR-NEXT:     "jsir.expression_statement"(%4) : (!jsir.any) -> ()
// JSHIR-NEXT:   }, {
// JSHIR-NEXT:   ^bb0:
// JSHIR-NEXT:   }) : () -> ()
// JSHIR-NEXT: }) : () -> ()
