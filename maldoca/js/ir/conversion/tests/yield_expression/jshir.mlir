// JSHIR:      "jsir.file"() <{comments = []}> ({
// JSHIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSHIR-NEXT:     "jsir.function_declaration"() <{async = false, generator = true, id = #jsir<identifier <L 1 C 10>, <L 1 C 13>, "gen", 10, 13, 1, "gen">}> ({
// JSHIR-NEXT:       "jsir.exprs_region_end"() : () -> ()
// JSHIR-NEXT:     }, {
// JSHIR-NEXT:       "jshir.block_statement"() ({
// JSHIR-NEXT:         %0 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "1", 1.000000e+00 : f64>, value = 1.000000e+00 : f64}> : () -> !jsir.any
// JSHIR-NEXT:         %1 = "jsir.yield_expression"(%0) <{delegate = false}> : (!jsir.any) -> !jsir.any
// JSHIR-NEXT:         "jsir.expression_statement"(%1) : (!jsir.any) -> ()
// JSHIR-NEXT:         %2 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "2", 2.000000e+00 : f64>, value = 2.000000e+00 : f64}> : () -> !jsir.any
// JSHIR-NEXT:         %3 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "3", 3.000000e+00 : f64>, value = 3.000000e+00 : f64}> : () -> !jsir.any
// JSHIR-NEXT:         %4 = "jsir.array_expression"(%2, %3) : (!jsir.any, !jsir.any) -> !jsir.any
// JSHIR-NEXT:         %5 = "jsir.yield_expression"(%4) <{delegate = true}> : (!jsir.any) -> !jsir.any
// JSHIR-NEXT:         "jsir.expression_statement"(%5) : (!jsir.any) -> ()
// JSHIR-NEXT:       }, {
// JSHIR-NEXT:       ^bb0:
// JSHIR-NEXT:       }) : () -> ()
// JSHIR-NEXT:     }) : () -> ()
// JSHIR-NEXT:   }, {
// JSHIR-NEXT:   ^bb0:
// JSHIR-NEXT:   }) : () -> ()
// JSHIR-NEXT: }) : () -> ()
