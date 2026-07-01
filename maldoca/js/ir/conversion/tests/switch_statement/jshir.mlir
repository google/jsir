// JSIR:      "jsir.file"() <{comments = []}> ({
// JSIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSIR-NEXT:     %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:     "jshir.switch_statement"(%0) ({
// JSIR-NEXT:       "jshir.switch_case"() ({
// JSIR-NEXT:         %4 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "0", 0.000000e+00 : f64>, value = 0.000000e+00 : f64}> : () -> !jsir.any
// JSIR-NEXT:         "jsir.expr_region_end"(%4) : (!jsir.any) -> ()
// JSIR-NEXT:       }, {
// JSIR-NEXT:         %4 = "jsir.identifier"() <{name = "body0"}> : () -> !jsir.any
// JSIR-NEXT:         "jsir.expression_statement"(%4) : (!jsir.any) -> ()
// JSIR-NEXT:         "jshir.break_statement"() : () -> ()
// JSIR-NEXT:       }) : () -> ()
// JSIR-NEXT:       "jshir.switch_case"() ({
// JSIR-NEXT:         %4 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "1", 1.000000e+00 : f64>, value = 1.000000e+00 : f64}> : () -> !jsir.any
// JSIR-NEXT:         "jsir.expr_region_end"(%4) : (!jsir.any) -> ()
// JSIR-NEXT:       }, {
// JSIR-NEXT:         %4 = "jsir.identifier"() <{name = "body1"}> : () -> !jsir.any
// JSIR-NEXT:         "jsir.expression_statement"(%4) : (!jsir.any) -> ()
// JSIR-NEXT:       }) : () -> ()
// JSIR-NEXT:       "jshir.switch_case"() ({
// JSIR-NEXT:       }, {
// JSIR-NEXT:         "jshir.break_statement"() : () -> ()
// JSIR-NEXT:       }) : () -> ()
// JSIR-NEXT:     }) : (!jsir.any) -> ()
// JSIR-NEXT:     %1 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:     "jshir.switch_statement"(%1) ({
// JSIR-NEXT:       "jshir.switch_case"() ({
// JSIR-NEXT:         %4 = "jsir.identifier"() <{name = "f"}> : () -> !jsir.any
// JSIR-NEXT:         %5 = "jsir.call_expression"(%4) : (!jsir.any) -> !jsir.any
// JSIR-NEXT:         "jsir.expr_region_end"(%5) : (!jsir.any) -> ()
// JSIR-NEXT:       }, {
// JSIR-NEXT:         %4 = "jsir.identifier"() <{name = "body0"}> : () -> !jsir.any
// JSIR-NEXT:         "jsir.expression_statement"(%4) : (!jsir.any) -> ()
// JSIR-NEXT:       }) : () -> ()
// JSIR-NEXT:       "jshir.switch_case"() ({
// JSIR-NEXT:       }, {
// JSIR-NEXT:         "jshir.break_statement"() : () -> ()
// JSIR-NEXT:       }) : () -> ()
// JSIR-NEXT:       "jshir.switch_case"() ({
// JSIR-NEXT:         %4 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "1", 1.000000e+00 : f64>, value = 1.000000e+00 : f64}> : () -> !jsir.any
// JSIR-NEXT:         %5 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "1", 1.000000e+00 : f64>, value = 1.000000e+00 : f64}> : () -> !jsir.any
// JSIR-NEXT:         %6 = "jsir.binary_expression"(%4, %5) <{operator_ = "+"}> : (!jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:         "jsir.expr_region_end"(%6) : (!jsir.any) -> ()
// JSIR-NEXT:       }, {
// JSIR-NEXT:         %4 = "jsir.identifier"() <{name = "body1"}> : () -> !jsir.any
// JSIR-NEXT:         "jsir.expression_statement"(%4) : (!jsir.any) -> ()
// JSIR-NEXT:         "jshir.break_statement"() : () -> ()
// JSIR-NEXT:       }) : () -> ()
// JSIR-NEXT:     }) : (!jsir.any) -> ()
// JSIR-NEXT:     %2 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:     "jshir.switch_statement"(%2) ({
// JSIR-NEXT:     ^bb0:
// JSIR-NEXT:     }) : (!jsir.any) -> ()
// JSIR-NEXT:     %3 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:     "jshir.switch_statement"(%3) ({
// JSIR-NEXT:       "jshir.switch_case"() ({
// JSIR-NEXT:         %4 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "0", 0.000000e+00 : f64>, value = 0.000000e+00 : f64}> : () -> !jsir.any
// JSIR-NEXT:         "jsir.expr_region_end"(%4) : (!jsir.any) -> ()
// JSIR-NEXT:       }, {
// JSIR-NEXT:       ^bb0:
// JSIR-NEXT:       }) : () -> ()
// JSIR-NEXT:     }) : (!jsir.any) -> ()
// JSIR-NEXT:   }, {
// JSIR-NEXT:   ^bb0:
// JSIR-NEXT:   }) : () -> ()
// JSIR-NEXT: }) : () -> ()
