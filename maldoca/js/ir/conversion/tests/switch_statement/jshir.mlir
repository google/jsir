// JSHIR:      "jsir.file"() <{comments = []}> ({
// JSHIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSHIR-NEXT:     %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSHIR-NEXT:     "jshir.switch_statement"(%0) ({
// JSHIR-NEXT:       "jshir.switch_case"() ({
// JSHIR-NEXT:         %4 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "0", 0.000000e+00 : f64>, value = 0.000000e+00 : f64}> : () -> !jsir.any
// JSHIR-NEXT:         "jsir.expr_region_end"(%4) : (!jsir.any) -> ()
// JSHIR-NEXT:       }, {
// JSHIR-NEXT:         %4 = "jsir.identifier"() <{name = "body0"}> : () -> !jsir.any
// JSHIR-NEXT:         "jsir.expression_statement"(%4) : (!jsir.any) -> ()
// JSHIR-NEXT:         "jshir.break_statement"() : () -> ()
// JSHIR-NEXT:       }) : () -> ()
// JSHIR-NEXT:       "jshir.switch_case"() ({
// JSHIR-NEXT:         %4 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "1", 1.000000e+00 : f64>, value = 1.000000e+00 : f64}> : () -> !jsir.any
// JSHIR-NEXT:         "jsir.expr_region_end"(%4) : (!jsir.any) -> ()
// JSHIR-NEXT:       }, {
// JSHIR-NEXT:         %4 = "jsir.identifier"() <{name = "body1"}> : () -> !jsir.any
// JSHIR-NEXT:         "jsir.expression_statement"(%4) : (!jsir.any) -> ()
// JSHIR-NEXT:       }) : () -> ()
// JSHIR-NEXT:       "jshir.switch_case"() ({
// JSHIR-NEXT:       }, {
// JSHIR-NEXT:         "jshir.break_statement"() : () -> ()
// JSHIR-NEXT:       }) : () -> ()
// JSHIR-NEXT:     }) : (!jsir.any) -> ()
// JSHIR-NEXT:     %1 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSHIR-NEXT:     "jshir.switch_statement"(%1) ({
// JSHIR-NEXT:       "jshir.switch_case"() ({
// JSHIR-NEXT:         %4 = "jsir.identifier"() <{name = "f"}> : () -> !jsir.any
// JSHIR-NEXT:         %5 = "jsir.call_expression"(%4) : (!jsir.any) -> !jsir.any
// JSHIR-NEXT:         "jsir.expr_region_end"(%5) : (!jsir.any) -> ()
// JSHIR-NEXT:       }, {
// JSHIR-NEXT:         %4 = "jsir.identifier"() <{name = "body0"}> : () -> !jsir.any
// JSHIR-NEXT:         "jsir.expression_statement"(%4) : (!jsir.any) -> ()
// JSHIR-NEXT:       }) : () -> ()
// JSHIR-NEXT:       "jshir.switch_case"() ({
// JSHIR-NEXT:       }, {
// JSHIR-NEXT:         "jshir.break_statement"() : () -> ()
// JSHIR-NEXT:       }) : () -> ()
// JSHIR-NEXT:       "jshir.switch_case"() ({
// JSHIR-NEXT:         %4 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "1", 1.000000e+00 : f64>, value = 1.000000e+00 : f64}> : () -> !jsir.any
// JSHIR-NEXT:         %5 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "1", 1.000000e+00 : f64>, value = 1.000000e+00 : f64}> : () -> !jsir.any
// JSHIR-NEXT:         %6 = "jsir.binary_expression"(%4, %5) <{operator_ = "+"}> : (!jsir.any, !jsir.any) -> !jsir.any
// JSHIR-NEXT:         "jsir.expr_region_end"(%6) : (!jsir.any) -> ()
// JSHIR-NEXT:       }, {
// JSHIR-NEXT:         %4 = "jsir.identifier"() <{name = "body1"}> : () -> !jsir.any
// JSHIR-NEXT:         "jsir.expression_statement"(%4) : (!jsir.any) -> ()
// JSHIR-NEXT:         "jshir.break_statement"() : () -> ()
// JSHIR-NEXT:       }) : () -> ()
// JSHIR-NEXT:     }) : (!jsir.any) -> ()
// JSHIR-NEXT:     %2 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSHIR-NEXT:     "jshir.switch_statement"(%2) ({
// JSHIR-NEXT:     ^bb0:
// JSHIR-NEXT:     }) : (!jsir.any) -> ()
// JSHIR-NEXT:     %3 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSHIR-NEXT:     "jshir.switch_statement"(%3) ({
// JSHIR-NEXT:       "jshir.switch_case"() ({
// JSHIR-NEXT:         %4 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "0", 0.000000e+00 : f64>, value = 0.000000e+00 : f64}> : () -> !jsir.any
// JSHIR-NEXT:         "jsir.expr_region_end"(%4) : (!jsir.any) -> ()
// JSHIR-NEXT:       }, {
// JSHIR-NEXT:       ^bb0:
// JSHIR-NEXT:       }) : () -> ()
// JSHIR-NEXT:     }) : (!jsir.any) -> ()
// JSHIR-NEXT:   }, {
// JSHIR-NEXT:   ^bb0:
// JSHIR-NEXT:   }) : () -> ()
// JSHIR-NEXT: }) : () -> ()
