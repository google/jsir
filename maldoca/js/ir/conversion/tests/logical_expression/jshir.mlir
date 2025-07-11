// JSHIR:      "jsir.file"() <{comments = []}> ({
// JSHIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSHIR-NEXT:     %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSHIR-NEXT:     %1 = "jshir.logical_expression"(%0) <{operator_ = "&&"}> ({
// JSHIR-NEXT:       %6 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSHIR-NEXT:       "jsir.expr_region_end"(%6) : (!jsir.any) -> ()
// JSHIR-NEXT:     }) : (!jsir.any) -> !jsir.any
// JSHIR-NEXT:     "jsir.expression_statement"(%1) : (!jsir.any) -> ()
// JSHIR-NEXT:     %2 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSHIR-NEXT:     %3 = "jshir.logical_expression"(%2) <{operator_ = "||"}> ({
// JSHIR-NEXT:       %6 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSHIR-NEXT:       "jsir.expr_region_end"(%6) : (!jsir.any) -> ()
// JSHIR-NEXT:     }) : (!jsir.any) -> !jsir.any
// JSHIR-NEXT:     "jsir.expression_statement"(%3) : (!jsir.any) -> ()
// JSHIR-NEXT:     %4 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSHIR-NEXT:     %5 = "jshir.logical_expression"(%4) <{operator_ = "??"}> ({
// JSHIR-NEXT:       %6 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSHIR-NEXT:       "jsir.expr_region_end"(%6) : (!jsir.any) -> ()
// JSHIR-NEXT:     }) : (!jsir.any) -> !jsir.any
// JSHIR-NEXT:     "jsir.expression_statement"(%5) : (!jsir.any) -> ()
// JSHIR-NEXT:   }, {
// JSHIR-NEXT:   ^bb0:
// JSHIR-NEXT:   }) : () -> ()
// JSHIR-NEXT: }) : () -> ()
