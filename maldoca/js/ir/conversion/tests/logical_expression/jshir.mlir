// JSIR:      "jsir.file"() <{comments = []}> ({
// JSIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSIR-NEXT:     %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:     %1 = "jshir.logical_expression"(%0) <{operator_ = "&&"}> ({
// JSIR-NEXT:       %6 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.expr_region_end"(%6) : (!jsir.any) -> ()
// JSIR-NEXT:     }) : (!jsir.any) -> !jsir.any
// JSIR-NEXT:     "jsir.expression_statement"(%1) : (!jsir.any) -> ()
// JSIR-NEXT:     %2 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:     %3 = "jshir.logical_expression"(%2) <{operator_ = "||"}> ({
// JSIR-NEXT:       %6 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.expr_region_end"(%6) : (!jsir.any) -> ()
// JSIR-NEXT:     }) : (!jsir.any) -> !jsir.any
// JSIR-NEXT:     "jsir.expression_statement"(%3) : (!jsir.any) -> ()
// JSIR-NEXT:     %4 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:     %5 = "jshir.logical_expression"(%4) <{operator_ = "??"}> ({
// JSIR-NEXT:       %6 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.expr_region_end"(%6) : (!jsir.any) -> ()
// JSIR-NEXT:     }) : (!jsir.any) -> !jsir.any
// JSIR-NEXT:     "jsir.expression_statement"(%5) : (!jsir.any) -> ()
// JSIR-NEXT:   }, {
// JSIR-NEXT:   ^bb0:
// JSIR-NEXT:   }) : () -> ()
// JSIR-NEXT: }) : () -> ()
