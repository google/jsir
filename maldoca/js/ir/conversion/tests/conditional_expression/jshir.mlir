// JSIR:      "jsir.file"() <{comments = []}> ({
// JSIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSIR-NEXT:     %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:     %1 = "jshir.conditional_expression"(%0) ({
// JSIR-NEXT:       %2 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.expr_region_end"(%2) : (!jsir.any) -> ()
// JSIR-NEXT:     }, {
// JSIR-NEXT:       %2 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.expr_region_end"(%2) : (!jsir.any) -> ()
// JSIR-NEXT:     }) : (!jsir.any) -> !jsir.any
// JSIR-NEXT:     "jsir.expression_statement"(%1) : (!jsir.any) -> ()
// JSIR-NEXT:   }, {
// JSIR-NEXT:   ^bb0:
// JSIR-NEXT:   }) : () -> ()
// JSIR-NEXT: }) : () -> ()
