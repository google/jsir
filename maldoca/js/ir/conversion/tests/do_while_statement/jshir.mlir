// JSIR:      "jsir.file"() <{comments = []}> ({
// JSIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSIR-NEXT:     "jshir.do_while_statement"() ({
// JSIR-NEXT:       %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.expression_statement"(%0) : (!jsir.any) -> ()
// JSIR-NEXT:     }, {
// JSIR-NEXT:       %0 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.expr_region_end"(%0) : (!jsir.any) -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:     "jshir.do_while_statement"() ({
// JSIR-NEXT:       "jshir.block_statement"() ({
// JSIR-NEXT:         %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:         "jsir.expression_statement"(%0) : (!jsir.any) -> ()
// JSIR-NEXT:       }, {
// JSIR-NEXT:       ^bb0:
// JSIR-NEXT:       }) : () -> ()
// JSIR-NEXT:     }, {
// JSIR-NEXT:       %0 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.expr_region_end"(%0) : (!jsir.any) -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:   }, {
// JSIR-NEXT:   ^bb0:
// JSIR-NEXT:   }) : () -> ()
// JSIR-NEXT: }) : () -> ()
