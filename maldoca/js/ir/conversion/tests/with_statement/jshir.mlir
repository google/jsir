// JSIR:      "jsir.file"() <{comments = []}> ({
// JSIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSIR-NEXT:     %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:     "jshir.with_statement"(%0) ({
// JSIR-NEXT:       %2 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.expression_statement"(%2) : (!jsir.any) -> ()
// JSIR-NEXT:     }) : (!jsir.any) -> ()
// JSIR-NEXT:     %1 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:     "jshir.with_statement"(%1) ({
// JSIR-NEXT:       "jshir.block_statement"() ({
// JSIR-NEXT:         %2 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSIR-NEXT:         "jsir.expression_statement"(%2) : (!jsir.any) -> ()
// JSIR-NEXT:         %3 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSIR-NEXT:         "jsir.expression_statement"(%3) : (!jsir.any) -> ()
// JSIR-NEXT:       }, {
// JSIR-NEXT:       ^bb0:
// JSIR-NEXT:       }) : () -> ()
// JSIR-NEXT:     }) : (!jsir.any) -> ()
// JSIR-NEXT:   }, {
// JSIR-NEXT:   ^bb0:
// JSIR-NEXT:   }) : () -> ()
// JSIR-NEXT: }) : () -> ()
