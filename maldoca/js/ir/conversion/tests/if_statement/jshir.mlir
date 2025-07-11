// JSHIR:      "jsir.file"() <{comments = []}> ({
// JSHIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSHIR-NEXT:     %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSHIR-NEXT:     "jshir.if_statement"(%0) ({
// JSHIR-NEXT:       %2 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSHIR-NEXT:       "jsir.expression_statement"(%2) : (!jsir.any) -> ()
// JSHIR-NEXT:     }, {
// JSHIR-NEXT:     }) : (!jsir.any) -> ()
// JSHIR-NEXT:     %1 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSHIR-NEXT:     "jshir.if_statement"(%1) ({
// JSHIR-NEXT:       "jshir.block_statement"() ({
// JSHIR-NEXT:         %2 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSHIR-NEXT:         "jsir.expression_statement"(%2) : (!jsir.any) -> ()
// JSHIR-NEXT:       }, {
// JSHIR-NEXT:       ^bb0:
// JSHIR-NEXT:       }) : () -> ()
// JSHIR-NEXT:     }, {
// JSHIR-NEXT:       %2 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSHIR-NEXT:       "jsir.expression_statement"(%2) : (!jsir.any) -> ()
// JSHIR-NEXT:     }) : (!jsir.any) -> ()
// JSHIR-NEXT:   }, {
// JSHIR-NEXT:   ^bb0:
// JSHIR-NEXT:   }) : () -> ()
// JSHIR-NEXT: }) : () -> ()
