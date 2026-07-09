// JSIR:      "jsir.file"() <{comments = []}> ({
// JSIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSIR-NEXT:     "jshir.labeled_statement"() <{label = #jsir<identifier <L 1 C 0>, <L 1 C 5>, "label", 0, 5, 0, "label">}> ({
// JSIR-NEXT:       %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:       "jshir.if_statement"(%0) ({
// JSIR-NEXT:         %1 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSIR-NEXT:         "jsir.expression_statement"(%1) : (!jsir.any) -> ()
// JSIR-NEXT:       }, {
// JSIR-NEXT:       }) : (!jsir.any) -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:   }, {
// JSIR-NEXT:   ^bb0:
// JSIR-NEXT:   }) : () -> ()
// JSIR-NEXT: }) : () -> ()
