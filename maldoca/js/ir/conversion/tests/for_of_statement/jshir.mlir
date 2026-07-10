// JSIR:      "jsir.file"() <{comments = []}> ({
// JSIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSIR-NEXT:     %0 = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:     %1 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSIR-NEXT:     "jshir.for_of_statement"(%0, %1) <{await = false}> ({
// JSIR-NEXT:       %6 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.expression_statement"(%6) : (!jsir.any) -> ()
// JSIR-NEXT:     }) : (!jsir.any, !jsir.any) -> ()
// JSIR-NEXT:     %2 = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:     %3 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSIR-NEXT:     "jshir.for_of_statement"(%2, %3) <{await = false, left_declaration = #jsir<for_in_of_declaration <L 4 C 5>, <L 4 C 10>, 24, 29, 2, <L 4 C 9>, <L 4 C 10>, 28, 29, 2, "a", 2, "let">}> ({
// JSIR-NEXT:       %6 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.expression_statement"(%6) : (!jsir.any) -> ()
// JSIR-NEXT:     }) : (!jsir.any, !jsir.any) -> ()
// JSIR-NEXT:     %4 = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:     %5 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSIR-NEXT:     "jshir.for_of_statement"(%4, %5) <{await = false}> ({
// JSIR-NEXT:       "jshir.block_statement"() ({
// JSIR-NEXT:         %6 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSIR-NEXT:         "jsir.expression_statement"(%6) : (!jsir.any) -> ()
// JSIR-NEXT:       }, {
// JSIR-NEXT:       ^bb0:
// JSIR-NEXT:       }) : () -> ()
// JSIR-NEXT:     }) : (!jsir.any, !jsir.any) -> ()
// JSIR-NEXT:   }, {
// JSIR-NEXT:   ^bb0:
// JSIR-NEXT:   }) : () -> ()
// JSIR-NEXT: }) : () -> ()
