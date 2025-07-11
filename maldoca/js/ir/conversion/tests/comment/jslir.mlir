// JSLIR:      "jsir.file"() <{comments = [#jsir<comment_line  <L 1 C 0>, <L 1 C 4>, 0, 4, " 1">, #jsir<comment_line  <L 3 C 0>, <L 3 C 4>, 8, 12, " 2">, #jsir<comment_block  <L 4 C 2>, <L 4 C 9>, 15, 22, " 3 ">, #jsir<comment_line  <L 5 C 0>, <L 5 C 4>, 24, 28, " 4">]}> ({
// JSLIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSLIR-NEXT:     %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%0) : (!jsir.any) -> ()
// JSLIR-NEXT:     %1 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%1) : (!jsir.any) -> ()
// JSLIR-NEXT:   }, {
// JSLIR-NEXT:   ^bb0:
// JSLIR-NEXT:   }) : () -> ()
// JSLIR-NEXT: }) : () -> ()
