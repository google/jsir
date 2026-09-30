// JSIR:      "jsir.file"() <{comments = []}> ({
// JSIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSIR-NEXT:     %0 = "jsir.identifier_ref"() <{name = "i"}> : () -> !jsir.any
// JSIR-NEXT:     %1 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "42", 4.200000e+01 : f64>, value = 4.200000e+01 : f64}> : () -> !jsir.any
// JSIR-NEXT:     %2 = "jsir.identifier"() <{name = "obj"}> : () -> !jsir.any
// JSIR-NEXT:     "jshir.for_in_statement"(%0, %1, %2) <{left_declaration = #jsir<for_in_of_declaration <L 1 C 5>, <L 1 C 15>, 5, 15, 1, <L 1 C 9>, <L 1 C 15>, 9, 15, 1, "i", 0, "var">}> ({
// JSIR-NEXT:       "jshir.block_statement"() ({
// JSIR-NEXT:         %3 = "jsir.identifier"() <{name = "foo"}> : () -> !jsir.any
// JSIR-NEXT:         "jsir.expression_statement"(%3) : (!jsir.any) -> ()
// JSIR-NEXT:       }, {
// JSIR-NEXT:       ^bb0:
// JSIR-NEXT:       }) : () -> ()
// JSIR-NEXT:     }) : (!jsir.any, !jsir.any, !jsir.any) -> ()
// JSIR-NEXT:   }, {
// JSIR-NEXT:   ^bb0:
// JSIR-NEXT:   }) : () -> ()
// JSIR-NEXT: }) : () -> ()
