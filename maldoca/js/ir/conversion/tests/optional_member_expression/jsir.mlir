// JSIR:      "jsir.file"() <{comments = []}> ({
// JSIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSIR-NEXT:     %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:     %1 = "jsir.optional_member_expression"(%0) <{literal_property = #jsir<identifier <L 1 C 3>, <L 1 C 4>, "b", 3, 4, 0, "b">, optional = true}> : (!jsir.any) -> !jsir.any
// JSIR-NEXT:     "jsir.expression_statement"(%1) : (!jsir.any) -> ()
// JSIR-NEXT:     %2 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:     %3 = "jsir.member_expression"(%2) <{literal_property = #jsir<identifier <L 3 C 2>, <L 3 C 3>, "b", 9, 10, 0, "b">}> : (!jsir.any) -> !jsir.any
// JSIR-NEXT:     %4 = "jsir.optional_member_expression"(%3) <{literal_property = #jsir<identifier <L 3 C 5>, <L 3 C 6>, "c", 12, 13, 0, "c">, optional = true}> : (!jsir.any) -> !jsir.any
// JSIR-NEXT:     "jsir.expression_statement"(%4) : (!jsir.any) -> ()
// JSIR-NEXT:     %5 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:     %6 = "jsir.optional_member_expression"(%5) <{literal_property = #jsir<identifier <L 5 C 3>, <L 5 C 4>, "b", 19, 20, 0, "b">, optional = true}> : (!jsir.any) -> !jsir.any
// JSIR-NEXT:     %7 = "jsir.optional_member_expression"(%6) <{literal_property = #jsir<identifier <L 5 C 5>, <L 5 C 6>, "c", 21, 22, 0, "c">, optional = false}> : (!jsir.any) -> !jsir.any
// JSIR-NEXT:     "jsir.expression_statement"(%7) : (!jsir.any) -> ()
// JSIR-NEXT:     %8 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:     %9 = "jsir.optional_member_expression"(%8) <{literal_property = #jsir<identifier <L 7 C 3>, <L 7 C 4>, "b", 28, 29, 0, "b">, optional = true}> : (!jsir.any) -> !jsir.any
// JSIR-NEXT:     %10 = "jsir.optional_member_expression"(%9) <{literal_property = #jsir<identifier <L 7 C 6>, <L 7 C 7>, "c", 31, 32, 0, "c">, optional = true}> : (!jsir.any) -> !jsir.any
// JSIR-NEXT:     "jsir.expression_statement"(%10) : (!jsir.any) -> ()
// JSIR-NEXT:   }, {
// JSIR-NEXT:   ^bb0:
// JSIR-NEXT:   }) : () -> ()
// JSIR-NEXT: }) : () -> ()
