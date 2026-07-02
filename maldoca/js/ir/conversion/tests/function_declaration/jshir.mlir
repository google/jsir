// JSIR:      "jsir.file"() <{comments = []}> ({
// JSIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSIR-NEXT:     "jsir.function_declaration"() <{async = false, generator = false, id = #jsir<identifier <L 1 C 9>, <L 1 C 12>, "foo", 9, 12, 1, "foo">}> ({
// JSIR-NEXT:       %0 = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.exprs_region_end"(%0) : (!jsir.any) -> ()
// JSIR-NEXT:     }, {
// JSIR-NEXT:       "jshir.block_statement"() ({
// JSIR-NEXT:         %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:         "jsir.return_statement"(%0) : (!jsir.any) -> ()
// JSIR-NEXT:       }, {
// JSIR-NEXT:       ^bb0:
// JSIR-NEXT:       }) : () -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:     "jsir.function_declaration"() <{async = false, generator = false, id = #jsir<identifier <L 5 C 9>, <L 5 C 12>, "bar", 42, 45, 2, "bar">}> ({
// JSIR-NEXT:       %0 = "jsir.identifier_ref"() <{name = "b"}> : () -> !jsir.any
// JSIR-NEXT:       %1 = "jsir.identifier"() <{name = "some_computation"}> : () -> !jsir.any
// JSIR-NEXT:       %2 = "jsir.call_expression"(%1) : (!jsir.any) -> !jsir.any
// JSIR-NEXT:       %3 = "jsir.assignment_pattern_ref"(%0, %2) : (!jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:       "jsir.exprs_region_end"(%3) : (!jsir.any) -> ()
// JSIR-NEXT:     }, {
// JSIR-NEXT:       "jshir.block_statement"() ({
// JSIR-NEXT:         %0 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSIR-NEXT:         "jsir.return_statement"(%0) : (!jsir.any) -> ()
// JSIR-NEXT:       }, {
// JSIR-NEXT:       ^bb0:
// JSIR-NEXT:       }) : () -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:   }, {
// JSIR-NEXT:   ^bb0:
// JSIR-NEXT:   }) : () -> ()
// JSIR-NEXT: }) : () -> ()
