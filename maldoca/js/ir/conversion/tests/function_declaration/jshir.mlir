// JSHIR:      "jsir.file"() <{comments = []}> ({
// JSHIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSHIR-NEXT:     "jsir.function_declaration"() <{async = false, generator = false, id = #jsir<identifier   <L 1 C 9>, <L 1 C 12>, "foo", 9, 12, 1, "foo">}> ({
// JSHIR-NEXT:       %0 = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
// JSHIR-NEXT:       "jsir.exprs_region_end"(%0) : (!jsir.any) -> ()
// JSHIR-NEXT:     }, {
// JSHIR-NEXT:       "jshir.block_statement"() ({
// JSHIR-NEXT:         %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSHIR-NEXT:         "jsir.return_statement"(%0) : (!jsir.any) -> ()
// JSHIR-NEXT:       }, {
// JSHIR-NEXT:       ^bb0:
// JSHIR-NEXT:       }) : () -> ()
// JSHIR-NEXT:     }) : () -> ()
// JSHIR-NEXT:     "jsir.function_declaration"() <{async = false, generator = false, id = #jsir<identifier   <L 5 C 9>, <L 5 C 12>, "bar", 42, 45, 2, "bar">}> ({
// JSHIR-NEXT:       %0 = "jsir.identifier_ref"() <{name = "b"}> : () -> !jsir.any
// JSHIR-NEXT:       %1 = "jsir.identifier"() <{name = "some_computation"}> : () -> !jsir.any
// JSHIR-NEXT:       %2 = "jsir.call_expression"(%1) : (!jsir.any) -> !jsir.any
// JSHIR-NEXT:       %3 = "jsir.assignment_pattern_ref"(%0, %2) : (!jsir.any, !jsir.any) -> !jsir.any
// JSHIR-NEXT:       "jsir.exprs_region_end"(%3) : (!jsir.any) -> ()
// JSHIR-NEXT:     }, {
// JSHIR-NEXT:       "jshir.block_statement"() ({
// JSHIR-NEXT:         %0 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSHIR-NEXT:         "jsir.return_statement"(%0) : (!jsir.any) -> ()
// JSHIR-NEXT:       }, {
// JSHIR-NEXT:       ^bb0:
// JSHIR-NEXT:       }) : () -> ()
// JSHIR-NEXT:     }) : () -> ()
// JSHIR-NEXT:   }, {
// JSHIR-NEXT:   ^bb0:
// JSHIR-NEXT:   }) : () -> ()
// JSHIR-NEXT: }) : () -> ()
