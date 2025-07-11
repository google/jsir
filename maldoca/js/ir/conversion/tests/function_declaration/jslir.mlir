// JSLIR:      "jsir.file"() <{comments = []}> ({
// JSLIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSLIR-NEXT:     "jsir.function_declaration"() <{async = false, generator = false, id = #jsir<identifier   <L 1 C 9>, <L 1 C 12>, "foo", 9, 12, 1, "foo">}> ({
// JSLIR-NEXT:       %0 = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:       "jsir.exprs_region_end"(%0) : (!jsir.any) -> ()
// JSLIR-NEXT:     }, {
// JSLIR-NEXT:       %0 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind BlockStatement>}> : () -> !jsir.any
// JSLIR-NEXT:       "jslir.control_flow_marker"(%0) <{kind = #jsir<cf_marker BlockStatementDirectives>}> : (!jsir.any) -> ()
// JSLIR-NEXT:       "jslir.control_flow_marker"(%0) <{kind = #jsir<cf_marker BlockStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:       %1 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:       "jsir.return_statement"(%1) : (!jsir.any) -> ()
// JSLIR-NEXT:       "jslir.control_flow_marker"(%0) <{kind = #jsir<cf_marker BlockStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     }) : () -> ()
// JSLIR-NEXT:     "jsir.function_declaration"() <{async = false, generator = false, id = #jsir<identifier   <L 5 C 9>, <L 5 C 12>, "bar", 42, 45, 2, "bar">}> ({
// JSLIR-NEXT:       %0 = "jsir.identifier_ref"() <{name = "b"}> : () -> !jsir.any
// JSLIR-NEXT:       %1 = "jsir.identifier"() <{name = "some_computation"}> : () -> !jsir.any
// JSLIR-NEXT:       %2 = "jsir.call_expression"(%1) : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:       %3 = "jsir.assignment_pattern_ref"(%0, %2) : (!jsir.any, !jsir.any) -> !jsir.any
// JSLIR-NEXT:       "jsir.exprs_region_end"(%3) : (!jsir.any) -> ()
// JSLIR-NEXT:     }, {
// JSLIR-NEXT:       %0 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind BlockStatement>}> : () -> !jsir.any
// JSLIR-NEXT:       "jslir.control_flow_marker"(%0) <{kind = #jsir<cf_marker BlockStatementDirectives>}> : (!jsir.any) -> ()
// JSLIR-NEXT:       "jslir.control_flow_marker"(%0) <{kind = #jsir<cf_marker BlockStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:       %1 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSLIR-NEXT:       "jsir.return_statement"(%1) : (!jsir.any) -> ()
// JSLIR-NEXT:       "jslir.control_flow_marker"(%0) <{kind = #jsir<cf_marker BlockStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     }) : () -> ()
// JSLIR-NEXT:   }, {
// JSLIR-NEXT:   ^bb0:
// JSLIR-NEXT:   }) : () -> ()
// JSLIR-NEXT: }) : () -> ()
