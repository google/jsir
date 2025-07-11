// JSLIR:      "jsir.file"() <{comments = []}> ({
// JSLIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSLIR-NEXT:     %0 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind BlockStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     "jslir.control_flow_marker"(%0) <{kind = #jsir<cf_marker BlockStatementDirectives>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.control_flow_marker"(%0) <{kind = #jsir<cf_marker BlockStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %1 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%1) : (!jsir.any) -> ()
// JSLIR-NEXT:     %2 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%2) : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.control_flow_marker"(%0) <{kind = #jsir<cf_marker BlockStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:   }, {
// JSLIR-NEXT:   ^bb0:
// JSLIR-NEXT:   }) : () -> ()
// JSLIR-NEXT: }) : () -> ()
