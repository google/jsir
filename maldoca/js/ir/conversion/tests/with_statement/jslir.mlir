// JSLIR:      "jsir.file"() <{comments = []}> ({
// JSLIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSLIR-NEXT:     %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     %1 = "jslir.with_statement_start"(%0) : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:     "cf.br"()[^bb1] : () -> ()
// JSLIR-NEXT:   ^bb1:  // pred: ^bb0
// JSLIR-NEXT:     "jslir.control_flow_marker"(%1) <{kind = #jsir<cf_marker WithStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %2 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%2) : (!jsir.any) -> ()
// JSLIR-NEXT:     "cf.br"()[^bb2] : () -> ()
// JSLIR-NEXT:   ^bb2:  // pred: ^bb1
// JSLIR-NEXT:     "jslir.control_flow_marker"(%1) <{kind = #jsir<cf_marker WithStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %3 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     %4 = "jslir.with_statement_start"(%3) : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:     "cf.br"()[^bb3] : () -> ()
// JSLIR-NEXT:   ^bb3:  // pred: ^bb2
// JSLIR-NEXT:     "jslir.control_flow_marker"(%4) <{kind = #jsir<cf_marker WithStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %5 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind BlockStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     "jslir.control_flow_marker"(%5) <{kind = #jsir<cf_marker BlockStatementDirectives>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.control_flow_marker"(%5) <{kind = #jsir<cf_marker BlockStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %6 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%6) : (!jsir.any) -> ()
// JSLIR-NEXT:     %7 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%7) : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.control_flow_marker"(%5) <{kind = #jsir<cf_marker BlockStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "cf.br"()[^bb4] : () -> ()
// JSLIR-NEXT:   ^bb4:  // pred: ^bb3
// JSLIR-NEXT:     "jslir.control_flow_marker"(%4) <{kind = #jsir<cf_marker WithStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:   }, {
// JSLIR-NEXT:   ^bb0:
// JSLIR-NEXT:   }) : () -> ()
// JSLIR-NEXT: }) : () -> ()
