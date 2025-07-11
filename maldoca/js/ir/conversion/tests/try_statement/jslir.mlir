// JSLIR:      "jsir.file"() <{comments = []}> ({
// JSLIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSLIR-NEXT:     %0 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind TryStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     "cf.br"()[^bb1] : () -> ()
// JSLIR-NEXT:   ^bb1:  // pred: ^bb0
// JSLIR-NEXT:     "jslir.control_flow_marker"(%0) <{kind = #jsir<cf_marker TryStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %1 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind BlockStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     "jslir.control_flow_marker"(%1) <{kind = #jsir<cf_marker BlockStatementDirectives>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.control_flow_marker"(%1) <{kind = #jsir<cf_marker BlockStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %2 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%2) : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.control_flow_marker"(%1) <{kind = #jsir<cf_marker BlockStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "cf.br"()[^bb3] : () -> ()
// JSLIR-NEXT:   ^bb2:  // no predecessors
// JSLIR-NEXT:     "jslir.control_flow_marker"(%0) <{kind = #jsir<cf_marker TryStatementHandler>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %3 = "jsir.identifier_ref"() <{name = "error"}> : () -> !jsir.any
// JSLIR-NEXT:     "jslir.catch_clause_start"(%3) : (!jsir.any) -> ()
// JSLIR-NEXT:     %4 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind BlockStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     "jslir.control_flow_marker"(%4) <{kind = #jsir<cf_marker BlockStatementDirectives>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.control_flow_marker"(%4) <{kind = #jsir<cf_marker BlockStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %5 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%5) : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.control_flow_marker"(%4) <{kind = #jsir<cf_marker BlockStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "cf.br"()[^bb3] : () -> ()
// JSLIR-NEXT:   ^bb3:  // 2 preds: ^bb1, ^bb2
// JSLIR-NEXT:     "jslir.control_flow_marker"(%0) <{kind = #jsir<cf_marker TryStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %6 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind TryStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     "cf.br"()[^bb4] : () -> ()
// JSLIR-NEXT:   ^bb4:  // pred: ^bb3
// JSLIR-NEXT:     "jslir.control_flow_marker"(%6) <{kind = #jsir<cf_marker TryStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %7 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind BlockStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     "jslir.control_flow_marker"(%7) <{kind = #jsir<cf_marker BlockStatementDirectives>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.control_flow_marker"(%7) <{kind = #jsir<cf_marker BlockStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %8 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%8) : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.control_flow_marker"(%7) <{kind = #jsir<cf_marker BlockStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "cf.br"()[^bb5] : () -> ()
// JSLIR-NEXT:   ^bb5:  // pred: ^bb4
// JSLIR-NEXT:     "jslir.control_flow_marker"(%6) <{kind = #jsir<cf_marker TryStatementFinalizer>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %9 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind BlockStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     "jslir.control_flow_marker"(%9) <{kind = #jsir<cf_marker BlockStatementDirectives>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.control_flow_marker"(%9) <{kind = #jsir<cf_marker BlockStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %10 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%10) : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.control_flow_marker"(%9) <{kind = #jsir<cf_marker BlockStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "cf.br"()[^bb6] : () -> ()
// JSLIR-NEXT:   ^bb6:  // pred: ^bb5
// JSLIR-NEXT:     "jslir.control_flow_marker"(%6) <{kind = #jsir<cf_marker TryStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %11 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind TryStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     "cf.br"()[^bb7] : () -> ()
// JSLIR-NEXT:   ^bb7:  // pred: ^bb6
// JSLIR-NEXT:     "jslir.control_flow_marker"(%11) <{kind = #jsir<cf_marker TryStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %12 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind BlockStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     "jslir.control_flow_marker"(%12) <{kind = #jsir<cf_marker BlockStatementDirectives>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.control_flow_marker"(%12) <{kind = #jsir<cf_marker BlockStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %13 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%13) : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.control_flow_marker"(%12) <{kind = #jsir<cf_marker BlockStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "cf.br"()[^bb9] : () -> ()
// JSLIR-NEXT:   ^bb8:  // no predecessors
// JSLIR-NEXT:     "jslir.control_flow_marker"(%11) <{kind = #jsir<cf_marker TryStatementHandler>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %14 = "jsir.identifier_ref"() <{name = "error"}> : () -> !jsir.any
// JSLIR-NEXT:     "jslir.catch_clause_start"(%14) : (!jsir.any) -> ()
// JSLIR-NEXT:     %15 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind BlockStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     "jslir.control_flow_marker"(%15) <{kind = #jsir<cf_marker BlockStatementDirectives>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.control_flow_marker"(%15) <{kind = #jsir<cf_marker BlockStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %16 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%16) : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.control_flow_marker"(%15) <{kind = #jsir<cf_marker BlockStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "cf.br"()[^bb9] : () -> ()
// JSLIR-NEXT:   ^bb9:  // 2 preds: ^bb7, ^bb8
// JSLIR-NEXT:     "jslir.control_flow_marker"(%11) <{kind = #jsir<cf_marker TryStatementFinalizer>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %17 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind BlockStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     "jslir.control_flow_marker"(%17) <{kind = #jsir<cf_marker BlockStatementDirectives>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.control_flow_marker"(%17) <{kind = #jsir<cf_marker BlockStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %18 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%18) : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.control_flow_marker"(%17) <{kind = #jsir<cf_marker BlockStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "cf.br"()[^bb10] : () -> ()
// JSLIR-NEXT:   ^bb10:  // pred: ^bb9
// JSLIR-NEXT:     "jslir.control_flow_marker"(%11) <{kind = #jsir<cf_marker TryStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:   }, {
// JSLIR-NEXT:   ^bb0:
// JSLIR-NEXT:   }) : () -> ()
// JSLIR-NEXT: }) : () -> ()
