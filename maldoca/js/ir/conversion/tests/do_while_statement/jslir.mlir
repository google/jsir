// JSLIR:      "jsir.file"() <{comments = []}> ({
// JSLIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSLIR-NEXT:     %0 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind DoWhileStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     "cf.br"()[^bb1] : () -> ()
// JSLIR-NEXT:   ^bb1:  // 2 preds: ^bb0, ^bb2
// JSLIR-NEXT:     "jslir.control_flow_marker"(%0) <{kind = #jsir<cf_marker DoWhileStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %1 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%1) : (!jsir.any) -> ()
// JSLIR-NEXT:     "cf.br"()[^bb2] : () -> ()
// JSLIR-NEXT:   ^bb2:  // pred: ^bb1
// JSLIR-NEXT:     "jslir.control_flow_marker"(%0) <{kind = #jsir<cf_marker DoWhileStatementTest>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %2 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSLIR-NEXT:     %3 = "builtin.unrealized_conversion_cast"(%2) : (!jsir.any) -> i1
// JSLIR-NEXT:     "cf.cond_br"(%3)[^bb1, ^bb3] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
// JSLIR-NEXT:   ^bb3:  // pred: ^bb2
// JSLIR-NEXT:     "jslir.control_flow_marker"(%0) <{kind = #jsir<cf_marker DoWhileStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %4 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind DoWhileStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     "cf.br"()[^bb4] : () -> ()
// JSLIR-NEXT:   ^bb4:  // 2 preds: ^bb3, ^bb5
// JSLIR-NEXT:     "jslir.control_flow_marker"(%4) <{kind = #jsir<cf_marker DoWhileStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %5 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind BlockStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     "jslir.control_flow_marker"(%5) <{kind = #jsir<cf_marker BlockStatementDirectives>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.control_flow_marker"(%5) <{kind = #jsir<cf_marker BlockStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %6 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%6) : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.control_flow_marker"(%5) <{kind = #jsir<cf_marker BlockStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "cf.br"()[^bb5] : () -> ()
// JSLIR-NEXT:   ^bb5:  // pred: ^bb4
// JSLIR-NEXT:     "jslir.control_flow_marker"(%4) <{kind = #jsir<cf_marker DoWhileStatementTest>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %7 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSLIR-NEXT:     %8 = "builtin.unrealized_conversion_cast"(%7) : (!jsir.any) -> i1
// JSLIR-NEXT:     "cf.cond_br"(%8)[^bb4, ^bb6] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
// JSLIR-NEXT:   ^bb6:  // pred: ^bb5
// JSLIR-NEXT:     "jslir.control_flow_marker"(%4) <{kind = #jsir<cf_marker DoWhileStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:   }, {
// JSLIR-NEXT:   ^bb0:
// JSLIR-NEXT:   }) : () -> ()
// JSLIR-NEXT: }) : () -> ()
