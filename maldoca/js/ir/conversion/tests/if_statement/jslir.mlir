// JSLIR:      "jsir.file"() <{comments = []}> ({
// JSLIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSLIR-NEXT:     %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     %1 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind IfStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     %2 = "builtin.unrealized_conversion_cast"(%0) : (!jsir.any) -> i1
// JSLIR-NEXT:     "cf.cond_br"(%2)[^bb1, ^bb2] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
// JSLIR-NEXT:   ^bb1:  // pred: ^bb0
// JSLIR-NEXT:     "jslir.control_flow_marker"(%1) <{kind = #jsir<cf_marker IfStatementConsequent>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %3 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%3) : (!jsir.any) -> ()
// JSLIR-NEXT:     "cf.br"()[^bb2] : () -> ()
// JSLIR-NEXT:   ^bb2:  // 2 preds: ^bb0, ^bb1
// JSLIR-NEXT:     "jslir.control_flow_marker"(%1) <{kind = #jsir<cf_marker IfStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %4 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     %5 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind IfStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     %6 = "builtin.unrealized_conversion_cast"(%4) : (!jsir.any) -> i1
// JSLIR-NEXT:     "cf.cond_br"(%6)[^bb3, ^bb4] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
// JSLIR-NEXT:   ^bb3:  // pred: ^bb2
// JSLIR-NEXT:     "jslir.control_flow_marker"(%5) <{kind = #jsir<cf_marker IfStatementConsequent>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %7 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind BlockStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     "jslir.control_flow_marker"(%7) <{kind = #jsir<cf_marker BlockStatementDirectives>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.control_flow_marker"(%7) <{kind = #jsir<cf_marker BlockStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %8 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%8) : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.control_flow_marker"(%7) <{kind = #jsir<cf_marker BlockStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "cf.br"()[^bb5] : () -> ()
// JSLIR-NEXT:   ^bb4:  // pred: ^bb2
// JSLIR-NEXT:     "jslir.control_flow_marker"(%5) <{kind = #jsir<cf_marker IfStatementAlternate>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %9 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%9) : (!jsir.any) -> ()
// JSLIR-NEXT:     "cf.br"()[^bb5] : () -> ()
// JSLIR-NEXT:   ^bb5:  // 2 preds: ^bb3, ^bb4
// JSLIR-NEXT:     "jslir.control_flow_marker"(%5) <{kind = #jsir<cf_marker IfStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:   }, {
// JSLIR-NEXT:   ^bb0:
// JSLIR-NEXT:   }) : () -> ()
// JSLIR-NEXT: }) : () -> ()
