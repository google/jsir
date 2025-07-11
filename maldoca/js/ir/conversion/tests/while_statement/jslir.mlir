// JSLIR:      "jsir.file"() <{comments = []}> ({
// JSLIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSLIR-NEXT:     %0 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind WhileStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     "cf.br"()[^bb1] : () -> ()
// JSLIR-NEXT:   ^bb1:  // 2 preds: ^bb0, ^bb2
// JSLIR-NEXT:     "jslir.control_flow_marker"(%0) <{kind = #jsir<cf_marker WhileStatementTest>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %1 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     %2 = "builtin.unrealized_conversion_cast"(%1) : (!jsir.any) -> i1
// JSLIR-NEXT:     "cf.cond_br"(%2)[^bb2, ^bb3] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
// JSLIR-NEXT:   ^bb2:  // pred: ^bb1
// JSLIR-NEXT:     "jslir.control_flow_marker"(%0) <{kind = #jsir<cf_marker WhileStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %3 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%3) : (!jsir.any) -> ()
// JSLIR-NEXT:     "cf.br"()[^bb1] : () -> ()
// JSLIR-NEXT:   ^bb3:  // pred: ^bb1
// JSLIR-NEXT:     "jslir.control_flow_marker"(%0) <{kind = #jsir<cf_marker WhileStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:   }, {
// JSLIR-NEXT:   ^bb0:
// JSLIR-NEXT:   }) : () -> ()
// JSLIR-NEXT: }) : () -> ()
