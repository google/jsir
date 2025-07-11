// JSLIR:      "jsir.file"() <{comments = []}> ({
// JSLIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSLIR-NEXT:     %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     %1 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind ConditionalExpression>}> : () -> !jsir.any
// JSLIR-NEXT:     %2 = "builtin.unrealized_conversion_cast"(%0) : (!jsir.any) -> i1
// JSLIR-NEXT:     "cf.cond_br"(%2)[^bb2, ^bb1] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
// JSLIR-NEXT:   ^bb1:  // pred: ^bb0
// JSLIR-NEXT:     "jslir.control_flow_marker"(%1) <{kind = #jsir<cf_marker ConditionalExpressionAlternate>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %3 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSLIR-NEXT:     "cf.br"(%3)[^bb3] : (!jsir.any) -> ()
// JSLIR-NEXT:   ^bb2:  // pred: ^bb0
// JSLIR-NEXT:     "jslir.control_flow_marker"(%1) <{kind = #jsir<cf_marker ConditionalExpressionConsequent>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %4 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSLIR-NEXT:     "cf.br"(%4)[^bb3] : (!jsir.any) -> ()
// JSLIR-NEXT:   ^bb3(%5: !jsir.any):  // 2 preds: ^bb1, ^bb2
// JSLIR-NEXT:     "jslir.control_flow_marker"(%1) <{kind = #jsir<cf_marker ConditionalExpressionEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "jsir.expression_statement"(%5) : (!jsir.any) -> ()
// JSLIR-NEXT:   }, {
// JSLIR-NEXT:   ^bb0:
// JSLIR-NEXT:   }) : () -> ()
// JSLIR-NEXT: }) : () -> ()
