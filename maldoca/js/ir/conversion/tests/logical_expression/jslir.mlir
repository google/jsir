// JSLIR:      "jsir.file"() <{comments = []}> ({
// JSLIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSLIR-NEXT:     %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     %1 = "jslir.logical_expression_start"(%0) <{operator_ = "&&"}> : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:     %2 = "builtin.unrealized_conversion_cast"(%0) : (!jsir.any) -> i1
// JSLIR-NEXT:     "cf.cond_br"(%2, %0)[^bb1, ^bb2] <{operandSegmentSizes = array<i32: 1, 0, 1>}> : (i1, !jsir.any) -> ()
// JSLIR-NEXT:   ^bb1:  // pred: ^bb0
// JSLIR-NEXT:     "jslir.control_flow_marker"(%1) <{kind = #jsir<cf_marker LogicalExpressionRight>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %3 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSLIR-NEXT:     "cf.br"(%3)[^bb2] : (!jsir.any) -> ()
// JSLIR-NEXT:   ^bb2(%4: !jsir.any):  // 2 preds: ^bb0, ^bb1
// JSLIR-NEXT:     "jslir.control_flow_marker"(%1) <{kind = #jsir<cf_marker LogicalExpressionEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "jsir.expression_statement"(%4) : (!jsir.any) -> ()
// JSLIR-NEXT:     %5 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     %6 = "jslir.logical_expression_start"(%5) <{operator_ = "||"}> : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:     %7 = "builtin.unrealized_conversion_cast"(%5) : (!jsir.any) -> i1
// JSLIR-NEXT:     "cf.cond_br"(%7, %5)[^bb4, ^bb3] <{operandSegmentSizes = array<i32: 1, 1, 0>}> : (i1, !jsir.any) -> ()
// JSLIR-NEXT:   ^bb3:  // pred: ^bb2
// JSLIR-NEXT:     "jslir.control_flow_marker"(%6) <{kind = #jsir<cf_marker LogicalExpressionRight>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %8 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSLIR-NEXT:     "cf.br"(%8)[^bb4] : (!jsir.any) -> ()
// JSLIR-NEXT:   ^bb4(%9: !jsir.any):  // 2 preds: ^bb2, ^bb3
// JSLIR-NEXT:     "jslir.control_flow_marker"(%6) <{kind = #jsir<cf_marker LogicalExpressionEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "jsir.expression_statement"(%9) : (!jsir.any) -> ()
// JSLIR-NEXT:     %10 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     %11 = "jslir.logical_expression_start"(%10) <{operator_ = "??"}> : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:     %12 = "jsir.null_literal"() : () -> !jsir.any
// JSLIR-NEXT:     %13 = "jsir.binary_expression"(%10, %12) <{operator_ = "=="}> : (!jsir.any, !jsir.any) -> !jsir.any
// JSLIR-NEXT:     %14 = "builtin.unrealized_conversion_cast"(%13) : (!jsir.any) -> i1
// JSLIR-NEXT:     "cf.cond_br"(%14, %10)[^bb5, ^bb6] <{operandSegmentSizes = array<i32: 1, 0, 1>}> : (i1, !jsir.any) -> ()
// JSLIR-NEXT:   ^bb5:  // pred: ^bb4
// JSLIR-NEXT:     "jslir.control_flow_marker"(%11) <{kind = #jsir<cf_marker LogicalExpressionRight>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %15 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSLIR-NEXT:     "cf.br"(%15)[^bb6] : (!jsir.any) -> ()
// JSLIR-NEXT:   ^bb6(%16: !jsir.any):  // 2 preds: ^bb4, ^bb5
// JSLIR-NEXT:     "jslir.control_flow_marker"(%11) <{kind = #jsir<cf_marker LogicalExpressionEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "jsir.expression_statement"(%16) : (!jsir.any) -> ()
// JSLIR-NEXT:   }, {
// JSLIR-NEXT:   ^bb0:
// JSLIR-NEXT:   }) : () -> ()
// JSLIR-NEXT: }) : () -> ()
