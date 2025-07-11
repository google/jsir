// JSLIR:      "jsir.file"() <{comments = []}> ({
// JSLIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSLIR-NEXT:     %0 = "jslir.labeled_statement_start"() <{label = #jsir<identifier   <L 1 C 0>, <L 1 C 5>, "label", 0, 5, 0, "label">}> : () -> !jsir.any
// JSLIR-NEXT:     %1 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     %2 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind IfStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     %3 = "builtin.unrealized_conversion_cast"(%1) : (!jsir.any) -> i1
// JSLIR-NEXT:     "cf.cond_br"(%3)[^bb1, ^bb2] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
// JSLIR-NEXT:   ^bb1:  // pred: ^bb0
// JSLIR-NEXT:     "jslir.control_flow_marker"(%2) <{kind = #jsir<cf_marker IfStatementConsequent>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %4 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%4) : (!jsir.any) -> ()
// JSLIR-NEXT:     "cf.br"()[^bb2] : () -> ()
// JSLIR-NEXT:   ^bb2:  // 2 preds: ^bb0, ^bb1
// JSLIR-NEXT:     "jslir.control_flow_marker"(%2) <{kind = #jsir<cf_marker IfStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.control_flow_marker"(%0) <{kind = #jsir<cf_marker LabeledStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:   }, {
// JSLIR-NEXT:   ^bb0:
// JSLIR-NEXT:   }) : () -> ()
// JSLIR-NEXT: }) : () -> ()
