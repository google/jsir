// JSLIR:      "jsir.file"() <{comments = []}> ({
// JSLIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSLIR-NEXT:     %0 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind WhileStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     "cf.br"()[^bb1] : () -> ()
// JSLIR-NEXT:   ^bb1:  // 3 preds: ^bb0, ^bb3, ^bb5
// JSLIR-NEXT:     "jslir.control_flow_marker"(%0) <{kind = #jsir<cf_marker WhileStatementTest>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %1 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     %2 = "builtin.unrealized_conversion_cast"(%1) : (!jsir.any) -> i1
// JSLIR-NEXT:     "cf.cond_br"(%2)[^bb2, ^bb6] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
// JSLIR-NEXT:   ^bb2:  // pred: ^bb1
// JSLIR-NEXT:     "jslir.control_flow_marker"(%0) <{kind = #jsir<cf_marker WhileStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %3 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind BlockStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     "jslir.control_flow_marker"(%3) <{kind = #jsir<cf_marker BlockStatementDirectives>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.control_flow_marker"(%3) <{kind = #jsir<cf_marker BlockStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %4 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSLIR-NEXT:     %5 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind IfStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     %6 = "builtin.unrealized_conversion_cast"(%4) : (!jsir.any) -> i1
// JSLIR-NEXT:     "cf.cond_br"(%6)[^bb3, ^bb5] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
// JSLIR-NEXT:   ^bb3:  // pred: ^bb2
// JSLIR-NEXT:     "jslir.control_flow_marker"(%5) <{kind = #jsir<cf_marker IfStatementConsequent>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.continue_statement"() : () -> ()
// JSLIR-NEXT:     "cf.br"()[^bb1] : () -> ()
// JSLIR-NEXT:   ^bb4:  // no predecessors
// JSLIR-NEXT:     "cf.br"()[^bb5] : () -> ()
// JSLIR-NEXT:   ^bb5:  // 2 preds: ^bb2, ^bb4
// JSLIR-NEXT:     "jslir.control_flow_marker"(%5) <{kind = #jsir<cf_marker IfStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %7 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%7) : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.control_flow_marker"(%3) <{kind = #jsir<cf_marker BlockStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "cf.br"()[^bb1] : () -> ()
// JSLIR-NEXT:   ^bb6:  // pred: ^bb1
// JSLIR-NEXT:     "jslir.control_flow_marker"(%0) <{kind = #jsir<cf_marker WhileStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %8 = "jslir.labeled_statement_start"() <{label = #jsir<identifier   <L 7 C 0>, <L 7 C 6>, "label0", 43, 49, 0, "label0">}> : () -> !jsir.any
// JSLIR-NEXT:     %9 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind WhileStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     "cf.br"()[^bb7] : () -> ()
// JSLIR-NEXT:   ^bb7:  // 3 preds: ^bb6, ^bb11, ^bb14
// JSLIR-NEXT:     "jslir.control_flow_marker"(%9) <{kind = #jsir<cf_marker WhileStatementTest>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %10 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     %11 = "builtin.unrealized_conversion_cast"(%10) : (!jsir.any) -> i1
// JSLIR-NEXT:     "cf.cond_br"(%11)[^bb8, ^bb15] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
// JSLIR-NEXT:   ^bb8:  // pred: ^bb7
// JSLIR-NEXT:     "jslir.control_flow_marker"(%9) <{kind = #jsir<cf_marker WhileStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %12 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind BlockStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     "jslir.control_flow_marker"(%12) <{kind = #jsir<cf_marker BlockStatementDirectives>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.control_flow_marker"(%12) <{kind = #jsir<cf_marker BlockStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %13 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%13) : (!jsir.any) -> ()
// JSLIR-NEXT:     %14 = "jslir.labeled_statement_start"() <{label = #jsir<identifier   <L 9 C 2>, <L 9 C 8>, "label1", 70, 76, 4, "label1">}> : () -> !jsir.any
// JSLIR-NEXT:     %15 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind WhileStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     "cf.br"()[^bb9] : () -> ()
// JSLIR-NEXT:   ^bb9:  // 2 preds: ^bb8, ^bb13
// JSLIR-NEXT:     "jslir.control_flow_marker"(%15) <{kind = #jsir<cf_marker WhileStatementTest>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %16 = "jsir.identifier"() <{name = "d"}> : () -> !jsir.any
// JSLIR-NEXT:     %17 = "builtin.unrealized_conversion_cast"(%16) : (!jsir.any) -> i1
// JSLIR-NEXT:     "cf.cond_br"(%17)[^bb10, ^bb14] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
// JSLIR-NEXT:   ^bb10:  // pred: ^bb9
// JSLIR-NEXT:     "jslir.control_flow_marker"(%15) <{kind = #jsir<cf_marker WhileStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %18 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSLIR-NEXT:     %19 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind IfStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     %20 = "builtin.unrealized_conversion_cast"(%18) : (!jsir.any) -> i1
// JSLIR-NEXT:     "cf.cond_br"(%20)[^bb11, ^bb13] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
// JSLIR-NEXT:   ^bb11:  // pred: ^bb10
// JSLIR-NEXT:     "jslir.control_flow_marker"(%19) <{kind = #jsir<cf_marker IfStatementConsequent>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.continue_statement"() <{label = #jsir<identifier   <L 11 C 15>, <L 11 C 21>, "label0", 114, 120, 5, "label0">}> : () -> ()
// JSLIR-NEXT:     "cf.br"()[^bb7] : () -> ()
// JSLIR-NEXT:   ^bb12:  // no predecessors
// JSLIR-NEXT:     "cf.br"()[^bb13] : () -> ()
// JSLIR-NEXT:   ^bb13:  // 2 preds: ^bb10, ^bb12
// JSLIR-NEXT:     "jslir.control_flow_marker"(%19) <{kind = #jsir<cf_marker IfStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "cf.br"()[^bb9] : () -> ()
// JSLIR-NEXT:   ^bb14:  // pred: ^bb9
// JSLIR-NEXT:     "jslir.control_flow_marker"(%15) <{kind = #jsir<cf_marker WhileStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.control_flow_marker"(%14) <{kind = #jsir<cf_marker LabeledStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.control_flow_marker"(%12) <{kind = #jsir<cf_marker BlockStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "cf.br"()[^bb7] : () -> ()
// JSLIR-NEXT:   ^bb15:  // pred: ^bb7
// JSLIR-NEXT:     "jslir.control_flow_marker"(%9) <{kind = #jsir<cf_marker WhileStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.control_flow_marker"(%8) <{kind = #jsir<cf_marker LabeledStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:   }, {
// JSLIR-NEXT:   ^bb0:
// JSLIR-NEXT:   }) : () -> ()
// JSLIR-NEXT: }) : () -> ()
