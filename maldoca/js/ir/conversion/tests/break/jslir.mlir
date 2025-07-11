// JSLIR:      "jsir.file"() <{comments = [#jsir<comment_line  <L 1 C 0>, <L 1 C 80>, 0, 80, " =============================================================================">, #jsir<comment_line  <L 2 C 0>, <L 2 C 31>, 81, 112, " Breaking out of a while loop">, #jsir<comment_line  <L 3 C 0>, <L 3 C 80>, 113, 193, " =============================================================================">, #jsir<comment_line  <L 11 C 0>, <L 11 C 80>, 235, 315, " =============================================================================">, #jsir<comment_line  <L 12 C 0>, <L 12 C 36>, 316, 352, " Breaking out of second while loop">, #jsir<comment_line  <L 13 C 0>, <L 13 C 80>, 353, 433, " =============================================================================">, #jsir<comment_line  <L 22 C 0>, <L 22 C 80>, 514, 594, " =============================================================================">, #jsir<comment_line  <L 23 C 0>, <L 23 C 35>, 595, 630, " Breaking immediately after label">, #jsir<comment_line  <L 24 C 0>, <L 24 C 80>, 631, 711, " =============================================================================">]}> ({
// JSLIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSLIR-NEXT:     %0 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind WhileStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     "cf.br"()[^bb1] : () -> ()
// JSLIR-NEXT:   ^bb1:  // 2 preds: ^bb0, ^bb5
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
// JSLIR-NEXT:     "jsir.expression_statement"(%4) : (!jsir.any) -> ()
// JSLIR-NEXT:     %5 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSLIR-NEXT:     %6 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind IfStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     %7 = "builtin.unrealized_conversion_cast"(%5) : (!jsir.any) -> i1
// JSLIR-NEXT:     "cf.cond_br"(%7)[^bb3, ^bb5] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
// JSLIR-NEXT:   ^bb3:  // pred: ^bb2
// JSLIR-NEXT:     "jslir.control_flow_marker"(%6) <{kind = #jsir<cf_marker IfStatementConsequent>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.break_statement"() : () -> ()
// JSLIR-NEXT:     "cf.br"()[^bb6] : () -> ()
// JSLIR-NEXT:   ^bb4:  // no predecessors
// JSLIR-NEXT:     "cf.br"()[^bb5] : () -> ()
// JSLIR-NEXT:   ^bb5:  // 2 preds: ^bb2, ^bb4
// JSLIR-NEXT:     "jslir.control_flow_marker"(%6) <{kind = #jsir<cf_marker IfStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.control_flow_marker"(%3) <{kind = #jsir<cf_marker BlockStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "cf.br"()[^bb1] : () -> ()
// JSLIR-NEXT:   ^bb6:  // 2 preds: ^bb1, ^bb3
// JSLIR-NEXT:     "jslir.control_flow_marker"(%0) <{kind = #jsir<cf_marker WhileStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %8 = "jslir.labeled_statement_start"() <{label = #jsir<identifier   <L 15 C 0>, <L 15 C 6>, "label0", 435, 441, 0, "label0">}> : () -> !jsir.any
// JSLIR-NEXT:     %9 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind WhileStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     "cf.br"()[^bb7] : () -> ()
// JSLIR-NEXT:   ^bb7:  // 2 preds: ^bb6, ^bb14
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
// JSLIR-NEXT:     %14 = "jslir.labeled_statement_start"() <{label = #jsir<identifier   <L 17 C 2>, <L 17 C 8>, "label1", 462, 468, 4, "label1">}> : () -> !jsir.any
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
// JSLIR-NEXT:     "jslir.break_statement"() <{label = #jsir<identifier   <L 19 C 12>, <L 19 C 18>, "label0", 503, 509, 5, "label0">}> : () -> ()
// JSLIR-NEXT:     "cf.br"()[^bb15] : () -> ()
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
// JSLIR-NEXT:   ^bb15:  // 2 preds: ^bb7, ^bb11
// JSLIR-NEXT:     "jslir.control_flow_marker"(%9) <{kind = #jsir<cf_marker WhileStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.control_flow_marker"(%8) <{kind = #jsir<cf_marker LabeledStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %21 = "jslir.labeled_statement_start"() <{label = #jsir<identifier   <L 26 C 0>, <L 26 C 5>, "label", 713, 718, 0, "label">}> : () -> !jsir.any
// JSLIR-NEXT:     "jslir.break_statement"() <{label = #jsir<identifier   <L 26 C 13>, <L 26 C 18>, "label", 726, 731, 0, "label">}> : () -> ()
// JSLIR-NEXT:     "jslir.control_flow_marker"(%21) <{kind = #jsir<cf_marker LabeledStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:   }, {
// JSLIR-NEXT:   ^bb0:
// JSLIR-NEXT:   }) : () -> ()
// JSLIR-NEXT: }) : () -> ()
