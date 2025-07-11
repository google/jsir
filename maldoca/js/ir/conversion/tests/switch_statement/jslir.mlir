// JSLIR:      "jsir.file"() <{comments = []}> ({
// JSLIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSLIR-NEXT:     %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     %1 = "jslir.switch_statement_start"(%0) : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:     "cf.br"()[^bb1] : () -> ()
// JSLIR-NEXT:   ^bb1:  // pred: ^bb0
// JSLIR-NEXT:     %2 = "jslir.switch_statement_case_start"(%1) <{case_idx = 0 : ui32}> : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:     %3 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "0", 0.000000e+00 : f64>, value = 0.000000e+00 : f64}> : () -> !jsir.any
// JSLIR-NEXT:     "jslir.switch_statement_case_test"(%3) : (!jsir.any) -> ()
// JSLIR-NEXT:     %4 = "jsir.binary_expression"(%0, %3) <{operator_ = "==="}> : (!jsir.any, !jsir.any) -> !jsir.any
// JSLIR-NEXT:     %5 = "builtin.unrealized_conversion_cast"(%4) : (!jsir.any) -> i1
// JSLIR-NEXT:     "cf.cond_br"(%5)[^bb2, ^bb4] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
// JSLIR-NEXT:   ^bb2:  // pred: ^bb1
// JSLIR-NEXT:     "jslir.control_flow_marker"(%2) <{kind = #jsir<cf_marker SwitchStatementCaseBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %6 = "jsir.identifier"() <{name = "body0"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%6) : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.break_statement"() : () -> ()
// JSLIR-NEXT:     "cf.br"()[^bb8] : () -> ()
// JSLIR-NEXT:   ^bb3:  // no predecessors
// JSLIR-NEXT:     "cf.br"()[^bb5] : () -> ()
// JSLIR-NEXT:   ^bb4:  // pred: ^bb1
// JSLIR-NEXT:     %7 = "jslir.switch_statement_case_start"(%1) <{case_idx = 1 : ui32}> : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:     %8 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "1", 1.000000e+00 : f64>, value = 1.000000e+00 : f64}> : () -> !jsir.any
// JSLIR-NEXT:     "jslir.switch_statement_case_test"(%8) : (!jsir.any) -> ()
// JSLIR-NEXT:     %9 = "jsir.binary_expression"(%0, %8) <{operator_ = "==="}> : (!jsir.any, !jsir.any) -> !jsir.any
// JSLIR-NEXT:     %10 = "builtin.unrealized_conversion_cast"(%9) : (!jsir.any) -> i1
// JSLIR-NEXT:     "cf.cond_br"(%10)[^bb5, ^bb6] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
// JSLIR-NEXT:   ^bb5:  // 2 preds: ^bb3, ^bb4
// JSLIR-NEXT:     "jslir.control_flow_marker"(%7) <{kind = #jsir<cf_marker SwitchStatementCaseBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %11 = "jsir.identifier"() <{name = "body1"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%11) : (!jsir.any) -> ()
// JSLIR-NEXT:     "cf.br"()[^bb6] : () -> ()
// JSLIR-NEXT:   ^bb6:  // 2 preds: ^bb4, ^bb5
// JSLIR-NEXT:     "jslir.switch_statement_default_start"(%1) <{case_idx = 2 : ui32}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.break_statement"() : () -> ()
// JSLIR-NEXT:     "cf.br"()[^bb8] : () -> ()
// JSLIR-NEXT:   ^bb7:  // no predecessors
// JSLIR-NEXT:     "cf.br"()[^bb8] : () -> ()
// JSLIR-NEXT:   ^bb8:  // 3 preds: ^bb2, ^bb6, ^bb7
// JSLIR-NEXT:     "jslir.control_flow_marker"(%1) <{kind = #jsir<cf_marker SwitchStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %12 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     %13 = "jslir.switch_statement_start"(%12) : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:     "cf.br"()[^bb9] : () -> ()
// JSLIR-NEXT:   ^bb9:  // pred: ^bb8
// JSLIR-NEXT:     %14 = "jslir.switch_statement_case_start"(%13) <{case_idx = 0 : ui32}> : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:     %15 = "jsir.identifier"() <{name = "f"}> : () -> !jsir.any
// JSLIR-NEXT:     %16 = "jsir.call_expression"(%15) : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:     "jslir.switch_statement_case_test"(%16) : (!jsir.any) -> ()
// JSLIR-NEXT:     %17 = "jsir.binary_expression"(%12, %16) <{operator_ = "==="}> : (!jsir.any, !jsir.any) -> !jsir.any
// JSLIR-NEXT:     %18 = "builtin.unrealized_conversion_cast"(%17) : (!jsir.any) -> i1
// JSLIR-NEXT:     "cf.cond_br"(%18)[^bb10, ^bb13] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
// JSLIR-NEXT:   ^bb10:  // pred: ^bb9
// JSLIR-NEXT:     "jslir.control_flow_marker"(%14) <{kind = #jsir<cf_marker SwitchStatementCaseBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %19 = "jsir.identifier"() <{name = "body0"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%19) : (!jsir.any) -> ()
// JSLIR-NEXT:     "cf.br"()[^bb11] : () -> ()
// JSLIR-NEXT:   ^bb11:  // 2 preds: ^bb10, ^bb13
// JSLIR-NEXT:     "jslir.switch_statement_default_start"(%13) <{case_idx = 1 : ui32}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.break_statement"() : () -> ()
// JSLIR-NEXT:     "cf.br"()[^bb16] : () -> ()
// JSLIR-NEXT:   ^bb12:  // no predecessors
// JSLIR-NEXT:     "cf.br"()[^bb14] : () -> ()
// JSLIR-NEXT:   ^bb13:  // pred: ^bb9
// JSLIR-NEXT:     %20 = "jslir.switch_statement_case_start"(%13) <{case_idx = 2 : ui32}> : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:     %21 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "1", 1.000000e+00 : f64>, value = 1.000000e+00 : f64}> : () -> !jsir.any
// JSLIR-NEXT:     %22 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "1", 1.000000e+00 : f64>, value = 1.000000e+00 : f64}> : () -> !jsir.any
// JSLIR-NEXT:     %23 = "jsir.binary_expression"(%21, %22) <{operator_ = "+"}> : (!jsir.any, !jsir.any) -> !jsir.any
// JSLIR-NEXT:     "jslir.switch_statement_case_test"(%23) : (!jsir.any) -> ()
// JSLIR-NEXT:     %24 = "jsir.binary_expression"(%12, %23) <{operator_ = "==="}> : (!jsir.any, !jsir.any) -> !jsir.any
// JSLIR-NEXT:     %25 = "builtin.unrealized_conversion_cast"(%24) : (!jsir.any) -> i1
// JSLIR-NEXT:     "cf.cond_br"(%25)[^bb14, ^bb11] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
// JSLIR-NEXT:   ^bb14:  // 2 preds: ^bb12, ^bb13
// JSLIR-NEXT:     "jslir.control_flow_marker"(%20) <{kind = #jsir<cf_marker SwitchStatementCaseBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %26 = "jsir.identifier"() <{name = "body1"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%26) : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.break_statement"() : () -> ()
// JSLIR-NEXT:     "cf.br"()[^bb16] : () -> ()
// JSLIR-NEXT:   ^bb15:  // no predecessors
// JSLIR-NEXT:     "cf.br"()[^bb16] : () -> ()
// JSLIR-NEXT:   ^bb16:  // 3 preds: ^bb11, ^bb14, ^bb15
// JSLIR-NEXT:     "jslir.control_flow_marker"(%13) <{kind = #jsir<cf_marker SwitchStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %27 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     %28 = "jslir.switch_statement_start"(%27) : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:     "cf.br"()[^bb17] : () -> ()
// JSLIR-NEXT:   ^bb17:  // pred: ^bb16
// JSLIR-NEXT:     "jslir.control_flow_marker"(%28) <{kind = #jsir<cf_marker SwitchStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %29 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     %30 = "jslir.switch_statement_start"(%29) : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:     "cf.br"()[^bb18] : () -> ()
// JSLIR-NEXT:   ^bb18:  // pred: ^bb17
// JSLIR-NEXT:     %31 = "jslir.switch_statement_case_start"(%30) <{case_idx = 0 : ui32}> : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:     %32 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "0", 0.000000e+00 : f64>, value = 0.000000e+00 : f64}> : () -> !jsir.any
// JSLIR-NEXT:     "jslir.switch_statement_case_test"(%32) : (!jsir.any) -> ()
// JSLIR-NEXT:     %33 = "jsir.binary_expression"(%29, %32) <{operator_ = "==="}> : (!jsir.any, !jsir.any) -> !jsir.any
// JSLIR-NEXT:     %34 = "builtin.unrealized_conversion_cast"(%33) : (!jsir.any) -> i1
// JSLIR-NEXT:     "cf.cond_br"(%34)[^bb19, ^bb20] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
// JSLIR-NEXT:   ^bb19:  // pred: ^bb18
// JSLIR-NEXT:     "jslir.control_flow_marker"(%31) <{kind = #jsir<cf_marker SwitchStatementCaseBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "cf.br"()[^bb20] : () -> ()
// JSLIR-NEXT:   ^bb20:  // 2 preds: ^bb18, ^bb19
// JSLIR-NEXT:     "jslir.control_flow_marker"(%30) <{kind = #jsir<cf_marker SwitchStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:   }, {
// JSLIR-NEXT:   ^bb0:
// JSLIR-NEXT:   }) : () -> ()
// JSLIR-NEXT: }) : () -> ()
