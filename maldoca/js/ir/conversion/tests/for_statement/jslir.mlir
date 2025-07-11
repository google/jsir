// JSLIR:      "jsir.file"() <{comments = []}> ({
// JSLIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSLIR-NEXT:     %0 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind ForStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     "cf.br"()[^bb1] : () -> ()
// JSLIR-NEXT:   ^bb1:  // pred: ^bb0
// JSLIR-NEXT:     "jslir.control_flow_marker"(%0) <{kind = #jsir<cf_marker ForStatementInit>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %1 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     "cf.br"(%1)[^bb2] : (!jsir.any) -> ()
// JSLIR-NEXT:   ^bb2:  // 2 preds: ^bb1, ^bb2
// JSLIR-NEXT:     "jslir.control_flow_marker"(%0) <{kind = #jsir<cf_marker ForStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %2 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%2) : (!jsir.any) -> ()
// JSLIR-NEXT:     "cf.br"()[^bb2] : () -> ()
// JSLIR-NEXT:   ^bb3:  // no predecessors
// JSLIR-NEXT:     "jslir.control_flow_marker"(%0) <{kind = #jsir<cf_marker ForStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %3 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind ForStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     "cf.br"()[^bb4] : () -> ()
// JSLIR-NEXT:   ^bb4:  // pred: ^bb3
// JSLIR-NEXT:     "jslir.control_flow_marker"(%3) <{kind = #jsir<cf_marker ForStatementInit>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "jsir.variable_declaration"() <{kind = "let"}> ({
// JSLIR-NEXT:       %14 = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:       %15 = "jsir.variable_declarator"(%14) : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:       "jsir.exprs_region_end"(%15) : (!jsir.any) -> ()
// JSLIR-NEXT:     }) : () -> ()
// JSLIR-NEXT:     "cf.br"()[^bb5] : () -> ()
// JSLIR-NEXT:   ^bb5:  // 2 preds: ^bb4, ^bb5
// JSLIR-NEXT:     "jslir.control_flow_marker"(%3) <{kind = #jsir<cf_marker ForStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %4 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%4) : (!jsir.any) -> ()
// JSLIR-NEXT:     "cf.br"()[^bb5] : () -> ()
// JSLIR-NEXT:   ^bb6:  // no predecessors
// JSLIR-NEXT:     "jslir.control_flow_marker"(%3) <{kind = #jsir<cf_marker ForStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %5 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind ForStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     "cf.br"()[^bb7] : () -> ()
// JSLIR-NEXT:   ^bb7:  // 2 preds: ^bb6, ^bb9
// JSLIR-NEXT:     "jslir.control_flow_marker"(%5) <{kind = #jsir<cf_marker ForStatementTest>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %6 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     %7 = "builtin.unrealized_conversion_cast"(%6) : (!jsir.any) -> i1
// JSLIR-NEXT:     "cf.cond_br"(%7)[^bb8, ^bb10] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
// JSLIR-NEXT:   ^bb8:  // pred: ^bb7
// JSLIR-NEXT:     "jslir.control_flow_marker"(%5) <{kind = #jsir<cf_marker ForStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %8 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%8) : (!jsir.any) -> ()
// JSLIR-NEXT:     "cf.br"()[^bb9] : () -> ()
// JSLIR-NEXT:   ^bb9:  // pred: ^bb8
// JSLIR-NEXT:     "jslir.control_flow_marker"(%5) <{kind = #jsir<cf_marker ForStatementUpdate>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %9 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expr_region_end"(%9) : (!jsir.any) -> ()
// JSLIR-NEXT:     "cf.br"()[^bb7] : () -> ()
// JSLIR-NEXT:   ^bb10:  // pred: ^bb7
// JSLIR-NEXT:     "jslir.control_flow_marker"(%5) <{kind = #jsir<cf_marker ForStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %10 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind ForStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     "cf.br"()[^bb11] : () -> ()
// JSLIR-NEXT:   ^bb11:  // 2 preds: ^bb10, ^bb12
// JSLIR-NEXT:     "jslir.control_flow_marker"(%10) <{kind = #jsir<cf_marker ForStatementTest>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %11 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     %12 = "builtin.unrealized_conversion_cast"(%11) : (!jsir.any) -> i1
// JSLIR-NEXT:     "cf.cond_br"(%12)[^bb12, ^bb13] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
// JSLIR-NEXT:   ^bb12:  // pred: ^bb11
// JSLIR-NEXT:     "jslir.control_flow_marker"(%10) <{kind = #jsir<cf_marker ForStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %13 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%13) : (!jsir.any) -> ()
// JSLIR-NEXT:     "cf.br"()[^bb11] : () -> ()
// JSLIR-NEXT:   ^bb13:  // pred: ^bb11
// JSLIR-NEXT:     "jslir.control_flow_marker"(%10) <{kind = #jsir<cf_marker ForStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:   }, {
// JSLIR-NEXT:   ^bb0:
// JSLIR-NEXT:   }) : () -> ()
// JSLIR-NEXT: }) : () -> ()
