// JSLIR:      "jsir.file"() <{comments = []}> ({
// JSLIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSLIR-NEXT:     %0 = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     %1 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSLIR-NEXT:     %2 = "jslir.for_of_statement_start"(%0, %1) <{await = false}> : (!jsir.any, !jsir.any) -> !jsir.any
// JSLIR-NEXT:     "cf.br"()[^bb1] : () -> ()
// JSLIR-NEXT:   ^bb1:  // 2 preds: ^bb0, ^bb2
// JSLIR-NEXT:     %3 = "jslir.for_in_of_statement_has_next"(%2) : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:     %4 = "builtin.unrealized_conversion_cast"(%3) : (!jsir.any) -> i1
// JSLIR-NEXT:     "cf.cond_br"(%4)[^bb2, ^bb3] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
// JSLIR-NEXT:   ^bb2:  // pred: ^bb1
// JSLIR-NEXT:     "jslir.for_in_of_statement_get_next"(%2) : (!jsir.any) -> ()
// JSLIR-NEXT:     %5 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%5) : (!jsir.any) -> ()
// JSLIR-NEXT:     "cf.br"()[^bb1] : () -> ()
// JSLIR-NEXT:   ^bb3:  // pred: ^bb1
// JSLIR-NEXT:     "jslir.for_in_of_statement_end"(%2) : (!jsir.any) -> ()
// JSLIR-NEXT:     %6 = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     %7 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSLIR-NEXT:     %8 = "jslir.for_of_statement_start"(%6, %7) <{await = false, left_declaration = #jsir<for_in_of_declaration   <L 4 C 5>, <L 4 C 10>, 24, 29, 2,   <L 4 C 9>, <L 4 C 10>, 28, 29, 2,  "a", 2, "let">}> : (!jsir.any, !jsir.any) -> !jsir.any
// JSLIR-NEXT:     "cf.br"()[^bb4] : () -> ()
// JSLIR-NEXT:   ^bb4:  // 2 preds: ^bb3, ^bb5
// JSLIR-NEXT:     %9 = "jslir.for_in_of_statement_has_next"(%8) : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:     %10 = "builtin.unrealized_conversion_cast"(%9) : (!jsir.any) -> i1
// JSLIR-NEXT:     "cf.cond_br"(%10)[^bb5, ^bb6] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
// JSLIR-NEXT:   ^bb5:  // pred: ^bb4
// JSLIR-NEXT:     "jslir.for_in_of_statement_get_next"(%8) : (!jsir.any) -> ()
// JSLIR-NEXT:     %11 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%11) : (!jsir.any) -> ()
// JSLIR-NEXT:     "cf.br"()[^bb4] : () -> ()
// JSLIR-NEXT:   ^bb6:  // pred: ^bb4
// JSLIR-NEXT:     "jslir.for_in_of_statement_end"(%8) : (!jsir.any) -> ()
// JSLIR-NEXT:     %12 = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     %13 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSLIR-NEXT:     %14 = "jslir.for_of_statement_start"(%12, %13) <{await = false}> : (!jsir.any, !jsir.any) -> !jsir.any
// JSLIR-NEXT:     "cf.br"()[^bb7] : () -> ()
// JSLIR-NEXT:   ^bb7:  // 2 preds: ^bb6, ^bb8
// JSLIR-NEXT:     %15 = "jslir.for_in_of_statement_has_next"(%14) : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:     %16 = "builtin.unrealized_conversion_cast"(%15) : (!jsir.any) -> i1
// JSLIR-NEXT:     "cf.cond_br"(%16)[^bb8, ^bb9] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
// JSLIR-NEXT:   ^bb8:  // pred: ^bb7
// JSLIR-NEXT:     "jslir.for_in_of_statement_get_next"(%14) : (!jsir.any) -> ()
// JSLIR-NEXT:     %17 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind BlockStatement>}> : () -> !jsir.any
// JSLIR-NEXT:     "jslir.control_flow_marker"(%17) <{kind = #jsir<cf_marker BlockStatementDirectives>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.control_flow_marker"(%17) <{kind = #jsir<cf_marker BlockStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     %18 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%18) : (!jsir.any) -> ()
// JSLIR-NEXT:     "jslir.control_flow_marker"(%17) <{kind = #jsir<cf_marker BlockStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     "cf.br"()[^bb7] : () -> ()
// JSLIR-NEXT:   ^bb9:  // pred: ^bb7
// JSLIR-NEXT:     "jslir.for_in_of_statement_end"(%14) : (!jsir.any) -> ()
// JSLIR-NEXT:   }, {
// JSLIR-NEXT:   ^bb0:
// JSLIR-NEXT:   }) : () -> ()
// JSLIR-NEXT: }) : () -> ()
