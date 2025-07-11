// JSLIR:      "jsir.file"() <{comments = []}> ({
// JSLIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSLIR-NEXT:     %0 = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     %1 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSLIR-NEXT:     %2 = "jslir.for_in_statement_start"(%0, %1) : (!jsir.any, !jsir.any) -> !jsir.any
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
// JSLIR-NEXT:     %8 = "jslir.for_in_statement_start"(%6, %7) <{left_declaration = #jsir<for_in_of_declaration   <L 4 C 5>, <L 4 C 10>, 24, 29, 2,   <L 4 C 9>, <L 4 C 10>, 28, 29, 2,  "a", 2, "let">}> : (!jsir.any, !jsir.any) -> !jsir.any
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
// JSLIR-NEXT:   }, {
// JSLIR-NEXT:   ^bb0:
// JSLIR-NEXT:   }) : () -> ()
// JSLIR-NEXT: }) : () -> ()
