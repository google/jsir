// JSHIR:      "jsir.file"() <{comments = []}> ({
// JSHIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSHIR-NEXT:     %0 = "jsir.identifier_ref"() <{name = "x"}> : () -> !jsir.any
// JSHIR-NEXT:     %1 = "jsir.arrow_function_expression"(%0) <{async = false, generator = false, operandSegmentSizes = array<i32: 0, 1>}> ({
// JSHIR-NEXT:       %4 = "jsir.identifier"() <{name = "y"}> : () -> !jsir.any
// JSHIR-NEXT:       "jsir.expr_region_end"(%4) : (!jsir.any) -> ()
// JSHIR-NEXT:     }) : (!jsir.any) -> !jsir.any
// JSHIR-NEXT:     "jsir.expression_statement"(%1) : (!jsir.any) -> ()
// JSHIR-NEXT:     %2 = "jsir.identifier_ref"() <{name = "x"}> : () -> !jsir.any
// JSHIR-NEXT:     %3 = "jsir.arrow_function_expression"(%2) <{async = false, generator = false, operandSegmentSizes = array<i32: 0, 1>}> ({
// JSHIR-NEXT:       "jshir.block_statement"() ({
// JSHIR-NEXT:         %4 = "jsir.identifier"() <{name = "y"}> : () -> !jsir.any
// JSHIR-NEXT:         "jsir.expression_statement"(%4) : (!jsir.any) -> ()
// JSHIR-NEXT:       }, {
// JSHIR-NEXT:       ^bb0:
// JSHIR-NEXT:       }) : () -> ()
// JSHIR-NEXT:     }) : (!jsir.any) -> !jsir.any
// JSHIR-NEXT:     "jsir.expression_statement"(%3) : (!jsir.any) -> ()
// JSHIR-NEXT:   }, {
// JSHIR-NEXT:   ^bb0:
// JSHIR-NEXT:   }) : () -> ()
// JSHIR-NEXT: }) : () -> ()
