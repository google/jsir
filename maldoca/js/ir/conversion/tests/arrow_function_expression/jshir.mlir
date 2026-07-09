// JSIR:      "jsir.file"() <{comments = []}> ({
// JSIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSIR-NEXT:     %0 = "jsir.identifier_ref"() <{name = "x"}> : () -> !jsir.any
// JSIR-NEXT:     %1 = "jsir.arrow_function_expression"(%0) <{async = false, generator = false, operandSegmentSizes = array<i32: 0, 1>}> ({
// JSIR-NEXT:       %4 = "jsir.identifier"() <{name = "y"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.expr_region_end"(%4) : (!jsir.any) -> ()
// JSIR-NEXT:     }) : (!jsir.any) -> !jsir.any
// JSIR-NEXT:     "jsir.expression_statement"(%1) : (!jsir.any) -> ()
// JSIR-NEXT:     %2 = "jsir.identifier_ref"() <{name = "x"}> : () -> !jsir.any
// JSIR-NEXT:     %3 = "jsir.arrow_function_expression"(%2) <{async = false, generator = false, operandSegmentSizes = array<i32: 0, 1>}> ({
// JSIR-NEXT:       "jshir.block_statement"() ({
// JSIR-NEXT:         %4 = "jsir.identifier"() <{name = "y"}> : () -> !jsir.any
// JSIR-NEXT:         "jsir.expression_statement"(%4) : (!jsir.any) -> ()
// JSIR-NEXT:       }, {
// JSIR-NEXT:       ^bb0:
// JSIR-NEXT:       }) : () -> ()
// JSIR-NEXT:     }) : (!jsir.any) -> !jsir.any
// JSIR-NEXT:     "jsir.expression_statement"(%3) : (!jsir.any) -> ()
// JSIR-NEXT:   }, {
// JSIR-NEXT:   ^bb0:
// JSIR-NEXT:   }) : () -> ()
// JSIR-NEXT: }) : () -> ()
