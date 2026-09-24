// JSIR:      "jsir.file"() <{comments = []}> ({
// JSIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSIR-NEXT:     %0 = "jsir.arrow_function_expression"() <{async = false, generator = false}> ({
// JSIR-NEXT:       %2 = "jsir.identifier_ref"() <{name = "x"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.exprs_region_end"(%2) : (!jsir.any) -> ()
// JSIR-NEXT:     }, {
// JSIR-NEXT:       %2 = "jsir.identifier"() <{name = "y"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.expr_region_end"(%2) : (!jsir.any) -> ()
// JSIR-NEXT:     }) : () -> !jsir.any
// JSIR-NEXT:     "jsir.expression_statement"(%0) : (!jsir.any) -> ()
// JSIR-NEXT:     %1 = "jsir.arrow_function_expression"() <{async = false, generator = false}> ({
// JSIR-NEXT:       %2 = "jsir.identifier_ref"() <{name = "x"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.exprs_region_end"(%2) : (!jsir.any) -> ()
// JSIR-NEXT:     }, {
// JSIR-NEXT:       "jshir.block_statement"() ({
// JSIR-NEXT:         %2 = "jsir.identifier"() <{name = "y"}> : () -> !jsir.any
// JSIR-NEXT:         "jsir.expression_statement"(%2) : (!jsir.any) -> ()
// JSIR-NEXT:       }, {
// JSIR-NEXT:       ^bb0:
// JSIR-NEXT:       }) : () -> ()
// JSIR-NEXT:     }) : () -> !jsir.any
// JSIR-NEXT:     "jsir.expression_statement"(%1) : (!jsir.any) -> ()
// JSIR-NEXT:   }, {
// JSIR-NEXT:   ^bb0:
// JSIR-NEXT:   }) : () -> ()
// JSIR-NEXT: }) : () -> ()
