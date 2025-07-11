// JSLIR:      "jsir.file"() <{comments = []}> ({
// JSLIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSLIR-NEXT:     %0 = "jsir.identifier_ref"() <{name = "x"}> : () -> !jsir.any
// JSLIR-NEXT:     %1 = "jsir.arrow_function_expression"(%0) <{async = false, generator = false, operandSegmentSizes = array<i32: 0, 1>}> ({
// JSLIR-NEXT:       %4 = "jsir.identifier"() <{name = "y"}> : () -> !jsir.any
// JSLIR-NEXT:       "jsir.expr_region_end"(%4) : (!jsir.any) -> ()
// JSLIR-NEXT:     }) : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%1) : (!jsir.any) -> ()
// JSLIR-NEXT:     %2 = "jsir.identifier_ref"() <{name = "x"}> : () -> !jsir.any
// JSLIR-NEXT:     %3 = "jsir.arrow_function_expression"(%2) <{async = false, generator = false, operandSegmentSizes = array<i32: 0, 1>}> ({
// JSLIR-NEXT:       %4 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind BlockStatement>}> : () -> !jsir.any
// JSLIR-NEXT:       "jslir.control_flow_marker"(%4) <{kind = #jsir<cf_marker BlockStatementDirectives>}> : (!jsir.any) -> ()
// JSLIR-NEXT:       "jslir.control_flow_marker"(%4) <{kind = #jsir<cf_marker BlockStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:       %5 = "jsir.identifier"() <{name = "y"}> : () -> !jsir.any
// JSLIR-NEXT:       "jsir.expression_statement"(%5) : (!jsir.any) -> ()
// JSLIR-NEXT:       "jslir.control_flow_marker"(%4) <{kind = #jsir<cf_marker BlockStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:     }) : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%3) : (!jsir.any) -> ()
// JSLIR-NEXT:   }, {
// JSLIR-NEXT:   ^bb0:
// JSLIR-NEXT:   }) : () -> ()
// JSLIR-NEXT: }) : () -> ()
