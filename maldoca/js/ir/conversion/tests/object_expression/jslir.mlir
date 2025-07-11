// JSLIR:      "jsir.file"() <{comments = []}> ({
// JSLIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSLIR-NEXT:     %0 = "jsir.object_expression"() ({
// JSLIR-NEXT:       %2 = "jsir.identifier"() <{name = "short_hand"}> : () -> !jsir.any
// JSLIR-NEXT:       %3 = "jsir.object_property"(%2) <{literal_key = #jsir<identifier   <L 2 C 2>, <L 2 C 12>, "short_hand", 5, 15, 0, "short_hand">, shorthand = true}> : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:       %4 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "1", 1.000000e+00 : f64>, value = 1.000000e+00 : f64}> : () -> !jsir.any
// JSLIR-NEXT:       %5 = "jsir.object_property"(%4) <{literal_key = #jsir<identifier   <L 3 C 2>, <L 3 C 21>, "property_identifier", 19, 38, 0, "property_identifier">, shorthand = false}> : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:       %6 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "2", 2.000000e+00 : f64>, value = 2.000000e+00 : f64}> : () -> !jsir.any
// JSLIR-NEXT:       %7 = "jsir.object_property"(%6) <{literal_key = #jsir<string_literal   <L 4 C 2>, <L 4 C 27>, 45, 70, 0, "property_string_literal",  "\22property_string_literal\22", "property_string_literal">, shorthand = false}> : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:       %8 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "3", 3.000000e+00 : f64>, value = 3.000000e+00 : f64}> : () -> !jsir.any
// JSLIR-NEXT:       %9 = "jsir.object_property"(%8) <{literal_key = #jsir<numeric_literal   <L 5 C 2>, <L 5 C 5>, 77, 80, 0, 1.000000e+00 : f64,  "1.0", 1.000000e+00 : f64>, shorthand = false}> : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:       %10 = "jsir.string_literal"() <{extra = #jsir<string_literal_extra "\22property_computed\22", "property_computed">, value = "property_computed"}> : () -> !jsir.any
// JSLIR-NEXT:       %11 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "4", 4.000000e+00 : f64>, value = 4.000000e+00 : f64}> : () -> !jsir.any
// JSLIR-NEXT:       %12 = "jsir.object_property"(%10, %11) <{shorthand = false}> : (!jsir.any, !jsir.any) -> !jsir.any
// JSLIR-NEXT:       %13 = "jsir.object_method"() <{async = false, generator = false, kind = "method", literal_key = #jsir<identifier   <L 7 C 2>, <L 7 C 19>, "method_identifier", 115, 132, 0, "method_identifier">, operandSegmentSizes = array<i32: 0, 0>}> ({
// JSLIR-NEXT:         %20 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind BlockStatement>}> : () -> !jsir.any
// JSLIR-NEXT:         "jslir.control_flow_marker"(%20) <{kind = #jsir<cf_marker BlockStatementDirectives>}> : (!jsir.any) -> ()
// JSLIR-NEXT:         "jslir.control_flow_marker"(%20) <{kind = #jsir<cf_marker BlockStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:         "jslir.control_flow_marker"(%20) <{kind = #jsir<cf_marker BlockStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:       }) : () -> !jsir.any
// JSLIR-NEXT:       %14 = "jsir.object_method"() <{async = false, generator = false, kind = "method", literal_key = #jsir<string_literal   <L 8 C 2>, <L 8 C 27>, 141, 166, 0, "property_string_literal",  "\22property_string_literal\22", "property_string_literal">, operandSegmentSizes = array<i32: 0, 0>}> ({
// JSLIR-NEXT:         %20 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind BlockStatement>}> : () -> !jsir.any
// JSLIR-NEXT:         "jslir.control_flow_marker"(%20) <{kind = #jsir<cf_marker BlockStatementDirectives>}> : (!jsir.any) -> ()
// JSLIR-NEXT:         "jslir.control_flow_marker"(%20) <{kind = #jsir<cf_marker BlockStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:         "jslir.control_flow_marker"(%20) <{kind = #jsir<cf_marker BlockStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:       }) : () -> !jsir.any
// JSLIR-NEXT:       %15 = "jsir.object_method"() <{async = false, generator = false, kind = "method", literal_key = #jsir<numeric_literal   <L 9 C 2>, <L 9 C 5>, 175, 178, 0, 1.000000e+00 : f64,  "1.0", 1.000000e+00 : f64>, operandSegmentSizes = array<i32: 0, 0>}> ({
// JSLIR-NEXT:         %20 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind BlockStatement>}> : () -> !jsir.any
// JSLIR-NEXT:         "jslir.control_flow_marker"(%20) <{kind = #jsir<cf_marker BlockStatementDirectives>}> : (!jsir.any) -> ()
// JSLIR-NEXT:         "jslir.control_flow_marker"(%20) <{kind = #jsir<cf_marker BlockStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:         "jslir.control_flow_marker"(%20) <{kind = #jsir<cf_marker BlockStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:       }) : () -> !jsir.any
// JSLIR-NEXT:       %16 = "jsir.string_literal"() <{extra = #jsir<string_literal_extra "\22property_computed\22", "property_computed">, value = "property_computed"}> : () -> !jsir.any
// JSLIR-NEXT:       %17 = "jsir.object_method"(%16) <{async = false, generator = false, kind = "method", operandSegmentSizes = array<i32: 1, 0>}> ({
// JSLIR-NEXT:         %20 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind BlockStatement>}> : () -> !jsir.any
// JSLIR-NEXT:         "jslir.control_flow_marker"(%20) <{kind = #jsir<cf_marker BlockStatementDirectives>}> : (!jsir.any) -> ()
// JSLIR-NEXT:         "jslir.control_flow_marker"(%20) <{kind = #jsir<cf_marker BlockStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:         "jslir.control_flow_marker"(%20) <{kind = #jsir<cf_marker BlockStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:       }) : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:       %18 = "jsir.identifier"() <{name = "spread_element"}> : () -> !jsir.any
// JSLIR-NEXT:       %19 = "jsir.spread_element"(%18) : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:       "jsir.exprs_region_end"(%3, %5, %7, %9, %12, %13, %14, %15, %17, %19) : (!jsir.any, !jsir.any, !jsir.any, !jsir.any, !jsir.any, !jsir.any, !jsir.any, !jsir.any, !jsir.any, !jsir.any) -> ()
// JSLIR-NEXT:     }) : () -> !jsir.any
// JSLIR-NEXT:     %1 = "jsir.parenthesized_expression"(%0) : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%1) : (!jsir.any) -> ()
// JSLIR-NEXT:   }, {
// JSLIR-NEXT:   ^bb0:
// JSLIR-NEXT:   }) : () -> ()
// JSLIR-NEXT: }) : () -> ()
