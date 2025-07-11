// JSLIR:      "jsir.file"() <{comments = []}> ({
// JSLIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSLIR-NEXT:     %0 = "jsir.class_expression"() ({
// JSLIR-NEXT:       "jsir.class_body"() ({
// JSLIR-NEXT:         "jsir.class_property"() <{literal_key = #jsir<identifier   <L 2 C 2>, <L 2 C 21>, "property_identifier", 11, 30, 1, "property_identifier">, static_ = false}> ({
// JSLIR-NEXT:           %4 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "1", 1.000000e+00 : f64>, value = 1.000000e+00 : f64}> : () -> !jsir.any
// JSLIR-NEXT:           "jsir.expr_region_end"(%4) : (!jsir.any) -> ()
// JSLIR-NEXT:         }) : () -> ()
// JSLIR-NEXT:         "jsir.class_private_property"() <{key = #jsir<private_name   <L 3 C 2>, <L 3 C 24>, 38, 60, 1,    <L 3 C 3>, <L 3 C 24>, "property_private_name", 39, 60, 1, "property_private_name">, static_ = false}> ({
// JSLIR-NEXT:           %4 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "2", 2.000000e+00 : f64>, value = 2.000000e+00 : f64}> : () -> !jsir.any
// JSLIR-NEXT:           "jsir.expr_region_end"(%4) : (!jsir.any) -> ()
// JSLIR-NEXT:         }) : () -> ()
// JSLIR-NEXT:         "jsir.class_property"() <{literal_key = #jsir<string_literal   <L 4 C 2>, <L 4 C 27>, 68, 93, 1, "property_literal_string",  "\22property_literal_string\22", "property_literal_string">, static_ = false}> ({
// JSLIR-NEXT:           %4 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "3", 3.000000e+00 : f64>, value = 3.000000e+00 : f64}> : () -> !jsir.any
// JSLIR-NEXT:           "jsir.expr_region_end"(%4) : (!jsir.any) -> ()
// JSLIR-NEXT:         }) : () -> ()
// JSLIR-NEXT:         "jsir.class_property"() <{literal_key = #jsir<numeric_literal   <L 5 C 2>, <L 5 C 5>, 101, 104, 1, 1.000000e+00 : f64,  "1.0", 1.000000e+00 : f64>, static_ = false}> ({
// JSLIR-NEXT:           %4 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "4", 4.000000e+00 : f64>, value = 4.000000e+00 : f64}> : () -> !jsir.any
// JSLIR-NEXT:           "jsir.expr_region_end"(%4) : (!jsir.any) -> ()
// JSLIR-NEXT:         }) : () -> ()
// JSLIR-NEXT:         %2 = "jsir.string_literal"() <{extra = #jsir<string_literal_extra "\22property_computed\22", "property_computed">, value = "property_computed"}> : () -> !jsir.any
// JSLIR-NEXT:         "jsir.class_property"(%2) <{static_ = false}> ({
// JSLIR-NEXT:           %4 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "5", 5.000000e+00 : f64>, value = 5.000000e+00 : f64}> : () -> !jsir.any
// JSLIR-NEXT:           "jsir.expr_region_end"(%4) : (!jsir.any) -> ()
// JSLIR-NEXT:         }) : (!jsir.any) -> ()
// JSLIR-NEXT:         "jsir.class_method"() <{async = false, generator = false, kind = "method", literal_key = #jsir<identifier   <L 7 C 2>, <L 7 C 19>, "method_identifier", 141, 158, 1, "method_identifier">, operandSegmentSizes = array<i32: 0, 0>, static_ = false}> ({
// JSLIR-NEXT:           %4 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind BlockStatement>}> : () -> !jsir.any
// JSLIR-NEXT:           "jslir.control_flow_marker"(%4) <{kind = #jsir<cf_marker BlockStatementDirectives>}> : (!jsir.any) -> ()
// JSLIR-NEXT:           "jslir.control_flow_marker"(%4) <{kind = #jsir<cf_marker BlockStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:           "jslir.control_flow_marker"(%4) <{kind = #jsir<cf_marker BlockStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:         }) : () -> ()
// JSLIR-NEXT:         "jsir.class_private_method"() <{async = false, generator = false, key = #jsir<private_name   <L 8 C 2>, <L 8 C 22>, 166, 186, 1,    <L 8 C 3>, <L 8 C 22>, "method_private_name", 167, 186, 1, "method_private_name">, kind = "method", static_ = false}> ({
// JSLIR-NEXT:           %4 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind BlockStatement>}> : () -> !jsir.any
// JSLIR-NEXT:           "jslir.control_flow_marker"(%4) <{kind = #jsir<cf_marker BlockStatementDirectives>}> : (!jsir.any) -> ()
// JSLIR-NEXT:           "jslir.control_flow_marker"(%4) <{kind = #jsir<cf_marker BlockStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:           "jslir.control_flow_marker"(%4) <{kind = #jsir<cf_marker BlockStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:         }) : () -> ()
// JSLIR-NEXT:         "jsir.class_method"() <{async = false, generator = false, kind = "method", literal_key = #jsir<string_literal   <L 9 C 2>, <L 9 C 25>, 194, 217, 1, "method_literal_string",  "\22method_literal_string\22", "method_literal_string">, operandSegmentSizes = array<i32: 0, 0>, static_ = false}> ({
// JSLIR-NEXT:           %4 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind BlockStatement>}> : () -> !jsir.any
// JSLIR-NEXT:           "jslir.control_flow_marker"(%4) <{kind = #jsir<cf_marker BlockStatementDirectives>}> : (!jsir.any) -> ()
// JSLIR-NEXT:           "jslir.control_flow_marker"(%4) <{kind = #jsir<cf_marker BlockStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:           "jslir.control_flow_marker"(%4) <{kind = #jsir<cf_marker BlockStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:         }) : () -> ()
// JSLIR-NEXT:         "jsir.class_method"() <{async = false, generator = false, kind = "method", literal_key = #jsir<numeric_literal   <L 10 C 2>, <L 10 C 5>, 225, 228, 1, 1.000000e+00 : f64,  "1.0", 1.000000e+00 : f64>, operandSegmentSizes = array<i32: 0, 0>, static_ = false}> ({
// JSLIR-NEXT:           %4 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind BlockStatement>}> : () -> !jsir.any
// JSLIR-NEXT:           "jslir.control_flow_marker"(%4) <{kind = #jsir<cf_marker BlockStatementDirectives>}> : (!jsir.any) -> ()
// JSLIR-NEXT:           "jslir.control_flow_marker"(%4) <{kind = #jsir<cf_marker BlockStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:           "jslir.control_flow_marker"(%4) <{kind = #jsir<cf_marker BlockStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:         }) : () -> ()
// JSLIR-NEXT:         %3 = "jsir.string_literal"() <{extra = #jsir<string_literal_extra "\22method_computed\22", "method_computed">, value = "method_computed"}> : () -> !jsir.any
// JSLIR-NEXT:         "jsir.class_method"(%3) <{async = false, generator = false, kind = "method", operandSegmentSizes = array<i32: 0, 1>, static_ = false}> ({
// JSLIR-NEXT:           %4 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind BlockStatement>}> : () -> !jsir.any
// JSLIR-NEXT:           "jslir.control_flow_marker"(%4) <{kind = #jsir<cf_marker BlockStatementDirectives>}> : (!jsir.any) -> ()
// JSLIR-NEXT:           "jslir.control_flow_marker"(%4) <{kind = #jsir<cf_marker BlockStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:           "jslir.control_flow_marker"(%4) <{kind = #jsir<cf_marker BlockStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:         }) : (!jsir.any) -> ()
// JSLIR-NEXT:       }) : () -> ()
// JSLIR-NEXT:     }) : () -> !jsir.any
// JSLIR-NEXT:     %1 = "jsir.parenthesized_expression"(%0) : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%1) : (!jsir.any) -> ()
// JSLIR-NEXT:   }, {
// JSLIR-NEXT:   ^bb0:
// JSLIR-NEXT:   }) : () -> ()
// JSLIR-NEXT: }) : () -> ()
