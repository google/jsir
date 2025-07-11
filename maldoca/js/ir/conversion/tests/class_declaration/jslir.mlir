// JSLIR:      "jsir.file"() <{comments = []}> ({
// JSLIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSLIR-NEXT:     "jsir.class_declaration"() <{id = #jsir<identifier   <L 1 C 6>, <L 1 C 9>, "Foo", 6, 9, 1, "Foo">}> ({
// JSLIR-NEXT:       "jsir.class_body"() ({
// JSLIR-NEXT:         "jsir.class_property"() <{literal_key = #jsir<identifier   <L 2 C 2>, <L 2 C 21>, "property_identifier", 14, 33, 1, "property_identifier">, static_ = false}> ({
// JSLIR-NEXT:           %2 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "1", 1.000000e+00 : f64>, value = 1.000000e+00 : f64}> : () -> !jsir.any
// JSLIR-NEXT:           "jsir.expr_region_end"(%2) : (!jsir.any) -> ()
// JSLIR-NEXT:         }) : () -> ()
// JSLIR-NEXT:         "jsir.class_private_property"() <{key = #jsir<private_name   <L 3 C 2>, <L 3 C 24>, 41, 63, 1,    <L 3 C 3>, <L 3 C 24>, "property_private_name", 42, 63, 1, "property_private_name">, static_ = false}> ({
// JSLIR-NEXT:           %2 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "2", 2.000000e+00 : f64>, value = 2.000000e+00 : f64}> : () -> !jsir.any
// JSLIR-NEXT:           "jsir.expr_region_end"(%2) : (!jsir.any) -> ()
// JSLIR-NEXT:         }) : () -> ()
// JSLIR-NEXT:         "jsir.class_property"() <{literal_key = #jsir<string_literal   <L 4 C 2>, <L 4 C 27>, 71, 96, 1, "property_literal_string",  "\22property_literal_string\22", "property_literal_string">, static_ = false}> ({
// JSLIR-NEXT:           %2 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "3", 3.000000e+00 : f64>, value = 3.000000e+00 : f64}> : () -> !jsir.any
// JSLIR-NEXT:           "jsir.expr_region_end"(%2) : (!jsir.any) -> ()
// JSLIR-NEXT:         }) : () -> ()
// JSLIR-NEXT:         "jsir.class_property"() <{literal_key = #jsir<numeric_literal   <L 5 C 2>, <L 5 C 5>, 104, 107, 1, 1.000000e+00 : f64,  "1.0", 1.000000e+00 : f64>, static_ = false}> ({
// JSLIR-NEXT:           %2 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "4", 4.000000e+00 : f64>, value = 4.000000e+00 : f64}> : () -> !jsir.any
// JSLIR-NEXT:           "jsir.expr_region_end"(%2) : (!jsir.any) -> ()
// JSLIR-NEXT:         }) : () -> ()
// JSLIR-NEXT:         %0 = "jsir.string_literal"() <{extra = #jsir<string_literal_extra "\22property_computed\22", "property_computed">, value = "property_computed"}> : () -> !jsir.any
// JSLIR-NEXT:         "jsir.class_property"(%0) <{static_ = false}> ({
// JSLIR-NEXT:           %2 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "5", 5.000000e+00 : f64>, value = 5.000000e+00 : f64}> : () -> !jsir.any
// JSLIR-NEXT:           "jsir.expr_region_end"(%2) : (!jsir.any) -> ()
// JSLIR-NEXT:         }) : (!jsir.any) -> ()
// JSLIR-NEXT:         "jsir.class_method"() <{async = false, generator = false, kind = "method", literal_key = #jsir<identifier   <L 7 C 2>, <L 7 C 19>, "method_identifier", 144, 161, 1, "method_identifier">, operandSegmentSizes = array<i32: 0, 0>, static_ = false}> ({
// JSLIR-NEXT:           %2 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind BlockStatement>}> : () -> !jsir.any
// JSLIR-NEXT:           "jslir.control_flow_marker"(%2) <{kind = #jsir<cf_marker BlockStatementDirectives>}> : (!jsir.any) -> ()
// JSLIR-NEXT:           "jslir.control_flow_marker"(%2) <{kind = #jsir<cf_marker BlockStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:           "jslir.control_flow_marker"(%2) <{kind = #jsir<cf_marker BlockStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:         }) : () -> ()
// JSLIR-NEXT:         "jsir.class_private_method"() <{async = false, generator = false, key = #jsir<private_name   <L 8 C 2>, <L 8 C 22>, 169, 189, 1,    <L 8 C 3>, <L 8 C 22>, "method_private_name", 170, 189, 1, "method_private_name">, kind = "method", static_ = false}> ({
// JSLIR-NEXT:           %2 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind BlockStatement>}> : () -> !jsir.any
// JSLIR-NEXT:           "jslir.control_flow_marker"(%2) <{kind = #jsir<cf_marker BlockStatementDirectives>}> : (!jsir.any) -> ()
// JSLIR-NEXT:           "jslir.control_flow_marker"(%2) <{kind = #jsir<cf_marker BlockStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:           "jslir.control_flow_marker"(%2) <{kind = #jsir<cf_marker BlockStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:         }) : () -> ()
// JSLIR-NEXT:         "jsir.class_method"() <{async = false, generator = false, kind = "method", literal_key = #jsir<string_literal   <L 9 C 2>, <L 9 C 25>, 197, 220, 1, "method_literal_string",  "\22method_literal_string\22", "method_literal_string">, operandSegmentSizes = array<i32: 0, 0>, static_ = false}> ({
// JSLIR-NEXT:           %2 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind BlockStatement>}> : () -> !jsir.any
// JSLIR-NEXT:           "jslir.control_flow_marker"(%2) <{kind = #jsir<cf_marker BlockStatementDirectives>}> : (!jsir.any) -> ()
// JSLIR-NEXT:           "jslir.control_flow_marker"(%2) <{kind = #jsir<cf_marker BlockStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:           "jslir.control_flow_marker"(%2) <{kind = #jsir<cf_marker BlockStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:         }) : () -> ()
// JSLIR-NEXT:         "jsir.class_method"() <{async = false, generator = false, kind = "method", literal_key = #jsir<numeric_literal   <L 10 C 2>, <L 10 C 5>, 228, 231, 1, 1.000000e+00 : f64,  "1.0", 1.000000e+00 : f64>, operandSegmentSizes = array<i32: 0, 0>, static_ = false}> ({
// JSLIR-NEXT:           %2 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind BlockStatement>}> : () -> !jsir.any
// JSLIR-NEXT:           "jslir.control_flow_marker"(%2) <{kind = #jsir<cf_marker BlockStatementDirectives>}> : (!jsir.any) -> ()
// JSLIR-NEXT:           "jslir.control_flow_marker"(%2) <{kind = #jsir<cf_marker BlockStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:           "jslir.control_flow_marker"(%2) <{kind = #jsir<cf_marker BlockStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:         }) : () -> ()
// JSLIR-NEXT:         %1 = "jsir.string_literal"() <{extra = #jsir<string_literal_extra "\22method_computed\22", "method_computed">, value = "method_computed"}> : () -> !jsir.any
// JSLIR-NEXT:         "jsir.class_method"(%1) <{async = false, generator = false, kind = "method", operandSegmentSizes = array<i32: 0, 1>, static_ = false}> ({
// JSLIR-NEXT:           %2 = "jslir.control_flow_starter"() <{kind = #jsir<cf_kind BlockStatement>}> : () -> !jsir.any
// JSLIR-NEXT:           "jslir.control_flow_marker"(%2) <{kind = #jsir<cf_marker BlockStatementDirectives>}> : (!jsir.any) -> ()
// JSLIR-NEXT:           "jslir.control_flow_marker"(%2) <{kind = #jsir<cf_marker BlockStatementBody>}> : (!jsir.any) -> ()
// JSLIR-NEXT:           "jslir.control_flow_marker"(%2) <{kind = #jsir<cf_marker BlockStatementEnd>}> : (!jsir.any) -> ()
// JSLIR-NEXT:         }) : (!jsir.any) -> ()
// JSLIR-NEXT:       }) : () -> ()
// JSLIR-NEXT:     }) : () -> ()
// JSLIR-NEXT:   }, {
// JSLIR-NEXT:   ^bb0:
// JSLIR-NEXT:   }) : () -> ()
// JSLIR-NEXT: }) : () -> ()
