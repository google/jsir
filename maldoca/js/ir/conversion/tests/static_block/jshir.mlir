// JSIR:      "jsir.file"() <{comments = []}> ({
// JSIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSIR-NEXT:     "jsir.class_declaration"() <{id = #jsir<identifier <L 1 C 6>, <L 1 C 9>, "Foo", 6, 9, 1, "Foo">}> ({
// JSIR-NEXT:       "jsir.class_body"() ({
// JSIR-NEXT:         "jsir.static_block"() ({
// JSIR-NEXT:           "jsir.variable_declaration"() <{kind = "let"}> ({
// JSIR-NEXT:             %0 = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:             %1 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "1", 1.000000e+00 : f64>, value = 1.000000e+00 : f64}> : () -> !jsir.any
// JSIR-NEXT:             %2 = "jsir.variable_declarator"(%0, %1) : (!jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:             "jsir.exprs_region_end"(%2) : (!jsir.any) -> ()
// JSIR-NEXT:           }) : () -> ()
// JSIR-NEXT:           "jsir.variable_declaration"() <{kind = "let"}> ({
// JSIR-NEXT:             %0 = "jsir.identifier_ref"() <{name = "b"}> : () -> !jsir.any
// JSIR-NEXT:             %1 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "2", 2.000000e+00 : f64>, value = 2.000000e+00 : f64}> : () -> !jsir.any
// JSIR-NEXT:             %2 = "jsir.variable_declarator"(%0, %1) : (!jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:             "jsir.exprs_region_end"(%2) : (!jsir.any) -> ()
// JSIR-NEXT:           }) : () -> ()
// JSIR-NEXT:         }) : () -> ()
// JSIR-NEXT:       }) : () -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:   }, {
// JSIR-NEXT:   ^bb0:
// JSIR-NEXT:   }) : () -> ()
// JSIR-NEXT: }) : () -> ()
