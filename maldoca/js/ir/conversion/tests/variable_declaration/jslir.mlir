// JSLIR:      "jsir.file"() <{comments = []}> ({
// JSLIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSLIR-NEXT:     "jsir.variable_declaration"() <{kind = "let"}> ({
// JSLIR-NEXT:       %0 = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:       %1 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "0", 0.000000e+00 : f64>, value = 0.000000e+00 : f64}> : () -> !jsir.any
// JSLIR-NEXT:       %2 = "jsir.variable_declarator"(%0, %1) : (!jsir.any, !jsir.any) -> !jsir.any
// JSLIR-NEXT:       "jsir.exprs_region_end"(%2) : (!jsir.any) -> ()
// JSLIR-NEXT:     }) : () -> ()
// JSLIR-NEXT:     "jsir.variable_declaration"() <{kind = "var"}> ({
// JSLIR-NEXT:       %0 = "jsir.identifier_ref"() <{name = "b"}> : () -> !jsir.any
// JSLIR-NEXT:       %1 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "1", 1.000000e+00 : f64>, value = 1.000000e+00 : f64}> : () -> !jsir.any
// JSLIR-NEXT:       %2 = "jsir.variable_declarator"(%0, %1) : (!jsir.any, !jsir.any) -> !jsir.any
// JSLIR-NEXT:       "jsir.exprs_region_end"(%2) : (!jsir.any) -> ()
// JSLIR-NEXT:     }) : () -> ()
// JSLIR-NEXT:     "jsir.variable_declaration"() <{kind = "const"}> ({
// JSLIR-NEXT:       %0 = "jsir.identifier_ref"() <{name = "c"}> : () -> !jsir.any
// JSLIR-NEXT:       %1 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "2", 2.000000e+00 : f64>, value = 2.000000e+00 : f64}> : () -> !jsir.any
// JSLIR-NEXT:       %2 = "jsir.variable_declarator"(%0, %1) : (!jsir.any, !jsir.any) -> !jsir.any
// JSLIR-NEXT:       %3 = "jsir.identifier_ref"() <{name = "d"}> : () -> !jsir.any
// JSLIR-NEXT:       %4 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "3", 3.000000e+00 : f64>, value = 3.000000e+00 : f64}> : () -> !jsir.any
// JSLIR-NEXT:       %5 = "jsir.variable_declarator"(%3, %4) : (!jsir.any, !jsir.any) -> !jsir.any
// JSLIR-NEXT:       "jsir.exprs_region_end"(%2, %5) : (!jsir.any, !jsir.any) -> ()
// JSLIR-NEXT:     }) : () -> ()
// JSLIR-NEXT:   }, {
// JSLIR-NEXT:   ^bb0:
// JSLIR-NEXT:   }) : () -> ()
// JSLIR-NEXT: }) : () -> ()
