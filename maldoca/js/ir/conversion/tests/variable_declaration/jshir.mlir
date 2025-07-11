// JSHIR:      "jsir.file"() <{comments = []}> ({
// JSHIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSHIR-NEXT:     "jsir.variable_declaration"() <{kind = "let"}> ({
// JSHIR-NEXT:       %0 = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
// JSHIR-NEXT:       %1 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "0", 0.000000e+00 : f64>, value = 0.000000e+00 : f64}> : () -> !jsir.any
// JSHIR-NEXT:       %2 = "jsir.variable_declarator"(%0, %1) : (!jsir.any, !jsir.any) -> !jsir.any
// JSHIR-NEXT:       "jsir.exprs_region_end"(%2) : (!jsir.any) -> ()
// JSHIR-NEXT:     }) : () -> ()
// JSHIR-NEXT:     "jsir.variable_declaration"() <{kind = "var"}> ({
// JSHIR-NEXT:       %0 = "jsir.identifier_ref"() <{name = "b"}> : () -> !jsir.any
// JSHIR-NEXT:       %1 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "1", 1.000000e+00 : f64>, value = 1.000000e+00 : f64}> : () -> !jsir.any
// JSHIR-NEXT:       %2 = "jsir.variable_declarator"(%0, %1) : (!jsir.any, !jsir.any) -> !jsir.any
// JSHIR-NEXT:       "jsir.exprs_region_end"(%2) : (!jsir.any) -> ()
// JSHIR-NEXT:     }) : () -> ()
// JSHIR-NEXT:     "jsir.variable_declaration"() <{kind = "const"}> ({
// JSHIR-NEXT:       %0 = "jsir.identifier_ref"() <{name = "c"}> : () -> !jsir.any
// JSHIR-NEXT:       %1 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "2", 2.000000e+00 : f64>, value = 2.000000e+00 : f64}> : () -> !jsir.any
// JSHIR-NEXT:       %2 = "jsir.variable_declarator"(%0, %1) : (!jsir.any, !jsir.any) -> !jsir.any
// JSHIR-NEXT:       %3 = "jsir.identifier_ref"() <{name = "d"}> : () -> !jsir.any
// JSHIR-NEXT:       %4 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "3", 3.000000e+00 : f64>, value = 3.000000e+00 : f64}> : () -> !jsir.any
// JSHIR-NEXT:       %5 = "jsir.variable_declarator"(%3, %4) : (!jsir.any, !jsir.any) -> !jsir.any
// JSHIR-NEXT:       "jsir.exprs_region_end"(%2, %5) : (!jsir.any, !jsir.any) -> ()
// JSHIR-NEXT:     }) : () -> ()
// JSHIR-NEXT:   }, {
// JSHIR-NEXT:   ^bb0:
// JSHIR-NEXT:   }) : () -> ()
// JSHIR-NEXT: }) : () -> ()
