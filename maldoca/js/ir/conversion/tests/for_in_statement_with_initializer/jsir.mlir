// JSIR:      "jsir.file"() <{comments = []}> ({
// JSIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSIR-NEXT:     %0 = "jsir.identifier"() <{name = "obj"}> : () -> !jsir.any
// JSIR-NEXT:     "jshir.for_in_statement"(%0) ({
// JSIR-NEXT:       "jsir.variable_declaration"() <{kind = "var"}> ({
// JSIR-NEXT:         %1 = "jsir.identifier_ref"() <{name = "i"}> : () -> !jsir.any
// JSIR-NEXT:         %2 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "42", 4.200000e+01 : f64>, value = 4.200000e+01 : f64}> : () -> !jsir.any
// JSIR-NEXT:         %3 = "jsir.variable_declarator"(%1, %2) : (!jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:         "jsir.exprs_region_end"(%3) : (!jsir.any) -> ()
// JSIR-NEXT:       }) : () -> ()
// JSIR-NEXT:     }, {
// JSIR-NEXT:       "jshir.block_statement"() ({
// JSIR-NEXT:         %1 = "jsir.identifier"() <{name = "foo"}> : () -> !jsir.any
// JSIR-NEXT:         "jsir.expression_statement"(%1) : (!jsir.any) -> ()
// JSIR-NEXT:       }, {
// JSIR-NEXT:       ^bb0:
// JSIR-NEXT:       }) : () -> ()
// JSIR-NEXT:     }) : (!jsir.any) -> ()
// JSIR-NEXT:   }, {
// JSIR-NEXT:   ^bb0:
// JSIR-NEXT:   }) : () -> ()
// JSIR-NEXT: }) : () -> ()
