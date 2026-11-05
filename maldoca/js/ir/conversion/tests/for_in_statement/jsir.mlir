// JSIR:      "jsir.file"() <{comments = []}> ({
// JSIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSIR-NEXT:     %0 = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:     %1 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSIR-NEXT:     "jshir.for_in_statement"(%0, %1) ({
// JSIR-NEXT:     }, {
// JSIR-NEXT:       %3 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.expression_statement"(%3) : (!jsir.any) -> ()
// JSIR-NEXT:     }) : (!jsir.any, !jsir.any) -> ()
// JSIR-NEXT:     %2 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSIR-NEXT:     "jshir.for_in_statement"(%2) ({
// JSIR-NEXT:       "jsir.variable_declaration"() <{kind = "let"}> ({
// JSIR-NEXT:         %3 = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:         %4 = "jsir.variable_declarator"(%3) : (!jsir.any) -> !jsir.any
// JSIR-NEXT:         "jsir.exprs_region_end"(%4) : (!jsir.any) -> ()
// JSIR-NEXT:       }) : () -> ()
// JSIR-NEXT:     }, {
// JSIR-NEXT:       %3 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.expression_statement"(%3) : (!jsir.any) -> ()
// JSIR-NEXT:     }) : (!jsir.any) -> ()
// JSIR-NEXT:   }, {
// JSIR-NEXT:   ^bb0:
// JSIR-NEXT:   }) : () -> ()
// JSIR-NEXT: }) : () -> ()
