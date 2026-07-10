// JSIR:      "jsir.file"() <{comments = []}> ({
// JSIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSIR-NEXT:     "jsir.variable_declaration"() <{kind = "let"}> ({
// JSIR-NEXT:       %0 = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:       %1 = "jsir.none"() : () -> !jsir.any
// JSIR-NEXT:       %2 = "jsir.identifier_ref"() <{name = "b"}> : () -> !jsir.any
// JSIR-NEXT:       %3 = "jsir.array_pattern_ref"(%0, %1, %2) : (!jsir.any, !jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:       %4 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSIR-NEXT:       %5 = "jsir.variable_declarator"(%3, %4) : (!jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:       "jsir.exprs_region_end"(%5) : (!jsir.any) -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:   }, {
// JSIR-NEXT:   ^bb0:
// JSIR-NEXT:   }) : () -> ()
// JSIR-NEXT: }) : () -> ()
