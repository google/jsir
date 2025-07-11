// JSLIR:      "jsir.file"() <{comments = []}> ({
// JSLIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSLIR-NEXT:     "jsir.variable_declaration"() <{kind = "let"}> ({
// JSLIR-NEXT:       %0 = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:       %1 = "jsir.none"() : () -> !jsir.any
// JSLIR-NEXT:       %2 = "jsir.identifier_ref"() <{name = "b"}> : () -> !jsir.any
// JSLIR-NEXT:       %3 = "jsir.array_pattern_ref"(%0, %1, %2) : (!jsir.any, !jsir.any, !jsir.any) -> !jsir.any
// JSLIR-NEXT:       %4 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSLIR-NEXT:       %5 = "jsir.variable_declarator"(%3, %4) : (!jsir.any, !jsir.any) -> !jsir.any
// JSLIR-NEXT:       "jsir.exprs_region_end"(%5) : (!jsir.any) -> ()
// JSLIR-NEXT:     }) : () -> ()
// JSLIR-NEXT:   }, {
// JSLIR-NEXT:   ^bb0:
// JSLIR-NEXT:   }) : () -> ()
// JSLIR-NEXT: }) : () -> ()
