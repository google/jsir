// JSHIR:      "jsir.file"() <{comments = []}> ({
// JSHIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSHIR-NEXT:     "jsir.variable_declaration"() <{kind = "let"}> ({
// JSHIR-NEXT:       %0 = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
// JSHIR-NEXT:       %1 = "jsir.none"() : () -> !jsir.any
// JSHIR-NEXT:       %2 = "jsir.identifier_ref"() <{name = "b"}> : () -> !jsir.any
// JSHIR-NEXT:       %3 = "jsir.array_pattern_ref"(%0, %1, %2) : (!jsir.any, !jsir.any, !jsir.any) -> !jsir.any
// JSHIR-NEXT:       %4 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSHIR-NEXT:       %5 = "jsir.variable_declarator"(%3, %4) : (!jsir.any, !jsir.any) -> !jsir.any
// JSHIR-NEXT:       "jsir.exprs_region_end"(%5) : (!jsir.any) -> ()
// JSHIR-NEXT:     }) : () -> ()
// JSHIR-NEXT:   }, {
// JSHIR-NEXT:   ^bb0:
// JSHIR-NEXT:   }) : () -> ()
// JSHIR-NEXT: }) : () -> ()
