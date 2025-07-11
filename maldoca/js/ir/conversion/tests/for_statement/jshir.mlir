// JSHIR:      "jsir.file"() <{comments = []}> ({
// JSHIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSHIR-NEXT:     "jshir.for_statement"() ({
// JSHIR-NEXT:       %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSHIR-NEXT:       "jsir.expr_region_end"(%0) : (!jsir.any) -> ()
// JSHIR-NEXT:     }, {
// JSHIR-NEXT:     }, {
// JSHIR-NEXT:     }, {
// JSHIR-NEXT:       %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSHIR-NEXT:       "jsir.expression_statement"(%0) : (!jsir.any) -> ()
// JSHIR-NEXT:     }) : () -> ()
// JSHIR-NEXT:     "jshir.for_statement"() ({
// JSHIR-NEXT:       "jsir.variable_declaration"() <{kind = "let"}> ({
// JSHIR-NEXT:         %0 = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
// JSHIR-NEXT:         %1 = "jsir.variable_declarator"(%0) : (!jsir.any) -> !jsir.any
// JSHIR-NEXT:         "jsir.exprs_region_end"(%1) : (!jsir.any) -> ()
// JSHIR-NEXT:       }) : () -> ()
// JSHIR-NEXT:     }, {
// JSHIR-NEXT:     }, {
// JSHIR-NEXT:     }, {
// JSHIR-NEXT:       %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSHIR-NEXT:       "jsir.expression_statement"(%0) : (!jsir.any) -> ()
// JSHIR-NEXT:     }) : () -> ()
// JSHIR-NEXT:     "jshir.for_statement"() ({
// JSHIR-NEXT:     }, {
// JSHIR-NEXT:       %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSHIR-NEXT:       "jsir.expr_region_end"(%0) : (!jsir.any) -> ()
// JSHIR-NEXT:     }, {
// JSHIR-NEXT:       %0 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSHIR-NEXT:       "jsir.expr_region_end"(%0) : (!jsir.any) -> ()
// JSHIR-NEXT:     }, {
// JSHIR-NEXT:       %0 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSHIR-NEXT:       "jsir.expression_statement"(%0) : (!jsir.any) -> ()
// JSHIR-NEXT:     }) : () -> ()
// JSHIR-NEXT:     "jshir.for_statement"() ({
// JSHIR-NEXT:     }, {
// JSHIR-NEXT:       %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSHIR-NEXT:       "jsir.expr_region_end"(%0) : (!jsir.any) -> ()
// JSHIR-NEXT:     }, {
// JSHIR-NEXT:     }, {
// JSHIR-NEXT:       %0 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSHIR-NEXT:       "jsir.expression_statement"(%0) : (!jsir.any) -> ()
// JSHIR-NEXT:     }) : () -> ()
// JSHIR-NEXT:   }, {
// JSHIR-NEXT:   ^bb0:
// JSHIR-NEXT:   }) : () -> ()
// JSHIR-NEXT: }) : () -> ()
