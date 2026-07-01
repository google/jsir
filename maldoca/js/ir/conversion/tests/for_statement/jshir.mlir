// JSIR:      "jsir.file"() <{comments = []}> ({
// JSIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSIR-NEXT:     "jshir.for_statement"() ({
// JSIR-NEXT:       %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.expr_region_end"(%0) : (!jsir.any) -> ()
// JSIR-NEXT:     }, {
// JSIR-NEXT:     }, {
// JSIR-NEXT:     }, {
// JSIR-NEXT:       %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.expression_statement"(%0) : (!jsir.any) -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:     "jshir.for_statement"() ({
// JSIR-NEXT:       "jsir.variable_declaration"() <{kind = "let"}> ({
// JSIR-NEXT:         %0 = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:         %1 = "jsir.variable_declarator"(%0) : (!jsir.any) -> !jsir.any
// JSIR-NEXT:         "jsir.exprs_region_end"(%1) : (!jsir.any) -> ()
// JSIR-NEXT:       }) : () -> ()
// JSIR-NEXT:     }, {
// JSIR-NEXT:     }, {
// JSIR-NEXT:     }, {
// JSIR-NEXT:       %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.expression_statement"(%0) : (!jsir.any) -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:     "jshir.for_statement"() ({
// JSIR-NEXT:     }, {
// JSIR-NEXT:       %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.expr_region_end"(%0) : (!jsir.any) -> ()
// JSIR-NEXT:     }, {
// JSIR-NEXT:       %0 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.expr_region_end"(%0) : (!jsir.any) -> ()
// JSIR-NEXT:     }, {
// JSIR-NEXT:       %0 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.expression_statement"(%0) : (!jsir.any) -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:     "jshir.for_statement"() ({
// JSIR-NEXT:     }, {
// JSIR-NEXT:       %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.expr_region_end"(%0) : (!jsir.any) -> ()
// JSIR-NEXT:     }, {
// JSIR-NEXT:     }, {
// JSIR-NEXT:       %0 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.expression_statement"(%0) : (!jsir.any) -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:   }, {
// JSIR-NEXT:   ^bb0:
// JSIR-NEXT:   }) : () -> ()
// JSIR-NEXT: }) : () -> ()
