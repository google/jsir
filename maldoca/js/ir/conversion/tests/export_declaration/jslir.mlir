// JSLIR:      "jsir.file"() <{comments = [#jsir<comment_line  <L 9 C 0>, <L 9 C 36>, 139, 175, " TODO(b/182441574): Fix AST error.">, #jsir<comment_line  <L 10 C 0>, <L 10 C 39>, 176, 215, " export * as identifier_3 from \22foo\22;">]}> ({
// JSLIR-NEXT:   "jsir.program"() <{source_type = "module"}> ({
// JSLIR-NEXT:     "jsir.export_default_declaration"() ({
// JSLIR-NEXT:       %0 = "jsir.identifier"() <{name = "arbitrary_expression"}> : () -> !jsir.any
// JSLIR-NEXT:       "jsir.expr_region_end"(%0) : (!jsir.any) -> ()
// JSLIR-NEXT:     }) : () -> ()
// JSLIR-NEXT:     "jsir.export_named_declaration"() <{source = #jsir<string_literal   <L 3 C 45>, <L 3 C 50>, 83, 88, 0, "foo",  "\22foo\22", "foo">, specifiers = [#jsir<export_specifier   <L 3 C 8>, <L 3 C 20>, 46, 58, 0, #jsir<identifier   <L 3 C 8>, <L 3 C 20>, "identifier_1", 46, 58, 0, "identifier_1">, #jsir<identifier   <L 3 C 8>, <L 3 C 20>, "identifier_1", 46, 58, 0, "identifier_1">>, #jsir<export_specifier   <L 3 C 22>, <L 3 C 38>, 60, 76, 0, #jsir<string_literal   <L 3 C 22>, <L 3 C 38>, 60, 76, 0, "string_literal",  "\22string_literal\22", "string_literal">, #jsir<string_literal   <L 3 C 22>, <L 3 C 38>, 60, 76, 0, "string_literal",  "\22string_literal\22", "string_literal">>]}> ({
// JSLIR-NEXT:     }) : () -> ()
// JSLIR-NEXT:     "jsir.export_named_declaration"() <{specifiers = []}> ({
// JSLIR-NEXT:       "jsir.variable_declaration"() <{kind = "let"}> ({
// JSLIR-NEXT:         %0 = "jsir.identifier_ref"() <{name = "identifier_2"}> : () -> !jsir.any
// JSLIR-NEXT:         %1 = "jsir.variable_declarator"(%0) : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:         "jsir.exprs_region_end"(%1) : (!jsir.any) -> ()
// JSLIR-NEXT:       }) : () -> ()
// JSLIR-NEXT:     }) : () -> ()
// JSLIR-NEXT:     "jsir.export_all_declaration"() <{source = #jsir<string_literal   <L 7 C 14>, <L 7 C 19>, 131, 136, 0, "foo",  "\22foo\22", "foo">}> : () -> ()
// JSLIR-NEXT:   }, {
// JSLIR-NEXT:   ^bb0:
// JSLIR-NEXT:   }) : () -> ()
// JSLIR-NEXT: }) : () -> ()
