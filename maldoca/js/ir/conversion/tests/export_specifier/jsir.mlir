// JSIR:      "jsir.file"() <{comments = []}> ({
// JSIR-NEXT:   "jsir.program"() <{source_type = "module"}> ({
// JSIR-NEXT:     "jsir.variable_declaration"() <{kind = "let"}> ({
// JSIR-NEXT:       %0 = "jsir.identifier_ref"() <{name = "x"}> : () -> !jsir.any
// JSIR-NEXT:       %1 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "1", 1.000000e+00 : f64>, value = 1.000000e+00 : f64}> : () -> !jsir.any
// JSIR-NEXT:       %2 = "jsir.variable_declarator"(%0, %1) : (!jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:       "jsir.exprs_region_end"(%2) : (!jsir.any) -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:     "jsir.variable_declaration"() <{kind = "let"}> ({
// JSIR-NEXT:       %0 = "jsir.identifier_ref"() <{name = "y"}> : () -> !jsir.any
// JSIR-NEXT:       %1 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "2", 2.000000e+00 : f64>, value = 2.000000e+00 : f64}> : () -> !jsir.any
// JSIR-NEXT:       %2 = "jsir.variable_declarator"(%0, %1) : (!jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:       "jsir.exprs_region_end"(%2) : (!jsir.any) -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:     "jsir.export_named_declaration"() ({
// JSIR-NEXT:     }, {
// JSIR-NEXT:       %0 = "jsir.identifier_ref"() <{name = "x"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.export_specifier"(%0) <{exported = #jsir<identifier <L 3 C 8>, <L 3 C 9>, "x", 30, 31, 0, "x">}> : (!jsir.any) -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:     "jsir.export_named_declaration"() ({
// JSIR-NEXT:     }, {
// JSIR-NEXT:       %0 = "jsir.identifier_ref"() <{name = "x"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.export_specifier"(%0) <{exported = #jsir<identifier <L 4 C 13>, <L 4 C 14>, "a", 47, 48, 0, "a">}> : (!jsir.any) -> ()
// JSIR-NEXT:       %1 = "jsir.identifier_ref"() <{name = "y"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.export_specifier"(%1) <{exported = #jsir<identifier <L 4 C 21>, <L 4 C 22>, "b", 55, 56, 0, "b">}> : (!jsir.any) -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:     "jsir.export_named_declaration"() ({
// JSIR-NEXT:     }, {
// JSIR-NEXT:       %0 = "jsir.identifier_ref"() <{name = "x"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.export_specifier"(%0) <{exported = #jsir<string_literal <L 5 C 13>, <L 5 C 18>, 72, 77, 0, "a-b", "\22a-b\22", "a-b">}> : (!jsir.any) -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:     "jsir.export_named_declaration"() <{source = #jsir<string_literal <L 6 C 25>, <L 6 C 30>, 105, 110, 0, "foo", "\22foo\22", "foo">}> ({
// JSIR-NEXT:     }, {
// JSIR-NEXT:       %0 = "jsir.string_literal"() <{extra = #jsir<string_literal_extra "\22a-b\22", "a-b">, value = "a-b"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.export_specifier"(%0) <{exported = #jsir<identifier <L 6 C 17>, <L 6 C 18>, "c", 97, 98, 0, "c">}> : (!jsir.any) -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:     "jsir.export_named_declaration"() <{source = #jsir<string_literal <L 7 C 27>, <L 7 C 32>, 139, 144, 0, "foo", "\22foo\22", "foo">}> ({
// JSIR-NEXT:     }, {
// JSIR-NEXT:       %0 = "jsir.identifier_ref"() <{name = "default"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.export_specifier"(%0) <{exported = #jsir<identifier <L 7 C 19>, <L 7 C 20>, "d", 131, 132, 0, "d">}> : (!jsir.any) -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:     "jsir.export_named_declaration"() ({
// JSIR-NEXT:       "jsir.variable_declaration"() <{kind = "var"}> ({
// JSIR-NEXT:         %0 = "jsir.identifier_ref"() <{name = "e"}> : () -> !jsir.any
// JSIR-NEXT:         %1 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "1", 1.000000e+00 : f64>, value = 1.000000e+00 : f64}> : () -> !jsir.any
// JSIR-NEXT:         %2 = "jsir.variable_declarator"(%0, %1) : (!jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:         "jsir.exprs_region_end"(%2) : (!jsir.any) -> ()
// JSIR-NEXT:       }) : () -> ()
// JSIR-NEXT:     }, {
// JSIR-NEXT:     ^bb0:
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:   }, {
// JSIR-NEXT:   ^bb0:
// JSIR-NEXT:   }) : () -> ()
// JSIR-NEXT: }) : () -> ()
