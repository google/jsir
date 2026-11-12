// JSIR:      "jsir.file"() <{comments = []}> ({
// JSIR-NEXT:   "jsir.program"() <{source_type = "module"}> ({
// JSIR-NEXT:     "jsir.export_named_declaration"() <{source = #jsir<string_literal <L 1 C 20>, <L 1 C 25>, 20, 25, 0, "mod", "\22mod\22", "mod">}> ({
// JSIR-NEXT:     }, {
// JSIR-NEXT:       "jsir.export_namespace_specifier"() <{exported = #jsir<identifier <L 1 C 12>, <L 1 C 14>, "ns", 12, 14, 0, "ns">}> : () -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:     "jsir.export_named_declaration"() <{source = #jsir<string_literal <L 2 C 31>, <L 2 C 36>, 58, 63, 0, "mod", "\22mod\22", "mod">}> ({
// JSIR-NEXT:     }, {
// JSIR-NEXT:       "jsir.export_namespace_specifier"() <{exported = #jsir<string_literal <L 2 C 12>, <L 2 C 25>, 39, 52, 0, "string name", "\22string name\22", "string name">}> : () -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:     "jsir.export_named_declaration"() <{source = #jsir<string_literal <L 3 C 25>, <L 3 C 30>, 90, 95, 0, "mod", "\22mod\22", "mod">}> ({
// JSIR-NEXT:     }, {
// JSIR-NEXT:       "jsir.export_namespace_specifier"() <{exported = #jsir<identifier <L 3 C 12>, <L 3 C 19>, "default", 77, 84, 0, "default">}> : () -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:     "jsir.export_all_declaration"() <{source = #jsir<string_literal <L 4 C 14>, <L 4 C 19>, 111, 116, 0, "mod", "\22mod\22", "mod">}> : () -> ()
// JSIR-NEXT:     "jsir.export_named_declaration"() <{source = #jsir<string_literal <L 5 C 23>, <L 5 C 28>, 141, 146, 0, "mod", "\22mod\22", "mod">}> ({
// JSIR-NEXT:     }, {
// JSIR-NEXT:       %0 = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.export_specifier"(%0) <{exported = #jsir<identifier <L 5 C 14>, <L 5 C 15>, "b", 132, 133, 0, "b">}> : (!jsir.any) -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:   }, {
// JSIR-NEXT:   ^bb0:
// JSIR-NEXT:   }) : () -> ()
// JSIR-NEXT: }) : () -> ()
