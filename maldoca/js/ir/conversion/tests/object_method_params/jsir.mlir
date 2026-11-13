// JSIR:      "jsir.file"() <{comments = []}> ({
// JSIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSIR-NEXT:     "jsir.function_declaration"() <{async = false, generator = false, id = #jsir<identifier <L 1 C 9>, <L 1 C 19>, "sideEffect", 9, 19, 1, "sideEffect">}> ({
// JSIR-NEXT:       "jsir.exprs_region_end"() : () -> ()
// JSIR-NEXT:     }, {
// JSIR-NEXT:       "jshir.block_statement"() ({
// JSIR-NEXT:         %0 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "1", 1.000000e+00 : f64>, value = 1.000000e+00 : f64}> : () -> !jsir.any
// JSIR-NEXT:         "jsir.return_statement"(%0) : (!jsir.any) -> ()
// JSIR-NEXT:       }, {
// JSIR-NEXT:       ^bb0:
// JSIR-NEXT:       }) : () -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:     "jsir.variable_declaration"() <{kind = "const"}> ({
// JSIR-NEXT:       %0 = "jsir.identifier_ref"() <{name = "obj"}> : () -> !jsir.any
// JSIR-NEXT:       %1 = "jsir.object_expression"() ({
// JSIR-NEXT:         %3 = "jsir.object_method"() <{async = false, generator = false, kind = "method", literal_key = #jsir<identifier <L 5 C 2>, <L 5 C 8>, "method", 54, 60, 0, "method">}> ({
// JSIR-NEXT:           %8 = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:           %9 = "jsir.identifier_ref"() <{name = "b"}> : () -> !jsir.any
// JSIR-NEXT:           %10 = "jsir.identifier"() <{name = "sideEffect"}> : () -> !jsir.any
// JSIR-NEXT:           %11 = "jsir.call_expression"(%10) : (!jsir.any) -> !jsir.any
// JSIR-NEXT:           %12 = "jsir.assignment_pattern_ref"(%9, %11) : (!jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:           "jsir.exprs_region_end"(%8, %12) : (!jsir.any, !jsir.any) -> ()
// JSIR-NEXT:         }, {
// JSIR-NEXT:           "jshir.block_statement"() ({
// JSIR-NEXT:             %8 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:             %9 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSIR-NEXT:             %10 = "jsir.binary_expression"(%8, %9) <{operator_ = "+"}> : (!jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:             "jsir.return_statement"(%10) : (!jsir.any) -> ()
// JSIR-NEXT:           }, {
// JSIR-NEXT:           ^bb0:
// JSIR-NEXT:           }) : () -> ()
// JSIR-NEXT:         }) : () -> !jsir.any
// JSIR-NEXT:         %4 = "jsir.identifier"() <{name = "sideEffect"}> : () -> !jsir.any
// JSIR-NEXT:         %5 = "jsir.call_expression"(%4) : (!jsir.any) -> !jsir.any
// JSIR-NEXT:         %6 = "jsir.object_method"(%5) <{async = false, generator = false, kind = "method"}> ({
// JSIR-NEXT:           %8 = "jsir.identifier_ref"() <{name = "c"}> : () -> !jsir.any
// JSIR-NEXT:           %9 = "jsir.identifier_ref"() <{name = "d"}> : () -> !jsir.any
// JSIR-NEXT:           %10 = "jsir.identifier"() <{name = "sideEffect"}> : () -> !jsir.any
// JSIR-NEXT:           %11 = "jsir.call_expression"(%10) : (!jsir.any) -> !jsir.any
// JSIR-NEXT:           %12 = "jsir.assignment_pattern_ref"(%9, %11) : (!jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:           "jsir.exprs_region_end"(%8, %12) : (!jsir.any, !jsir.any) -> ()
// JSIR-NEXT:         }, {
// JSIR-NEXT:           "jshir.block_statement"() ({
// JSIR-NEXT:             %8 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSIR-NEXT:             %9 = "jsir.identifier"() <{name = "d"}> : () -> !jsir.any
// JSIR-NEXT:             %10 = "jsir.binary_expression"(%8, %9) <{operator_ = "+"}> : (!jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:             "jsir.return_statement"(%10) : (!jsir.any) -> ()
// JSIR-NEXT:           }, {
// JSIR-NEXT:           ^bb0:
// JSIR-NEXT:           }) : () -> ()
// JSIR-NEXT:         }) : (!jsir.any) -> !jsir.any
// JSIR-NEXT:         %7 = "jsir.object_method"() <{async = false, generator = false, kind = "get", literal_key = #jsir<identifier <L 11 C 6>, <L 11 C 10>, "prop", 176, 180, 0, "prop">}> ({
// JSIR-NEXT:           "jsir.exprs_region_end"() : () -> ()
// JSIR-NEXT:         }, {
// JSIR-NEXT:           "jshir.block_statement"() ({
// JSIR-NEXT:             %8 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "1", 1.000000e+00 : f64>, value = 1.000000e+00 : f64}> : () -> !jsir.any
// JSIR-NEXT:             "jsir.return_statement"(%8) : (!jsir.any) -> ()
// JSIR-NEXT:           }, {
// JSIR-NEXT:           ^bb0:
// JSIR-NEXT:           }) : () -> ()
// JSIR-NEXT:         }) : () -> !jsir.any
// JSIR-NEXT:         "jsir.exprs_region_end"(%3, %6, %7) : (!jsir.any, !jsir.any, !jsir.any) -> ()
// JSIR-NEXT:       }) : () -> !jsir.any
// JSIR-NEXT:       %2 = "jsir.variable_declarator"(%0, %1) : (!jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:       "jsir.exprs_region_end"(%2) : (!jsir.any) -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:   }, {
// JSIR-NEXT:   ^bb0:
// JSIR-NEXT:   }) : () -> ()
// JSIR-NEXT: }) : () -> ()
