// JSHIR:      "jsir.file"() <{comments = [#jsir<comment_line  <L 1 C 0>, <L 1 C 21>, 0, 21, " regular expression">, #jsir<comment_line  <L 4 C 0>, <L 4 C 7>, 28, 35, " null">, #jsir<comment_line  <L 7 C 0>, <L 7 C 9>, 42, 51, " string">, #jsir<comment_line  <L 10 C 0>, <L 10 C 10>, 58, 68, " boolean">, #jsir<comment_line  <L 13 C 0>, <L 13 C 9>, 76, 85, " number">, #jsir<comment_line  <L 16 C 0>, <L 16 C 10>, 90, 100, " big int">]}> ({
// JSHIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSHIR-NEXT:     %0 = "jsir.reg_exp_literal"() <{extra = #jsir<reg_exp_literal_extra "/1/">, flags = "", pattern = "1"}> : () -> !jsir.any
// JSHIR-NEXT:     "jsir.expression_statement"(%0) : (!jsir.any) -> ()
// JSHIR-NEXT:     %1 = "jsir.null_literal"() : () -> !jsir.any
// JSHIR-NEXT:     "jsir.expression_statement"(%1) : (!jsir.any) -> ()
// JSHIR-NEXT:     %2 = "jsir.string_literal"() <{extra = #jsir<string_literal_extra "'a'", "a">, value = "a"}> : () -> !jsir.any
// JSHIR-NEXT:     "jsir.expression_statement"(%2) : (!jsir.any) -> ()
// JSHIR-NEXT:     %3 = "jsir.boolean_literal"() <{value = true}> : () -> !jsir.any
// JSHIR-NEXT:     "jsir.expression_statement"(%3) : (!jsir.any) -> ()
// JSHIR-NEXT:     %4 = "jsir.numeric_literal"() <{extra = #jsir<numeric_literal_extra "1", 1.000000e+00 : f64>, value = 1.000000e+00 : f64}> : () -> !jsir.any
// JSHIR-NEXT:     "jsir.expression_statement"(%4) : (!jsir.any) -> ()
// JSHIR-NEXT:     %5 = "jsir.big_int_literal"() <{extra = #jsir<big_int_literal_extra "1n", "1">, value = "1"}> : () -> !jsir.any
// JSHIR-NEXT:     "jsir.expression_statement"(%5) : (!jsir.any) -> ()
// JSHIR-NEXT:   }, {
// JSHIR-NEXT:   ^bb0:
// JSHIR-NEXT:   }) : () -> ()
// JSHIR-NEXT: }) : () -> ()
