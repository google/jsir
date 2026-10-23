// JSIR:      "jsir.file"() <{comments = []}> ({
// JSIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSIR-NEXT:     "jsir.variable_declaration"() <{kind = "var"}> ({
// JSIR-NEXT:       %0 = "jsir.identifier_ref"() <{name = "i"}> : () -> !jsir.any
// JSIR-NEXT:       %1 = "jsir.string_literal"() <{extra = #jsir<string_literal_extra "\22\\udf06\\ud834\22", "\EF\BF\BD\EF\BF\BD">, value = "\EF\BF\BD\EF\BF\BD"}> : () -> !jsir.any
// JSIR-NEXT:       %2 = "jsir.variable_declarator"(%0, %1) : (!jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:       "jsir.exprs_region_end"(%2) : (!jsir.any) -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:     "jsir.variable_declaration"() <{kind = "var"}> ({
// JSIR-NEXT:       %0 = "jsir.identifier_ref"() <{name = "j"}> : () -> !jsir.any
// JSIR-NEXT:       %1 = "jsir.template_element_value"() <{cooked = "\EF\BF\BD", raw = "\\ud800"}> : () -> !jsir.any
// JSIR-NEXT:       %2 = "jsir.template_element"(%1) <{tail = true}> : (!jsir.any) -> !jsir.any
// JSIR-NEXT:       %3 = "jsir.template_literal"(%2) <{operandSegmentSizes = array<i32: 1, 0>}> : (!jsir.any) -> !jsir.any
// JSIR-NEXT:       %4 = "jsir.variable_declarator"(%0, %3) : (!jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:       "jsir.exprs_region_end"(%4) : (!jsir.any) -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:     "jsir.variable_declaration"() <{kind = "var"}> ({
// JSIR-NEXT:       %0 = "jsir.identifier_ref"() <{name = "k"}> : () -> !jsir.any
// JSIR-NEXT:       %1 = "jsir.reg_exp_literal"() <{extra = #jsir<reg_exp_literal_extra "/[\\uD800-\\uDBFF]/">, flags = "", pattern = "[\\uD800-\\uDBFF]"}> : () -> !jsir.any
// JSIR-NEXT:       %2 = "jsir.variable_declarator"(%0, %1) : (!jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:       "jsir.exprs_region_end"(%2) : (!jsir.any) -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:   }, {
// JSIR-NEXT:   ^bb0:
// JSIR-NEXT:   }) : () -> ()
// JSIR-NEXT: }) : () -> ()
