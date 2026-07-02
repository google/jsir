// JSIR:      "jsir.file"() <{comments = []}> ({
// JSIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSIR-NEXT:     %0 = "jsir.identifier"() <{name = "raw"}> : () -> !jsir.any
// JSIR-NEXT:     %1 = "jsir.template_element_value"() <{cooked = "42", raw = "42"}> : () -> !jsir.any
// JSIR-NEXT:     %2 = "jsir.template_element"(%1) <{tail = true}> : (!jsir.any) -> !jsir.any
// JSIR-NEXT:     %3 = "jsir.template_literal"(%2) <{operandSegmentSizes = array<i32: 1, 0>}> : (!jsir.any) -> !jsir.any
// JSIR-NEXT:     %4 = "jsir.tagged_template_expression"(%0, %3) : (!jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:     "jsir.expression_statement"(%4) : (!jsir.any) -> ()
// JSIR-NEXT:   }, {
// JSIR-NEXT:   ^bb0:
// JSIR-NEXT:   }) : () -> ()
// JSIR-NEXT: }) : () -> ()
