// JSLIR:      "jsir.file"() <{comments = []}> ({
// JSLIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSLIR-NEXT:     %0 = "jsir.identifier"() <{name = "raw"}> : () -> !jsir.any
// JSLIR-NEXT:     %1 = "jsir.template_element_value"() <{cooked = "42", raw = "42"}> : () -> !jsir.any
// JSLIR-NEXT:     %2 = "jsir.template_element"(%1) <{tail = true}> : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:     %3 = "jsir.template_literal"(%2) <{operandSegmentSizes = array<i32: 1, 0>}> : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:     %4 = "jsir.tagged_template_expression"(%0, %3) : (!jsir.any, !jsir.any) -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%4) : (!jsir.any) -> ()
// JSLIR-NEXT:   }, {
// JSLIR-NEXT:   ^bb0:
// JSLIR-NEXT:   }) : () -> ()
// JSLIR-NEXT: }) : () -> ()
