// JSHIR:      "jsir.file"() <{comments = []}> ({
// JSHIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSHIR-NEXT:     %0 = "jsir.identifier"() <{name = "raw"}> : () -> !jsir.any
// JSHIR-NEXT:     %1 = "jsir.template_element_value"() <{cooked = "42", raw = "42"}> : () -> !jsir.any
// JSHIR-NEXT:     %2 = "jsir.template_element"(%1) <{tail = true}> : (!jsir.any) -> !jsir.any
// JSHIR-NEXT:     %3 = "jsir.template_literal"(%2) <{operandSegmentSizes = array<i32: 1, 0>}> : (!jsir.any) -> !jsir.any
// JSHIR-NEXT:     %4 = "jsir.tagged_template_expression"(%0, %3) : (!jsir.any, !jsir.any) -> !jsir.any
// JSHIR-NEXT:     "jsir.expression_statement"(%4) : (!jsir.any) -> ()
// JSHIR-NEXT:   }, {
// JSHIR-NEXT:   ^bb0:
// JSHIR-NEXT:   }) : () -> ()
// JSHIR-NEXT: }) : () -> ()
